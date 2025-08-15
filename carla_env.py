"""
carla_env.py
=============

This module defines environments for training the policy network using
reinforcement learning. Two variants are provided:

* ``CarlaDrivingEnv`` interacts with the CARLA simulator to create
  realistic driving scenarios. It supports randomising road layouts,
  traffic, parked cars and intersections as described in the training
  specification. Each episode samples a new scenario and returns
  observations compatible with the policy network. Implementing the
  full CARLA-based environment requires CARLA 0.10.0 and openpilot's
  preprocessing pipeline; the skeleton provided here outlines the
  structure but leaves many details as TODOs.

* ``DummyDrivingEnv`` provides a lightweight, kinematic simulation for
  testing the training code when CARLA is unavailable. It simulates
  a vehicle following a sinusoidal lane centreline with random lane
  widths and orientation rates. The reward function mirrors the
  shaping described in the training plan: it penalises cross-track and
  heading error, rewards progress along the track and penalises large
  steering and acceleration commands. No vision model is used; instead,
  random feature vectors are returned via the openpilot wrapper.

Both environments implement the Gym API (compatible with gymnasium) and
produce observations comprising the history of vision features and
auxiliary inputs. The action space is continuous: an array of two
values representing curvature and acceleration. The desired curvature
and acceleration are applied directly to the simulated vehicle.
"""

from __future__ import annotations

import math
import random
import logging
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import carla  # type: ignore
    CARLA_AVAILABLE = True
except Exception:
    CARLA_AVAILABLE = False

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except Exception:
    GYM_AVAILABLE = False

from policy_model import PolicyConfig
from openpilot_wrapper import OpenPilotWrapper

logger = logging.getLogger(__name__)
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

class CarlaDrivingEnv:
    """CARLA-based environment for policy training.

    This environment connects to a CARLA 0.10.0 simulator instance and provides
    realistic driving scenarios for training the policy. It spawns vehicles,
    sets up cameras, and provides vision-based observations compatible with
    the openpilot policy network.
    """

    def __init__(
        self,
        openpilot: OpenPilotWrapper,
        cfg: PolicyConfig,
        seed: Optional[int] = None,
        host: str = "localhost",
        port: int = 2000,
        timeout: float = 10.0,
    ) -> None:
        assert CARLA_AVAILABLE and GYM_AVAILABLE, "CARLA and gym are required"
        self.openpilot = openpilot
        self.cfg = cfg
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # CARLA connection
        logger.info(f"Connecting to CARLA server at {host}:{port}")
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        
        # Test connection
        try:
            version = self.client.get_server_version()
            logger.info(f"Connected to CARLA server version: {version}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to CARLA server: {e}")
        
        # Domain randomization parameters (must be set before _setup_world)
        self.weather_presets = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.CloudyNoon, 
            carla.WeatherParameters.WetNoon,
            carla.WeatherParameters.ClearSunset,
            carla.WeatherParameters.CloudySunset,
        ]
        
        # World setup
        self.world = None
        self.original_settings = None
        self.map = None
        self._setup_world()
        self._setup_spaces()
        
        # Vehicle and sensors
        self.vehicle = None
        self.camera_main = None
        self.camera_extra = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        
        # State tracking
        self.spawn_points = []
        self.episode_steps = 0
        self.max_episode_steps = 1000  # 50 seconds at 20Hz
        self.collision_count = 0
        self.lane_invasion_count = 0
        
        # Camera data storage
        self.camera_data = {
            "main": None,
            "extra": None
        }
        
        # History buffers
        self.history = {
            "features": np.zeros((cfg.history_len, cfg.feature_len), dtype=np.float32),
            "desire": np.zeros((cfg.history_len, cfg.desire_len), dtype=np.float32),
            "prev_curv": np.zeros((cfg.history_len, cfg.prev_desired_curv_len), dtype=np.float32),
        }
        
        # Current desire state
        self.current_desire = np.zeros(cfg.desire_len, dtype=np.float32)
        self.target_waypoint = None
        
    def _setup_world(self):
        """Set up the CARLA world with appropriate settings."""
        # Load available maps and select one randomly
        available_maps = self.client.get_available_maps()
        town_maps = [m for m in available_maps if 'Town' in m]
        if not town_maps:
            town_maps = available_maps[:1]  # Use first available map
            
        selected_map = random.choice(town_maps)
        logger.info(f"Loading map: {selected_map}")
        
        self.world = self.client.load_world(selected_map)
        self.map = self.world.get_map()
        
        # Store original settings
        self.original_settings = self.world.get_settings()
        
        # Configure world settings for training
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20Hz
        settings.no_rendering_mode = False  # We need rendering for cameras
        self.world.apply_settings(settings)
        
        # Set random weather
        weather = random.choice(self.weather_presets)
        self.world.set_weather(weather)
        
        # Get spawn points
        self.spawn_points = self.map.get_spawn_points()
        if len(self.spawn_points) < 1:
            raise RuntimeError("No spawn points available on this map")
        
        logger.info(f"Map loaded: {len(self.spawn_points)} spawn points available")

    def _setup_spaces(self) -> None:
        """Initialise gym spaces for observations and actions."""
        # Observation is a dictionary of multiple arrays; we use a flat
        # continuous space for RL algorithm compatibility. The training code
        # will unpack it internally before feeding into the policy network.
        feat_shape = (self.cfg.history_len, self.cfg.feature_len)
        desire_shape = (self.cfg.history_len, self.cfg.desire_len)
        prev_shape = (self.cfg.history_len, self.cfg.prev_desired_curv_len)
        # Flatten all components into one vector
        obs_dim = (
            np.prod(feat_shape)
            + np.prod(desire_shape)
            + np.prod(prev_shape)
            + self.cfg.traffic_convention_len
            + self.cfg.lateral_control_params_len
        )
        high = np.inf * np.ones((obs_dim,), dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        # Action: desired curvature and desired acceleration
        self.action_space = spaces.Box(low=np.array([-1.0, -5.0]), high=np.array([1.0, 2.0]))

    def reset(self) -> np.ndarray:
        """Reset the environment for a new episode.

        Returns:
            Flattened observation vector.
        """
        logger.info("Resetting CARLA environment")
        
        # Clean up previous episode
        self._cleanup_actors()
        
        # Reset episode state
        self.episode_steps = 0
        self.collision_count = 0
        self.lane_invasion_count = 0
        
        # Domain randomization: change weather
        weather = random.choice(self.weather_presets)
        self.world.set_weather(weather)
        
        # Spawn ego vehicle
        self._spawn_ego_vehicle()
        
        # Set up sensors
        self._setup_sensors()
        
        # Clear history buffers
        self.history["features"].fill(0.0)
        self.history["desire"].fill(0.0) 
        self.history["prev_curv"].fill(0.0)
        
        # Reset desire state
        self.current_desire.fill(0.0)
        
        # Initialize history with a few frames
        for i in range(min(3, self.cfg.history_len)):
            # Step world to get initial camera data
            self.world.tick()
            
            # Wait for camera data
            self._wait_for_camera_data()
            
            # Extract vision features if we have camera data
            features = self._extract_vision_features()
            self.history["features"][-(i+1)] = features
        
        return self._compose_observation()
    
    def _spawn_ego_vehicle(self):
        """Spawn the ego vehicle at a random spawn point."""
        
        blueprint_library = self.world.get_blueprint_library()
        #vehicle_bp = blueprint_library.find('vehicle.mercedes.sprinter')  # Use Tesla Model 3
        vehicles = blueprint_library.filter("vehicle.*")
    
        vehicle_bp = vehicles[0]
        world = client.get_world()
        # Choose random spawn point
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
        #spawn_point = random.choice(self.spawn_points)
        
        # Try to spawn vehicle
        try:
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            logger.info(f"Spawned vehicle at {spawn_point.location}")
        except RuntimeError as e:
            # If spawn point is blocked, try a few more
            for i in range(10):
                spawn_point = random.choice(self.spawn_points)
                try:
                    self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                    logger.info(f"Spawned vehicle at {spawn_point.location} (attempt {i+2})")
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError("Failed to spawn vehicle after 10 attempts")
    
    def _setup_sensors(self):
        """Set up cameras and collision sensor on the ego vehicle."""
        blueprint_library = self.world.get_blueprint_library()
        
        # Main camera (front narrow FOV) - matches openpilot setup
        camera_bp_main = blueprint_library.find('sensor.camera.rgb')
        camera_bp_main.set_attribute('image_size_x', '1164')  # Openpilot camera resolution
        camera_bp_main.set_attribute('image_size_y', '874')
        camera_bp_main.set_attribute('fov', '20')  # Narrow FOV
        camera_bp_main.set_attribute('sensor_tick', '0.05')  # 20Hz
        
        # Wide camera (extra wide FOV) 
        camera_bp_extra = blueprint_library.find('sensor.camera.rgb')
        camera_bp_extra.set_attribute('image_size_x', '1164')
        camera_bp_extra.set_attribute('image_size_y', '874') 
        camera_bp_extra.set_attribute('fov', '120')  # Wide FOV
        camera_bp_extra.set_attribute('sensor_tick', '0.05')  # 20Hz
        
        # Camera positions (relative to vehicle center)
        main_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),  # Forward and up from vehicle center
            carla.Rotation(pitch=0, yaw=0, roll=0)
        )
        extra_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4), 
            carla.Rotation(pitch=0, yaw=0, roll=0)
        )
        
        # Spawn cameras
        self.camera_main = self.world.spawn_actor(camera_bp_main, main_transform, attach_to=self.vehicle)
        self.camera_extra = self.world.spawn_actor(camera_bp_extra, extra_transform, attach_to=self.vehicle)
        
        # Set up camera callbacks
        self.camera_main.listen(lambda image: self._on_camera_data(image, "main"))
        self.camera_extra.listen(lambda image: self._on_camera_data(image, "extra"))
        
        # Collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location(x=0, z=0))
        self.collision_sensor = self.world.spawn_actor(collision_bp, collision_transform, attach_to=self.vehicle)
        self.collision_sensor.listen(self._on_collision)
        
        # Lane invasion sensor
        lane_bp = blueprint_library.find('sensor.other.lane_invasion') 
        self.lane_invasion_sensor = self.world.spawn_actor(lane_bp, collision_transform, attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(self._on_lane_invasion)
        
        logger.info("Sensors set up successfully")
    
    def _on_camera_data(self, image, camera_type):
        """Callback for camera data."""
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # RGBA
        array = array[:, :, :3]  # Remove alpha channel -> RGB
        self.camera_data[camera_type] = array
    
    def _on_collision(self, event):
        """Callback for collision events."""
        self.collision_count += 1
        logger.warning(f"Collision detected with {event.other_actor.type_id}")
    
    def _on_lane_invasion(self, event):
        """Callback for lane invasion events."""
        self.lane_invasion_count += 1
        logger.info("Lane invasion detected")
    
    def _wait_for_camera_data(self, timeout=1.0):
        """Wait for camera data to be available."""
        import time
        start_time = time.time()
        while (self.camera_data["main"] is None or self.camera_data["extra"] is None):
            if time.time() - start_time > timeout:
                logger.warning("Timeout waiting for camera data")
                break
            time.sleep(0.01)
    
    def _extract_vision_features(self) -> np.ndarray:
        """Extract vision features using the openpilot wrapper."""
        if self.camera_data["main"] is None:
            # Return dummy features if no camera data
            return np.random.randn(self.cfg.feature_len).astype(np.float32) * 0.1
        
        # Convert camera images to the format expected by openpilot
        # CARLA images are in RGB format, but openpilot expects specific preprocessing
        main_img = self.camera_data["main"]
        extra_img = self.camera_data["extra"] if self.camera_data["extra"] is not None else self.camera_data["main"]
        
        # Simple preprocessing: normalize to 0-1 range and convert to float32
        main_img = (main_img.astype(np.float32) / 255.0)
        extra_img = (extra_img.astype(np.float32) / 255.0)
        
        imgs = {
            "road": main_img,
            "big_road": extra_img
        }
        
        try:
            features = self.openpilot.get_hidden_state(imgs)
            return np.array(features).astype(np.float32)
        except Exception as e:
            logger.warning(f"Vision feature extraction failed: {e}")
            return np.random.randn(self.cfg.feature_len).astype(np.float32) * 0.1
    
    def _cleanup_actors(self):
        """Clean up all spawned actors."""
        actors_to_destroy = []
        
        if self.vehicle is not None:
            actors_to_destroy.append(self.vehicle)
        if self.camera_main is not None:
            actors_to_destroy.append(self.camera_main)
        if self.camera_extra is not None:
            actors_to_destroy.append(self.camera_extra)
        if self.collision_sensor is not None:
            actors_to_destroy.append(self.collision_sensor)
        if self.lane_invasion_sensor is not None:
            actors_to_destroy.append(self.lane_invasion_sensor)
        
        for actor in actors_to_destroy:
            if actor is not None:
                try:
                    actor.destroy()
                except:
                    pass
        
        # Reset references
        self.vehicle = None
        self.camera_main = None
        self.camera_extra = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.camera_data = {"main": None, "extra": None}

    def _compose_observation(
        self, traffic_conv: np.ndarray, lat_params: np.ndarray
    ) -> np.ndarray:
        """Flatten history and side inputs into a single observation vector."""
        feat_flat = self.history["features"].reshape(-1)
        des_flat = self.history["desire"].reshape(-1)
        prev_flat = self.history["prev_curv"].reshape(-1)
        return np.concatenate([feat_flat, des_flat, prev_flat, traffic_conv, lat_params]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Advance the simulation by one timestep.

        Args:
            action: Array with desired curvature and desired acceleration.

        Returns:
            A tuple (obs, reward, done, info).
        """
        if self.vehicle is None:
            raise RuntimeError("Vehicle not spawned. Call reset() first.")
        
        self.episode_steps += 1
        
        # Extract and clip action
        desired_curvature = np.clip(float(action[0]), -1.0, 1.0)
        desired_acceleration = np.clip(float(action[1]), -5.0, 2.0)
        
        # Apply control to vehicle
        self._apply_control(desired_curvature, desired_acceleration)
        
        # Step CARLA world
        self.world.tick()
        
        # Wait for new camera data
        self._wait_for_camera_data()
        
        # Update history buffers
        self._update_history(desired_curvature)
        
        # Calculate reward
        reward, info = self._calculate_reward()
        
        # Check if episode is done
        done = self._check_done()
        
        # Get observation
        obs = self._compose_observation()
        
        return obs, reward, done, info
    
    def _apply_control(self, curvature: float, acceleration: float):
        """Apply curvature and acceleration control to the vehicle."""
        # Get current vehicle state
        velocity = self.vehicle.get_velocity()
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Convert curvature to steering angle
        # Simplified relationship: steering = curvature * speed_factor
        speed_factor = max(0.1, min(1.0, current_speed / 10.0))  # Scale steering by speed
        steering = curvature * speed_factor
        steering = np.clip(steering, -1.0, 1.0)
        
        # Convert acceleration to throttle/brake
        if acceleration >= 0:
            throttle = min(acceleration / 2.0, 1.0)  # Max accel 2.0 m/s²
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(-acceleration / 5.0, 1.0)  # Max brake 5.0 m/s²
        
        # Apply control
        control = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steering,
            hand_brake=False
        )
        self.vehicle.apply_control(control)
    
    def _update_history(self, curvature: float):
        """Update history buffers with new data."""
        # Shift history buffers
        self.history["features"][:-1] = self.history["features"][1:]
        self.history["desire"][:-1] = self.history["desire"][1:]
        self.history["prev_curv"][:-1] = self.history["prev_curv"][1:]
        
        # Add new data
        features = self._extract_vision_features()
        self.history["features"][-1] = features
        
        # Update desire state (simplified - just keep current desire)
        self._update_desire_state()
        self.history["desire"][-1] = self.current_desire.copy()
        
        # Store previous curvature
        self.history["prev_curv"][-1] = curvature
    
    def _update_desire_state(self):
        """Update the desire state based on navigation and maneuvers."""
        # For now, implement simple desire logic
        # In a full implementation, this would be based on planned route
        
        # Random desire pulse occasionally 
        if self.episode_steps % 100 == 0 and np.random.random() < 0.2:
            self.current_desire.fill(0.0)
            desire_type = np.random.choice([1, 2])  # Left or right lane change
            self.current_desire[desire_type] = 1.0
        else:
            # Decay current desire
            self.current_desire *= 0.95
            self.current_desire[self.current_desire < 0.1] = 0.0
    
    def _calculate_reward(self) -> Tuple[float, Dict]:
        """Calculate reward based on driving performance."""
        info = {}
        reward = 0.0
        
        if self.vehicle is None:
            return -100.0, info
        
        # Get vehicle state
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Get waypoint for lane-keeping reward
        waypoint = self.map.get_waypoint(transform.location)
        if waypoint is not None:
            # Cross-track error (distance from lane center)
            lane_center = waypoint.transform.location
            vehicle_location = transform.location
            
            # Calculate lateral distance from lane center
            lane_vector = waypoint.transform.get_forward_vector()
            to_vehicle = carla.Vector3D(
                vehicle_location.x - lane_center.x,
                vehicle_location.y - lane_center.y,
                0
            )
            
            # Cross product to get lateral distance
            cross_track_error = abs(
                to_vehicle.x * lane_vector.y - to_vehicle.y * lane_vector.x
            )
            
            # Heading error
            lane_heading = math.radians(waypoint.transform.rotation.yaw)
            vehicle_heading = math.radians(transform.rotation.yaw)
            heading_error = abs(math.atan2(
                math.sin(vehicle_heading - lane_heading),
                math.cos(vehicle_heading - lane_heading)
            ))
            
            info["cross_track_error"] = cross_track_error
            info["heading_error"] = heading_error
            
            # Lane keeping reward
            lane_keeping_reward = -2.0 * cross_track_error - 1.0 * heading_error
            reward += lane_keeping_reward
        
        # Speed reward (encourage maintaining reasonable speed)
        target_speed = 10.0  # m/s (36 km/h)
        speed_reward = -0.1 * abs(speed - target_speed)
        reward += speed_reward
        
        # Collision penalty
        if self.collision_count > 0:
            reward -= 100.0  # Large penalty for collision
            
        # Lane invasion penalty
        if self.lane_invasion_count > 0:
            reward -= 10.0
        
        # Small positive reward for making progress
        reward += 0.1
        
        info.update({
            "speed": speed,
            "collision_count": self.collision_count,
            "lane_invasion_count": self.lane_invasion_count,
            "episode_steps": self.episode_steps
        })
        
        return reward, info
    
    def _check_done(self) -> bool:
        """Check if episode should terminate."""
        # Episode done conditions
        if self.collision_count > 0:
            logger.info("Episode terminated: collision")
            return True
            
        if self.episode_steps >= self.max_episode_steps:
            logger.info("Episode terminated: time limit")
            return True
            
        # Check if vehicle is stuck (very low speed for too long)
        if self.vehicle is not None:
            velocity = self.vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            if speed < 0.5 and self.episode_steps > 1000:
                logger.info("Episode terminated: vehicle stuck")
                return True
        
        return False
    
    def _compose_observation(self) -> np.ndarray:
        """Compose observation matching openpilot policy interface."""
        # Traffic convention (randomly LHD/RHD)
        traffic_conv = np.array([1.0, 0.0], dtype=np.float32)  # LHD
        
        # Lateral control parameters
        if self.vehicle is not None:
            velocity = self.vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        else:
            speed = 10.0
            
        lat_delay = 0.15  # Realistic control latency
        lat_params = np.array([speed, lat_delay], dtype=np.float32)
        
        # Flatten history components
        feat_flat = self.history["features"].reshape(-1)
        des_flat = self.history["desire"].reshape(-1)
        prev_flat = self.history["prev_curv"].reshape(-1)
        
        # Combine all observation components
        obs = np.concatenate([
            feat_flat,
            des_flat,
            prev_flat,
            traffic_conv,
            lat_params,
        ], dtype=np.float32)
        
        return obs
    
    def close(self):
        """Clean up and close the environment."""
        logger.info("Closing CARLA environment")
        
        # Clean up actors
        self._cleanup_actors()
        
        # Restore original settings
        if self.original_settings is not None:
            try:
                self.world.apply_settings(self.original_settings)
            except:
                pass
                
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except:
            pass


class DummyDrivingEnv:
    """Simplified kinematic driving environment without CARLA.

    This environment is useful for verifying the training pipeline in the
    absence of the CARLA simulator. It simulates a vehicle following a
    randomly generated sinusoidal lane centreline. The state consists of
    the vehicle's lateral offset, heading error and speed. A simple
    bicycle model updates the vehicle given curvature and acceleration.
    Observations contain random vision features obtained from the
    openpilot wrapper along with the history of desires and previous
    curvature commands. The reward encourages staying near the centreline
    while maintaining comfort.
    """

    def __init__(self, openpilot: OpenPilotWrapper, cfg: PolicyConfig, seed: Optional[int] = None) -> None:
        assert GYM_AVAILABLE, "gymnasium is required"
        self.openpilot = openpilot
        self.cfg = cfg
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Vehicle state: [lateral_offset, heading_error, velocity]
        self.state = np.zeros(3, dtype=np.float32)
        self.dt = 0.05  # 20Hz to match openpilot
        
        # Randomized lane scenario parameters
        self._reset_scenario()
        
        # History buffers matching openpilot format
        self.history = {
            "features": np.zeros((cfg.history_len, cfg.feature_len), dtype=np.float32),
            "desire": np.zeros((cfg.history_len, cfg.desire_len), dtype=np.float32), 
            "prev_curv": np.zeros((cfg.history_len, cfg.prev_desired_curv_len), dtype=np.float32),
        }
        
        # Current desire pulse state
        self.current_desire = np.zeros(cfg.desire_len, dtype=np.float32)
        self.desire_timer = 0.0  # For pulse timing
        
        self.t = 0.0
        self.episode_steps = 0
        self.max_episode_steps = 1000  # 50 seconds at 20Hz
        self.done = False
        
        # Gym spaces
        obs_dim = (
            cfg.history_len * cfg.feature_len +
            cfg.history_len * cfg.desire_len +
            cfg.history_len * cfg.prev_desired_curv_len +
            cfg.traffic_convention_len +
            cfg.lateral_control_params_len
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(obs_dim, dtype=np.float32),
            high=np.inf * np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: [desired_curvature, desired_acceleration]
        # Curvature: reasonable range for road driving
        # Acceleration: -5 to +2 m/s² (strong braking to moderate accel)
        self.action_space = spaces.Box(
            low=np.array([-0.5, -5.0], dtype=np.float32),  # Max curvature ~= 1/2m radius
            high=np.array([0.5, 2.0], dtype=np.float32),
            dtype=np.float32
        )
        
    def _reset_scenario(self):
        """Reset scenario parameters for domain randomization."""
        # Sinusoidal lane parameters
        self.amp = np.random.uniform(0.3, 2.5)  # Lane curvature amplitude
        self.freq = np.random.uniform(0.005, 0.03)  # Spatial frequency  
        self.phase = np.random.uniform(0, 2 * np.pi)
        
        # Traffic convention (randomly LHD/RHD)
        self.is_rhd = np.random.choice([True, False])
        self.traffic_convention = np.array([0.0, 1.0] if self.is_rhd else [1.0, 0.0], dtype=np.float32)
        
        # Randomize physics parameters slightly
        self.friction_coeff = np.random.uniform(0.7, 1.2)
        self.mass_factor = np.random.uniform(0.8, 1.3)
        
        # Generate desire maneuver sequence
        self._generate_desire_sequence()

    def reset(self) -> np.ndarray:
        """Reset environment for new episode with domain randomization."""
        # Reset scenario parameters
        self._reset_scenario()
        
        # Reset vehicle state with small randomization
        self.state[0] = np.random.uniform(-0.5, 0.5)  # Small initial offset
        self.state[1] = np.random.uniform(-0.1, 0.1)  # Small initial heading error
        self.state[2] = np.random.uniform(8.0, 15.0)  # Initial velocity
        
        self.t = 0.0
        self.episode_steps = 0
        self.done = False
        
        # Clear history buffers
        self.history["features"].fill(0.0)
        self.history["desire"].fill(0.0)
        self.history["prev_curv"].fill(0.0)
        
        # Initialize with some random features to simulate startup
        for i in range(self.cfg.history_len):
            dummy_img = {"img": np.zeros((874, 1164), dtype=np.uint8)}
            self.history["features"][i] = self.openpilot.get_hidden_state(dummy_img)
        
        return self._compose_observation()

    def _generate_desire_sequence(self):
        """Generate desire pulse sequence for the episode."""
        # Randomly schedule desire pulses during episode
        self.desire_schedule = []
        if np.random.random() < 0.3:  # 30% chance of lane change
            # Schedule a lane change maneuver
            start_time = np.random.uniform(2.0, 8.0)  # Start between 2-8 seconds
            direction = np.random.choice([0, 1])  # Left or right
            self.desire_schedule.append((start_time, direction + 1))  # desire indices 1,2 for left/right
    
    def _update_desire(self):
        """Update desire pulse based on schedule."""
        # Reset previous desire
        self.current_desire.fill(0.0)
        
        # Check for scheduled desires
        for start_time, desire_idx in self.desire_schedule:
            if abs(self.t - start_time) < 0.1:  # Rising edge pulse
                self.current_desire[desire_idx] = 1.0
                break
    
    def _centreline(self, x: float) -> Tuple[float, float]:
        """Compute target path and heading for sinusoidal road."""
        y_target = self.amp * math.sin(2 * math.pi * self.freq * x + self.phase)
        dy_dx = 2 * math.pi * self.freq * self.amp * math.cos(2 * math.pi * self.freq * x + self.phase)
        target_heading = math.atan(dy_dx)
        return y_target, target_heading

    def _compose_observation(self) -> np.ndarray:
        """Compose observation matching openpilot policy interface."""
        # Lateral control parameters: [v_ego, lat_delay]
        lat_delay = np.random.uniform(0.1, 0.2)  # Realistic latency range
        lat_params = np.array([self.state[2], lat_delay], dtype=np.float32)
        
        # Flatten history components
        feat_flat = self.history["features"].reshape(-1)  # (history_len * feature_len,)
        des_flat = self.history["desire"].reshape(-1)    # (history_len * desire_len,)
        prev_flat = self.history["prev_curv"].reshape(-1) # (history_len * prev_desired_curv_len,)
        
        # Combine all observation components
        obs = np.concatenate([
            feat_flat,
            des_flat, 
            prev_flat,
            self.traffic_convention,  # (2,) - traffic convention
            lat_params,               # (2,) - lateral control params
        ], dtype=np.float32)
        
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step the kinematic driving simulation."""
        if self.done:
            return self.reset(), 0.0, True, {}
            
        self.episode_steps += 1
        curvature = np.clip(float(action[0]), -0.5, 0.5)  # Limit curvature
        acceleration = np.clip(float(action[1]), -5.0, 2.0)  # Limit acceleration
        
        # Current state
        lateral_offset, heading_error, velocity = self.state
        velocity = max(velocity, 0.1)  # Minimum velocity
        
        # Physics update with domain randomization
        effective_accel = acceleration * self.mass_factor
        velocity_new = velocity + effective_accel * self.dt * self.friction_coeff
        velocity_new = np.clip(velocity_new, 0.0, 35.0)  # 0-35 m/s max speed
        
        # Kinematic bicycle model
        # Heading rate from curvature command
        heading_rate = curvature * velocity
        heading_error_new = heading_error + heading_rate * self.dt
        
        # Lateral motion
        lateral_velocity = velocity * math.sin(heading_error)
        lateral_offset_new = lateral_offset + lateral_velocity * self.dt
        
        # Progress along road
        longitudinal_velocity = velocity * math.cos(heading_error)
        self.t += longitudinal_velocity * self.dt  # Distance-based progress
        
        # Target path at current position
        target_offset, target_heading = self._centreline(self.t)
        
        # Update desire pulse
        self._update_desire()
        
        # Compute reward components
        cte = lateral_offset_new - target_offset  # Cross-track error
        heading_delta = heading_error_new - target_heading  # Heading error
        
        # Multi-objective reward
        lane_keeping_reward = -abs(cte) - 0.5 * abs(heading_delta)
        comfort_penalty = -0.01 * (curvature**2 + acceleration**2)
        progress_reward = longitudinal_velocity * 0.1  # Encourage forward progress
        
        reward = lane_keeping_reward + comfort_penalty + progress_reward
        
        # Hard constraint violations (episode termination)
        violation = False
        if abs(cte) > 4.0:  # Off-road
            reward -= 100.0
            violation = True
        if velocity_new < 0.5 and self.episode_steps > 50:  # Stalled
            reward -= 50.0
            violation = True
        if self.episode_steps >= self.max_episode_steps:  # Time limit
            violation = True
        
        self.done = violation
        
        # Update state
        self.state[0] = lateral_offset_new
        self.state[1] = heading_error_new
        self.state[2] = velocity_new
        
        # Shift history buffers
        self.history["features"][:-1] = self.history["features"][1:]
        self.history["desire"][:-1] = self.history["desire"][1:]
        self.history["prev_curv"][:-1] = self.history["prev_curv"][1:]
        
        # Get new vision features (dummy for now)
        dummy_img = {"img": np.zeros((874, 1164), dtype=np.uint8)}
        self.history["features"][-1] = self.openpilot.get_hidden_state(dummy_img)
        
        # Store current desire and previous curvature
        self.history["desire"][-1] = self.current_desire.copy()
        self.history["prev_curv"][-1] = curvature
        
        obs = self._compose_observation()
        
        info = {
            "cte": float(cte),
            "heading_error": float(heading_delta), 
            "velocity": float(velocity_new),
            "episode_steps": self.episode_steps
        }
        
        return obs, float(reward), self.done, info

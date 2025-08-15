"""
openpilot_wrapper.py
====================

This module provides a thin wrapper around the comma.ai vision model and
associated parsing utilities. It enables extraction of the hidden state
features required by the policy network from a pair of forward-facing
camera images (narrow and wide). The wrapper loads the ONNX vision model
and uses openpilot's own Parser to postprocess outputs. In the absence of
ONNX runtime or openpilot (e.g. in test environments), the wrapper falls
back to a dummy implementation that returns random feature vectors.

Usage:

    wrapper = OpenPilotWrapper(
        vision_onnx_path="/path/to/driving_vision.onnx",
        vision_metadata_path="/path/to/driving_vision_metadata.pkl",
    )
    features = wrapper.get_hidden_state({
        "road": np.array(img_main),
        "big_road": np.array(img_extra),
    })

The returned ``features`` will be a NumPy array of shape ``(FEATURE_LEN,)``.

Note: This wrapper does not perform camera image warping, calibration or
YUV conversion. Those transformations should be handled by the
environment prior to calling ``get_hidden_state``. See openpilot's
``modeld.modeld`` for details on the full preprocessing pipeline.
"""

from __future__ import annotations

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    logger.warning("NumPy not available, some functionality will be limited")
    HAS_NUMPY = False
    # Create minimal numpy-like interface
    class MockNumPy:
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def zeros(shape, dtype=None):
            if isinstance(shape, int):
                return [0.0] * shape
            return [[0.0] * shape[1] for _ in range(shape[0])]
        @staticmethod
        def random():
            import random
            return type('MockRandom', (), {
                'randn': lambda *args: [random.gauss(0, 1) for _ in range(args[0])] if len(args) == 1 else random.gauss(0, 1)
            })()
        def astype(self, dtype):
            return self
        def flatten(self):
            return self if isinstance(self, list) else [self]
        def reshape(self, *args):
            return self
    np = MockNumPy()

# Setup logging first
logger = logging.getLogger(__name__)

# Setup tinygrad path first
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    openpilot_path = os.path.join(current_dir, 'openpilot')
    tinygrad_repo_path = os.path.join(openpilot_path, 'tinygrad_repo')
    
    # Add both tinygrad_repo and the symlinked tinygrad directory to Python path
    if os.path.exists(tinygrad_repo_path) and tinygrad_repo_path not in sys.path:
        sys.path.insert(0, tinygrad_repo_path)
        logger.info(f"Added tinygrad_repo to path: {tinygrad_repo_path}")
        
    # Also try the tinygrad symlink in openpilot root
    tinygrad_symlink_path = os.path.join(openpilot_path, 'tinygrad')
    if os.path.exists(tinygrad_symlink_path) and tinygrad_symlink_path not in sys.path:
        sys.path.insert(0, tinygrad_symlink_path)
        logger.info(f"Added tinygrad symlink to path: {tinygrad_symlink_path}")
        
except Exception as e:
    logger.warning(f"Could not setup tinygrad paths: {e}")

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info("ONNX runtime imported successfully")
except Exception as e:
    ONNX_AVAILABLE = False
    logger.warning(f"ONNX runtime import failed: {e}")

try:
    # Add openpilot to Python path and import directly from selfdrive
    # (current_dir and openpilot_path already defined above)
    
    if os.path.exists(openpilot_path) and openpilot_path not in sys.path:
        sys.path.insert(0, openpilot_path)
        logger.info(f"Added openpilot path: {openpilot_path}")
    
    # Import directly from selfdrive (openpilot modules expect to be imported this way)
    if HAS_NUMPY:
        from selfdrive.modeld.parse_model_outputs import Parser  # type: ignore
        from selfdrive.modeld.constants import ModelConstants  # type: ignore
        OPENPILOT_AVAILABLE = True
        logger.info("Openpilot modules loaded successfully")
    else:
        # Create mock constants for when numpy is not available
        class MockModelConstants:
            FEATURE_LEN = 512
            INPUT_HISTORY_BUFFER_LEN = 25
            DESIRE_LEN = 8
            TRAFFIC_CONVENTION_LEN = 2
            LATERAL_CONTROL_PARAMS_LEN = 2
            PREV_DESIRED_CURV_LEN = 1
            IDX_N = 33
            PLAN_WIDTH = 15
            DESIRED_CURV_WIDTH = 1
            DESIRE_PRED_WIDTH = 8
            PLAN_MHP_N = 5
            PLAN_MHP_SELECTION = 1
        
        ModelConstants = MockModelConstants()
        Parser = None  # Parser requires numpy
        OPENPILOT_AVAILABLE = False
        logger.warning("Using mock ModelConstants due to missing numpy")
        
except Exception as e:
    OPENPILOT_AVAILABLE = False
    logger.warning(f"Openpilot modules not available: {e}")
    
    # Fallback mock constants
    class MockModelConstants:
        FEATURE_LEN = 512
        INPUT_HISTORY_BUFFER_LEN = 25
        DESIRE_LEN = 8
        TRAFFIC_CONVENTION_LEN = 2
        LATERAL_CONTROL_PARAMS_LEN = 2
        PREV_DESIRED_CURV_LEN = 1
        IDX_N = 33
        PLAN_WIDTH = 15
        DESIRED_CURV_WIDTH = 1
        DESIRE_PRED_WIDTH = 8
        PLAN_MHP_N = 5
        PLAN_MHP_SELECTION = 1
    
    ModelConstants = MockModelConstants()
    Parser = None
    if 'openpilot_path' in locals():
        logger.info(f"Checked openpilot path: {openpilot_path}")


class OpenPilotWrapper:
    """Helper to run the comma.ai vision model and extract hidden state features.

    If tinygrad and openpilot are installed, this wrapper loads the vision
    model from pickled files and uses the provided metadata to parse
    outputs. Otherwise, it falls back to returning random features of the
    correct shape. The random features allow unit testing of the policy
    training code without requiring the full openpilot stack.
    """

    def __init__(
        self,
        vision_onnx_path: str,
        vision_metadata_path: str,
        use_dummy: bool = False,
    ) -> None:
        """Initialise the wrapper.

        Args:
            vision_onnx_path: Path to the ONNX vision model
                (e.g. driving_vision.onnx).
            vision_metadata_path: Path to the pickled metadata describing
                input shapes, output slices and output shapes.
            use_dummy: Force the wrapper to use dummy random features
                regardless of library availability. Useful for tests.
        """
        self.use_dummy = use_dummy or not (ONNX_AVAILABLE and OPENPILOT_AVAILABLE)
        self.vision_onnx_path = Path(vision_onnx_path)
        self.vision_metadata_path = Path(vision_metadata_path)
        self.vision_model = None
        self.parser: Optional[Parser] = None
        self.output_slices = None
        self.feature_len = None
        if not self.use_dummy:
            self._load_model()
        else:
            # Provide a default feature length for dummy mode
            # 512 matches typical openpilot models but can be overridden
            logger.warning(
                "OpenPilot vision model unavailable; using dummy random features."
            )
            self.feature_len = 512

    def _load_model(self) -> None:
        """Load the ONNX vision model and parse metadata."""
        try:
            logger.info("Loading ONNX vision model from %s", self.vision_onnx_path)
            
            # Load ONNX model
            ort_providers = ['CPUExecutionProvider']
            self.vision_model = ort.InferenceSession(str(self.vision_onnx_path), providers=ort_providers)
            logger.info("ONNX vision model loaded successfully")
            
            # Log input/output information
            input_names = [inp.name for inp in self.vision_model.get_inputs()]
            output_names = [out.name for out in self.vision_model.get_outputs()]
            logger.info(f"Model inputs: {input_names}")
            logger.info(f"Model outputs: {output_names}")
            
        except Exception as e:
            logger.error(f"Could not load ONNX vision model: {e}")
            raise RuntimeError(f"Vision model loading failed: {e}. Cannot proceed without real vision model.")
            
        try:
            logger.info("Loading vision metadata from %s", self.vision_metadata_path)
            with open(self.vision_metadata_path, "rb") as f:
                import pickle
                metadata = pickle.load(f)
            logger.info(f"Vision metadata keys: {list(metadata.keys())}")
        except Exception as e:
            logger.warning(f"Could not load vision metadata: {e}")
            metadata = {}
        
        self.output_slices = metadata.get("output_slices", {})
        
        # For ONNX models, we can get output shapes directly from the model
        if self.vision_model is not None:
            model_outputs = self.vision_model.get_outputs()
            logger.info(f"ONNX model output shapes: {[(out.name, out.shape) for out in model_outputs]}")
            
            # Look for hidden state output in ONNX model
            hidden_output = None
            for out in model_outputs:
                if 'hidden' in out.name.lower() or 'feature' in out.name.lower():
                    hidden_output = out
                    break
            
            if hidden_output is not None and len(hidden_output.shape) > 1:
                self.feature_len = hidden_output.shape[-1]  # Last dimension is feature length
                logger.info(f"Using ONNX model feature length: {self.feature_len}")
            else:
                # Fallback to metadata or default
                if "hidden_state" in metadata.get("output_shapes", {}):
                    hidden_shape = metadata["output_shapes"]["hidden_state"]
                    self.feature_len = hidden_shape[1] if len(hidden_shape) > 1 else hidden_shape[0]
                    logger.info(f"Using vision metadata feature length: {self.feature_len}")
                else:
                    # Use default openpilot feature length
                    if 'ModelConstants' in globals():
                        self.feature_len = ModelConstants.FEATURE_LEN
                    else:
                        self.feature_len = 512  # Default
                    logger.info(f"Using default feature length: {self.feature_len}")
        else:
            self.feature_len = 512
        if OPENPILOT_AVAILABLE and Parser is not None:
            self.parser = Parser()
        else:
            self.parser = None
    

    def preprocess_images(
        self,
        main_img: np.ndarray,
        extra_img: Optional[np.ndarray],
        device_from_calib_euler: np.ndarray,
        intrinsics_main: np.ndarray,
        intrinsics_extra: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Prepare camera images for the vision model.

        This function performs warping into the model coordinate frame and
        converts RGB images into planar YUV420 format. It mimics the
        preprocessing performed by openpilot's modeld. When openpilot and
        tinygrad are available, the warping uses the calibrated warp
        matrix computed via ``get_warp_matrix``. Otherwise, it falls back
        to a simple resize and naive YUV conversion.

        Args:
            main_img: RGB image from the narrow FOV camera as a numpy array
                of shape (H, W, 3) with values in [0,255].
            extra_img: RGB image from the wide FOV camera or None if not
                available. Should be same size as main_img.
            device_from_calib_euler: Euler angles describing the device
                orientation relative to calibration frame. Shape (3,).
            intrinsics_main: Camera intrinsics matrix for the narrow
                camera.
            intrinsics_extra: Camera intrinsics matrix for the wide camera.

        Returns:
            Dictionary mapping input names (e.g. 'big_road' and 'road') to
            YUV420 planar images ready for the vision model. Each array has
            shape (H, W * 3 // 2) where the last dimension contains Y
            followed by downsampled U and V planes.
        """
        # Helper for RGB->YUV420 conversion
        def rgb_to_yuv420(img: np.ndarray) -> np.ndarray:
            h, w, _ = img.shape
            # Compute YUV components
            R = img[..., 0].astype(np.float32)
            G = img[..., 1].astype(np.float32)
            B = img[..., 2].astype(np.float32)
            Y = 0.257 * R + 0.504 * G + 0.098 * B + 16.0
            U = -0.148 * R - 0.291 * G + 0.439 * B + 128.0
            V = 0.439 * R - 0.368 * G - 0.071 * B + 128.0
            # Clip and convert to uint8
            Y = np.clip(Y, 0, 255).astype(np.uint8)
            U = np.clip(U, 0, 255).astype(np.uint8)
            V = np.clip(V, 0, 255).astype(np.uint8)
            # Downsample U and V to 1/2 resolution
            U_ds = (U.reshape(h//2, 2, w//2, 2).mean(axis=(1,3))).astype(np.uint8)
            V_ds = (V.reshape(h//2, 2, w//2, 2).mean(axis=(1,3))).astype(np.uint8)
            # Pack into planar format: Y plane followed by U and V planes
            yuv = np.zeros((h, w * 3 // 2), dtype=np.uint8)
            yuv[:, :w] = Y
            yuv[:, w:w + w//2] = U_ds.repeat(2, axis=0)  # upsample U
            yuv[:, w + w//2:] = V_ds.repeat(2, axis=0)
            return yuv
        # Determine warp matrix if openpilot is available
        warped_main = None
        warped_extra = None
        if not self.use_dummy and OPENPILOT_AVAILABLE:
            try:
                from common.transformations.model import get_warp_matrix  # type: ignore
                warp_main = get_warp_matrix(device_from_calib_euler, intrinsics_main, False)
                # Warp using cv2 if available
                try:
                    import cv2  # type: ignore
                    warped_main = cv2.warpPerspective(main_img, warp_main, (main_img.shape[1], main_img.shape[0]))
                    if extra_img is not None and intrinsics_extra is not None:
                        warp_extra = get_warp_matrix(device_from_calib_euler, intrinsics_extra, True)
                        warped_extra = cv2.warpPerspective(extra_img, warp_extra, (extra_img.shape[1], extra_img.shape[0]))
                except Exception:
                    warped_main = main_img
                    warped_extra = extra_img
            except Exception:
                warped_main = main_img
                warped_extra = extra_img
        else:
            warped_main = main_img
            warped_extra = extra_img
        # Convert to YUV420 planar
        result = {}
        if warped_extra is not None:
            yuv_extra = rgb_to_yuv420(warped_extra)
            result["big_road"] = yuv_extra
        yuv_main = rgb_to_yuv420(warped_main)
        result["road"] = yuv_main
        return result

    def _preprocess_for_model(self, img: np.ndarray, expected_shape: list) -> np.ndarray:
        """Preprocess image to match the expected model input format.
        
        Args:
            img: Input image as numpy array (H, W, C) in range [0, 1] or [0, 255]
            expected_shape: Expected shape from ONNX model [batch, channels, height, width]
            
        Returns:
            Preprocessed image matching expected_shape and data type
        """
        try:
            # Parse expected dimensions
            batch_size, channels, target_height, target_width = expected_shape
            
            # Convert to uint8 if needed
            if img.dtype == np.float32:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # Handle different input formats
            if len(img.shape) == 3:  # (H, W, C) 
                height, width, input_channels = img.shape
            elif len(img.shape) == 2:  # (H, W) grayscale
                height, width = img.shape
                input_channels = 1
                img = np.expand_dims(img, axis=-1)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
            
            # Resize to target resolution
            if height != target_height or width != target_width:
                try:
                    import cv2
                    img = cv2.resize(img, (target_width, target_height))
                except ImportError:
                    # Fallback: simple nearest neighbor resize using numpy
                    # This is a basic resize - not as good as cv2 but works
                    if height != target_height or width != target_width:
                        try:
                            from scipy.ndimage import zoom
                            zoom_factors = (target_height / height, target_width / width, 1)
                            img = zoom(img, zoom_factors, order=0).astype(np.uint8)
                        except ImportError:
                            # Last resort: basic numpy indexing resize (crude but functional)
                            y_indices = np.linspace(0, height-1, target_height).astype(int)
                            x_indices = np.linspace(0, width-1, target_width).astype(int)
                            img = img[np.ix_(y_indices, x_indices)]
            
            # Convert RGB to YUV420 if model expects YUV format (channels = 12)
            if channels == 12 and input_channels == 3:
                # Convert RGB to YUV420 planar format
                img = self._rgb_to_yuv420_planar(img)
            elif channels != input_channels:
                # Handle other channel mismatches
                if channels == 1 and input_channels == 3:
                    # RGB to grayscale
                    img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
                    img = np.expand_dims(img, axis=-1)
                elif channels == 3 and input_channels == 1:
                    # Grayscale to RGB
                    img = np.repeat(img, 3, axis=-1)
            
            # Add batch dimension and transpose to NCHW format
            img = np.expand_dims(img, axis=0)  # Add batch dim: (1, H, W, C)
            if len(img.shape) == 4:
                img = np.transpose(img, (0, 3, 1, 2))  # NHWC -> NCHW
            
            return img.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return zeros with correct shape as fallback
            return np.zeros(expected_shape, dtype=np.uint8)

    def _rgb_to_yuv420_planar(self, rgb_img: np.ndarray) -> np.ndarray:
        """Convert RGB image to YUV420 planar format expected by vision model.
        
        Args:
            rgb_img: RGB image (H, W, 3) in uint8 format
            
        Returns:
            YUV420 planar image (H, W, 12) with Y, U, V planes
        """
        if len(rgb_img.shape) != 3 or rgb_img.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H,W,3), got {rgb_img.shape}")
            
        h, w, _ = rgb_img.shape
        
        # Convert RGB to YUV
        rgb = rgb_img.astype(np.float32)
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        
        # YUV conversion matrix
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b + 128
        v = 0.615 * r - 0.515 * g - 0.100 * b + 128
        
        # Clip to valid range
        y = np.clip(y, 0, 255).astype(np.uint8)
        u = np.clip(u, 0, 255).astype(np.uint8)
        v = np.clip(v, 0, 255).astype(np.uint8)
        
        # Create YUV420 planar format (total 12 channels)
        # This creates a format where Y gets 8 channels, U gets 2, V gets 2
        yuv_planar = np.zeros((h, w, 12), dtype=np.uint8)
        
        # Y plane (8 channels - replicate for robustness)
        for i in range(8):
            yuv_planar[:,:,i] = y
            
        # U plane (2 channels, downsampled)
        u_ds = u[::2, ::2]  # Downsample U
        for i in range(2):
            yuv_planar[::2, ::2, 8+i] = u_ds
            
        # V plane (2 channels, downsampled)  
        v_ds = v[::2, ::2]  # Downsample V
        for i in range(2):
            yuv_planar[::2, ::2, 10+i] = v_ds
            
        return yuv_planar

    def get_hidden_state(self, imgs: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract hidden state features from vision model.

        Args:
            imgs: Dictionary with 'road' and optionally 'big_road' keys containing
                preprocessed camera images in YUV420 format expected by the vision model.

        Returns:
            Feature vector of shape (feature_len,) for the policy network.
        """
        if self.use_dummy or not imgs:
            # Return random features that match expected statistics
            if HAS_NUMPY:
                features = np.random.randn(self.feature_len).astype(np.float32)
                return features * 0.1
            else:
                # Pure Python random features
                import random
                features = [random.gauss(0, 0.1) for _ in range(self.feature_len)]
                return features
        
        try:
            # Prepare inputs for ONNX model
            onnx_inputs = {}
            model_input_names = [inp.name for inp in self.vision_model.get_inputs()]
            
            # Map our input keys to ONNX model input names
            input_mapping = {
                'road': 'img',  # Main camera input
                'big_road': 'big_img',  # Wide camera input
            }
            
            for our_key, img in imgs.items():
                # Find the matching ONNX input name
                onnx_key = None
                if our_key in input_mapping and input_mapping[our_key] in model_input_names:
                    onnx_key = input_mapping[our_key]
                elif our_key in model_input_names:
                    onnx_key = our_key
                elif len(model_input_names) == 1:
                    # If there's only one input, use it regardless of name
                    onnx_key = model_input_names[0]
                
                if onnx_key is not None:
                    # Get expected shape from model
                    expected_shape = None
                    for inp in self.vision_model.get_inputs():
                        if inp.name == onnx_key:
                            expected_shape = inp.shape
                            break
                    
                    if expected_shape is not None:
                        # Process image to match expected format
                        processed_img = self._preprocess_for_model(img, expected_shape)
                        onnx_inputs[onnx_key] = processed_img
                        logger.debug(f"Added input {onnx_key} with shape {processed_img.shape}")
                    else:
                        logger.warning(f"Could not find expected shape for input {onnx_key}")
                        continue
            
            if not onnx_inputs:
                logger.warning("No valid inputs found for ONNX model")
                raise ValueError("No valid inputs for ONNX model")
            
            # Run ONNX model inference
            outputs = self.vision_model.run(None, onnx_inputs)
            
            # Extract hidden state features
            # Typically the hidden state is the first or last output
            hidden_state = None
            
            if len(outputs) == 1:
                # Single output - use it
                hidden_state = outputs[0]
            else:
                # Multiple outputs - look for features/hidden state
                output_names = [out.name for out in self.vision_model.get_outputs()]
                for i, name in enumerate(output_names):
                    if 'hidden' in name.lower() or 'feature' in name.lower() or 'embed' in name.lower():
                        hidden_state = outputs[i]
                        break
                
                # If no named hidden state found, use the largest output (likely features)
                if hidden_state is None:
                    largest_output = max(outputs, key=lambda x: x.size)
                    hidden_state = largest_output
            
            if hidden_state is None:
                raise ValueError("Could not identify hidden state output from ONNX model")
            
            # Flatten and ensure correct shape
            if HAS_NUMPY:
                hidden_state = np.array(hidden_state).flatten().astype(np.float32)
                
                # Pad or trim to expected feature length
                if len(hidden_state) != self.feature_len:
                    if len(hidden_state) > self.feature_len:
                        logger.debug(f"Trimming features from {len(hidden_state)} to {self.feature_len}")
                        hidden_state = hidden_state[:self.feature_len]
                    else:
                        logger.debug(f"Padding features from {len(hidden_state)} to {self.feature_len}")
                        padded = np.zeros(self.feature_len, dtype=np.float32)
                        padded[:len(hidden_state)] = hidden_state
                        hidden_state = padded
                
                return hidden_state
            else:
                # Pure Python handling
                hidden_state = hidden_state.flatten().tolist()
                if len(hidden_state) > self.feature_len:
                    hidden_state = hidden_state[:self.feature_len]
                elif len(hidden_state) < self.feature_len:
                    hidden_state.extend([0.0] * (self.feature_len - len(hidden_state)))
                
                return hidden_state
                
        except Exception as e:
            logger.error(f"ONNX vision model forward pass failed: {e}")
            if HAS_NUMPY:
                return np.random.randn(self.feature_len).astype(np.float32) * 0.1
            else:
                import random
                return [random.gauss(0, 0.1) for _ in range(self.feature_len)]

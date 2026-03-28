"""
NanoSplat :: Depth Engine
==========================
Monocular depth estimation using MiDaS-Small.

TWO INFERENCE PATHS:
  1. TensorRT (Jetson Nano) — FP16, ~8 FPS at 256×192
  2. ONNX Runtime (RPi / generic) — ~2–4 FPS at 192×144

WHY MiDaS-Small and not ZoeDepth / DPT-Large:
  - MiDaS-Small: 21MB, ~2.6 GFLOPs
  - ZoeDepth:    344MB, too slow on Nano
  - DPT-Large:   1.28GB, impossible
  MiDaS-Small gives metric-consistent relative depth that,
  when fused across keyframes with our pose estimator, produces
  coherent point clouds without metric ground-truth depth.

DEPTH SCALE RECOVERY:
  Since MiDaS gives relative depth, we use the median depth of
  the object ROI across consecutive frames + known object size
  priors (optional, user-provided) to recover absolute scale.
  Without priors: clouds are scale-ambiguous but still 3D-consistent.
"""

import numpy as np
import logging
import time
from pathlib import Path
from typing import Tuple, Optional

log = logging.getLogger("NanoSplat.Depth")


class DepthEngine:
    """
    Unified depth estimation interface.
    Automatically selects TensorRT or ONNX path based on hardware.
    """

    def __init__(self, hw_profile, model_dir: str = "models"):
        self.hw = hw_profile
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.input_size = hw_profile.depth_size  # (W, H)
        self._session = None
        self._trt_engine = None
        self._backend = None

        self._init_model()

    def _init_model(self):
        from .hardware import Backend
        if self.hw.backend == Backend.CUPY and self.hw.use_tensorrt:
            self._try_tensorrt()
        if self._trt_engine is None:
            self._try_onnx()
        if self._session is None and self._trt_engine is None:
            raise RuntimeError(
                "No depth backend available. Run:\n"
                "  python nanosplat/scripts/download_models.py"
            )

    def _try_tensorrt(self):
        """Build or load TensorRT engine from ONNX model for Jetson."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            engine_path = self.model_dir / "midas_small_fp16.trt"
            onnx_path   = self.model_dir / "midas_small.onnx"

            if engine_path.exists():
                log.info("Loading cached TensorRT engine...")
                with open(engine_path, 'rb') as f:
                    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                    self._trt_engine = runtime.deserialize_cuda_engine(f.read())
                self._backend = "tensorrt"
                log.info("TensorRT engine loaded.")
                return

            if not onnx_path.exists():
                log.warning(f"ONNX model not found at {onnx_path}. Skipping TensorRT.")
                return

            log.info("Building TensorRT engine (first run, ~2 min on Jetson)...")
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)

            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        log.error(parser.get_error(i))
                    return

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 on Jetson

            serialized = builder.build_serialized_network(network, config)
            with open(engine_path, 'wb') as f:
                f.write(serialized)

            runtime = trt.Runtime(logger)
            self._trt_engine = runtime.deserialize_cuda_engine(serialized)
            self._backend = "tensorrt"
            log.info("TensorRT engine built and cached.")

        except ImportError as e:
            log.info(f"TensorRT not available ({e}). Will use ONNX.")
        except Exception as e:
            log.warning(f"TensorRT build failed: {e}. Will use ONNX.")

    def _try_onnx(self):
        """Load MiDaS-Small as ONNX for CPU or generic GPU."""
        try:
            import onnxruntime as ort
        except ImportError:
            log.error("onnxruntime not installed. Run: pip install onnxruntime")
            return

        onnx_path = self.model_dir / "midas_small.onnx"
        if not onnx_path.exists():
            log.warning(f"Model not found: {onnx_path}")
            log.warning("Run: python nanosplat/scripts/download_models.py")
            # Create a dummy depth estimator for testing
            self._backend = "dummy"
            log.warning("Using DUMMY depth estimator. Download models for real depth!")
            return

        # Prefer CUDA execution provider if available, else CPU
        providers = ort.get_available_providers()
        pref = []
        if 'CUDAExecutionProvider' in providers:
            pref.append('CUDAExecutionProvider')
        pref.append('CPUExecutionProvider')

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.hw.cpu_cores
        opts.inter_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(str(onnx_path), opts, providers=pref)
        self._backend = "onnx"
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        log.info(f"ONNX depth model loaded. Providers: {pref}")

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a BGR frame.

        Args:
            frame_bgr: (H, W, 3) uint8

        Returns:
            depth: (H, W) float32, relative depth in [0, 1] (normalized)
                   Higher = closer. Rescale with depth_scale externally.
        """
        W, H = self.input_size
        t0 = time.perf_counter()

        if self._backend == "dummy":
            return self._dummy_depth(frame_bgr)

        # Preprocess: BGR → RGB, resize, normalize to ImageNet stats
        rgb = frame_bgr[:, :, ::-1]
        resized = self._resize_for_midas(rgb, W, H)
        inp = self._normalize_midas(resized)  # (1, 3, H, W) float32

        if self._backend == "tensorrt":
            depth_raw = self._infer_tensorrt(inp)
        else:
            depth_raw = self._infer_onnx(inp)

        # Resize back to original frame dimensions
        fh, fw = frame_bgr.shape[:2]
        depth_full = self._resize_depth(depth_raw, fw, fh)

        # Normalize to [0, 1]
        dmin, dmax = depth_full.min(), depth_full.max()
        if dmax > dmin:
            depth_norm = (depth_full - dmin) / (dmax - dmin)
        else:
            depth_norm = np.zeros_like(depth_full)

        elapsed = time.perf_counter() - t0
        log.debug(f"Depth estimate: {elapsed*1000:.1f}ms  backend={self._backend}")

        return depth_norm

    def estimate_metric(self, frame_bgr: np.ndarray, approx_object_dist_m: float = 1.0) -> np.ndarray:
        """
        Estimate depth with approximate metric scale.
        Uses MiDaS relative depth + user-provided approximate distance.
        For multi-view fusion, use the same scale factor across all frames.
        """
        rel_depth = self.estimate(frame_bgr)
        # MiDaS depth is inverted (higher = closer), convert to distance
        # depth_metric ≈ approx_dist / (rel_depth + epsilon)
        eps = 1e-4
        metric = approx_object_dist_m / (rel_depth + eps)
        return metric.astype(np.float32)

    def _resize_for_midas(self, rgb: np.ndarray, W: int, H: int) -> np.ndarray:
        import cv2
        return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

    def _normalize_midas(self, rgb: np.ndarray) -> np.ndarray:
        """Standard MiDaS normalization (ImageNet mean/std)."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (rgb.astype(np.float32) / 255.0 - mean) / std
        x = x.transpose(2, 0, 1)[None, ...]  # (1, 3, H, W)
        return x

    def _infer_onnx(self, inp: np.ndarray) -> np.ndarray:
        out = self._session.run([self._output_name], {self._input_name: inp})
        return out[0].squeeze()  # (H, W)

    def _infer_tensorrt(self, inp: np.ndarray) -> np.ndarray:
        """TensorRT inference on Jetson with CUDA memory."""
        import pycuda.driver as cuda
        import numpy as np

        context = self._trt_engine.create_execution_context()
        bindings = []
        outputs = []

        for i in range(self._trt_engine.num_bindings):
            shape = self._trt_engine.get_binding_shape(i)
            size  = int(np.prod(shape))
            dtype = np.float32
            mem = cuda.mem_alloc(size * dtype().itemsize)
            bindings.append(int(mem))
            if self._trt_engine.binding_is_input(i):
                cuda.memcpy_htod(mem, inp.astype(np.float32))
            else:
                outputs.append((mem, shape))

        stream = cuda.Stream()
        context.execute_async_v2(bindings, stream.handle)
        stream.synchronize()

        result_mem, result_shape = outputs[0]
        result = np.empty(result_shape, dtype=np.float32)
        cuda.memcpy_dtoh(result, result_mem)
        return result.squeeze()

    def _resize_depth(self, depth: np.ndarray, W: int, H: int) -> np.ndarray:
        import cv2
        return cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

    def _dummy_depth(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Dummy depth: uses Laplacian-weighted gradient as a rough depth proxy.
        Useful for testing the pipeline without downloading models.
        For a ball: edges tend to be boundary (far) and center (close).
        """
        import cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap  = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        # Invert: assume center of frame is close, edges far
        H, W = gray.shape
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        dist_from_center = np.sqrt((x - W/2)**2 + (y - H/2)**2)
        dist_norm = 1.0 - (dist_from_center / dist_from_center.max())
        # Mix: 70% center-based, 30% sharpness-based
        depth = 0.7 * dist_norm + 0.3 * (lap / (lap.max() + 1e-6))
        return depth.astype(np.float32)

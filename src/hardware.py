"""
NanoSplat :: Hardware Abstraction Layer
=======================================
Auto-detects: Jetson Nano (CUDA 2GB), RPi 4 (CPU), or x86 dev machine.
Selects compute backend: CuPy (GPU) → Numba JIT (CPU SIMD) → NumPy.

This layer is the foundation of NanoSplat's portability.
"""

import os
import sys
import platform
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

log = logging.getLogger("NanoSplat.HW")


class Backend(Enum):
    CUPY   = auto()   # Jetson Nano / any CUDA device ≥ 2GB
    NUMBA  = auto()   # RPi 4 / ARM64 CPU with JIT
    NUMPY  = auto()   # Pure fallback


@dataclass
class HardwareProfile:
    backend:       Backend
    device_name:   str
    gpu_mem_mb:    int   = 0
    cpu_cores:     int   = 4
    is_jetson:     bool  = False
    is_rpi:        bool  = False
    max_gaussians: int   = 50_000   # Tuned per device
    depth_size:    tuple = (256, 192)  # MiDaS input resolution
    det_size:      int   = 320
    splat_fps_target: float = 5.0
    # Computed fields
    use_tensorrt:  bool  = field(init=False, default=False)
    use_onnx:      bool  = field(init=False, default=True)

    def __post_init__(self):
        if self.is_jetson and self.backend == Backend.CUPY:
            self.use_tensorrt = True   # TensorRT path for Jetson
            self.use_onnx = False


def detect_hardware() -> HardwareProfile:
    """
    Auto-detect the running hardware and return an optimal HardwareProfile.
    Priority: Jetson CUDA → Generic CUDA → Numba CPU → NumPy CPU
    """
    # ── Jetson detection via tegra chip id ──────────────────────────────────
    is_jetson = _is_jetson()
    is_rpi    = _is_rpi()

    # ── Try CuPy (requires CUDA runtime) ────────────────────────────────────
    try:
        import cupy as cp
        device = cp.cuda.Device(0)
        mem_info = device.mem_info
        gpu_mem_mb = mem_info[1] // (1024 * 1024)

        if gpu_mem_mb >= 1800:  # At least ~2GB usable
            profile = HardwareProfile(
                backend=Backend.CUPY,
                device_name=f"CUDA:{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}",
                gpu_mem_mb=gpu_mem_mb,
                is_jetson=is_jetson,
                max_gaussians=80_000 if gpu_mem_mb >= 3500 else 40_000,
                depth_size=(384, 288) if gpu_mem_mb >= 3500 else (256, 192),
                splat_fps_target=8.0 if is_jetson else 15.0,
            )
            log.info(f"Backend: CuPy  |  Device: {profile.device_name}  |  VRAM: {gpu_mem_mb}MB")
            return profile
        else:
            log.warning(f"CUDA found but only {gpu_mem_mb}MB VRAM — falling back to CPU.")
    except Exception as e:
        log.info(f"CuPy unavailable ({type(e).__name__}), trying Numba...")

    # ── Try Numba (JIT-compiled CPU kernels, ARM64-friendly) ────────────────
    try:
        from numba import njit
        import multiprocessing
        cores = multiprocessing.cpu_count()

        profile = HardwareProfile(
            backend=Backend.NUMBA,
            device_name=f"CPU ({platform.processor() or platform.machine()})",
            cpu_cores=cores,
            is_rpi=is_rpi,
            max_gaussians=15_000,  # Memory-safe for 4GB RAM
            depth_size=(192, 144),
            det_size=256,
            splat_fps_target=3.0,
        )
        log.info(f"Backend: Numba JIT  |  Cores: {cores}  |  Device: {profile.device_name}")
        return profile
    except ImportError:
        log.warning("Numba unavailable, using pure NumPy.")

    # ── Pure NumPy fallback ──────────────────────────────────────────────────
    profile = HardwareProfile(
        backend=Backend.NUMPY,
        device_name="CPU (NumPy)",
        max_gaussians=5_000,
        depth_size=(128, 96),
        splat_fps_target=1.0,
    )
    log.info(f"Backend: NumPy  |  Expect low performance.")
    return profile


def _is_jetson() -> bool:
    try:
        with open("/proc/device-tree/model") as f:
            return "jetson" in f.read().lower()
    except Exception:
        return False


def _is_rpi() -> bool:
    try:
        with open("/proc/device-tree/model") as f:
            return "raspberry" in f.read().lower()
    except Exception:
        return False


# ── Compute array helpers (backend-agnostic) ─────────────────────────────────

class XP:
    """
    Backend-agnostic array operations.
    Use xp.array(), xp.zeros(), etc. just like numpy — works on GPU or CPU.
    """
    _np = None

    @classmethod
    def init(cls, backend: Backend):
        if backend == Backend.CUPY:
            try:
                import cupy
                cls._np = cupy
                return
            except ImportError:
                pass
        import numpy
        cls._np = numpy

    @classmethod
    def get(cls):
        if cls._np is None:
            import numpy
            return numpy
        return cls._np

    @classmethod
    def to_numpy(cls, arr):
        """Safely convert to NumPy (handles both CuPy and NumPy arrays)."""
        np = cls._np
        if hasattr(arr, 'get'):  # CuPy array
            return arr.get()
        return arr

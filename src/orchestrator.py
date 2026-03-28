"""
NanoSplat :: Reconstruction Orchestrator
=========================================
Ties together: Depth → Pose → Gaussian Seeding → Fusion → Export

This is the "new idea" part of NanoSplat:
  Instead of training a NeRF or 3DGS from scratch, we do:

  1. Detect object → get mask M_i for keyframe i
  2. Run MiDaS on ROI → depth map D_i
  3. Unproject masked pixels D_i[M_i] with camera intrinsics → 3D points
  4. Transform points to world frame using pose P_i
  5. Represent each 3D point as a tiny Gaussian G_i
  6. Fuse all G_i into a growing cloud C using confidence-weighted merge
  7. Periodically export C as PLY + .splat for real-time viewing

The result: a coloured, segmented 3D point cloud / Gaussian cloud
of ONLY the target object, built in real time, no training required.

PERFORMANCE BUDGET (Jetson Nano, 2GB):
  Depth (MiDaS TensorRT FP16):  ~100ms
  Pose (ORB matching):            ~15ms
  Gaussian seeding (NumPy):       ~25ms
  Fusion + prune:                 ~10ms
  PLY export (keyframe):          ~5ms
  Total per keyframe:            ~155ms → ~6 keyframes/sec

PERFORMANCE BUDGET (RPi 4, CPU):
  Depth (MiDaS ONNX):            ~400ms
  Pose (ORB):                     ~50ms
  Gaussian seeding:               ~80ms
  Total per keyframe:            ~530ms → ~2 keyframes/sec
"""

import cv2
import numpy as np
import logging
import time
import threading
from pathlib import Path
from typing import Optional
from queue import Queue, Empty

log = logging.getLogger("NanoSplat.Reconstruction")


class ReconstructionOrchestrator:
    """
    Manages the full 3D reconstruction pipeline for a detected object.

    Runs depth estimation and Gaussian fusion in a BACKGROUND THREAD
    so the main capture loop is never blocked by heavy computation.
    """

    def __init__(self, hw_profile, output_dir: str = "output_3d",
                 object_dist_m: float = 1.0, keyframe_interval: int = 10,
                 auto_export_every: int = 30):
        self.hw = hw_profile
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "ply").mkdir(exist_ok=True)
        (self.output_dir / "splat").mkdir(exist_ok=True)
        (self.output_dir / "crops").mkdir(exist_ok=True)
        (self.output_dir / "rgba").mkdir(exist_ok=True)

        self.object_dist_m = object_dist_m
        self.keyframe_interval = keyframe_interval   # Process every N frames
        self.auto_export_every = auto_export_every   # Export PLY every N keyframes

        # Lazy init (heavy objects)
        self._depth_engine = None
        self._pose_estimator = None
        self._cloud = None
        self._K: Optional[np.ndarray] = None

        # Background processing queue
        self._queue: Queue = Queue(maxsize=4)
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="NS-Reconstruction"
        )
        self._running = False
        self._keyframe_count = 0
        self._total_gaussians = 0
        self._lock = threading.Lock()
        self._latest_stats: dict = {}

    def start(self, frame_width: int, frame_height: int):
        """Initialise all heavy components and start background thread."""
        from .hardware import XP
        from ..core.depth import DepthEngine
        from ..core.pose import LightweightPoseEstimator
        from ..core.gaussian import MicroGaussianCloud

        XP.init(self.hw.backend)

        # Default camera intrinsics (override with calibration if available)
        self._K = LightweightPoseEstimator.default_K(frame_width, frame_height)

        # Components
        self._depth_engine   = DepthEngine(self.hw)
        self._pose_estimator = LightweightPoseEstimator(self._K,
                               max_features=300 if self.hw.is_rpi else 500)
        self._cloud = MicroGaussianCloud(self.hw.max_gaussians)

        self._running = True
        self._worker_thread.start()
        log.info(f"Reconstruction orchestrator started.  "
                 f"Output: {self.output_dir}  "
                 f"Max Gaussians: {self.hw.max_gaussians}")

    def stop(self):
        """Gracefully stop the background thread and do a final export."""
        self._running = False
        self._queue.put(None)  # Sentinel
        self._worker_thread.join(timeout=10)
        self._final_export()
        log.info(f"Reconstruction stopped. Total keyframes: {self._keyframe_count}  "
                 f"Gaussians: {self._total_gaussians}")

    def submit_frame(self, frame_bgr: np.ndarray, extraction_result,
                     frame_id: int) -> bool:
        """
        Submit a frame for background 3D reconstruction.
        Non-blocking — drops frame if queue is full (back-pressure).
        Returns True if frame was accepted.
        """
        if not self._running or not extraction_result.has_object:
            return False
        if frame_id % self.keyframe_interval != 0:
            return False

        try:
            # Copy only needed data to avoid race conditions
            payload = {
                'frame_bgr':    frame_bgr.copy(),
                'roi_bgr':      extraction_result.roi_bgr.copy()
                                if extraction_result.roi_bgr is not None else None,
                'mask_stable':  extraction_result.mask_stable.copy()
                                if extraction_result.mask_stable is not None else None,
                'bbox':         extraction_result.bbox,
                'frame_id':     frame_id,
            }
            self._queue.put_nowait(payload)
            return True
        except Exception:
            return False  # Queue full, drop frame

    def _worker_loop(self):
        """Background thread: runs depth + pose + Gaussian fusion."""
        log.info("Reconstruction worker started.")
        while self._running:
            try:
                payload = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if payload is None:  # Sentinel
                break

            try:
                self._process_keyframe(payload)
            except Exception as e:
                log.error(f"Reconstruction error: {e}", exc_info=True)

        log.info("Reconstruction worker stopped.")

    def _process_keyframe(self, payload: dict):
        t0 = time.perf_counter()

        frame_bgr  = payload['frame_bgr']
        roi_bgr    = payload['roi_bgr']
        mask       = payload['mask_stable']
        bbox       = payload['bbox']
        frame_id   = payload['frame_id']

        if roi_bgr is None or mask is None or bbox is None:
            return

        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox

        # ── Step 1: Depth estimation on full frame ────────────────────────
        depth_full = self._depth_engine.estimate_metric(frame_bgr, self.object_dist_m)

        # ── Step 2: Pose estimation ───────────────────────────────────────
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        pose = self._pose_estimator.process_frame(gray)
        if pose is None:
            return
        T_world_from_cam = pose.T_world_from_cam

        # ── Step 3: Extract depth ROI aligned with mask ───────────────────
        depth_roi = depth_full[y1:y2, x1:x2]

        # Resize mask to match depth_roi
        mask_roi = cv2.resize(mask, (depth_roi.shape[1], depth_roi.shape[0]))

        # ── Step 4: Seed Gaussians from depth + mask ──────────────────────
        # Compute ROI-relative intrinsics
        K_roi = self._K.copy()
        K_roi[0, 2] -= x1   # principal point shifts with crop
        K_roi[1, 2] -= y1

        # Colour map aligned with depth_roi
        color_roi = roi_bgr if roi_bgr.shape[:2] == depth_roi.shape else \
                    cv2.resize(roi_bgr, (depth_roi.shape[1], depth_roi.shape[0]))

        n_added = self._cloud.seed_from_depth(
            depth_map=depth_roi,
            mask=mask_roi,
            color_map=color_roi,
            K=K_roi,
            pose=T_world_from_cam.astype(np.float64),
            max_new=2_000 if self.hw.is_rpi else 4_000,
        )

        # ── Step 5: Save cropped outputs ──────────────────────────────────
        kf_id = self._keyframe_count
        tag   = f"{kf_id:05d}"
        cv2.imwrite(str(self.output_dir / "crops" / f"{tag}.jpg"), roi_bgr)

        rgba = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:,:,3] = cv2.resize(mask, (roi_bgr.shape[1], roi_bgr.shape[0]))
        cv2.imwrite(str(self.output_dir / "rgba" / f"{tag}.png"), rgba)

        # ── Step 6: Periodic PLY export ───────────────────────────────────
        with self._lock:
            self._keyframe_count += 1
            self._total_gaussians = self._cloud.count
            self._latest_stats = {
                'keyframes':  self._keyframe_count,
                'gaussians':  self._cloud.count,
                'fill_pct':   100 * self._cloud.count / self._cloud.max_n,
                'latency_ms': (time.perf_counter() - t0) * 1000,
                'n_added':    n_added,
            }

        if self._keyframe_count % self.auto_export_every == 0:
            self._export_cloud(self._keyframe_count)

        log.info(f"KF#{kf_id}: +{n_added} Gaussians  "
                 f"total={self._cloud.count}  "
                 f"latency={self._latest_stats['latency_ms']:.0f}ms")

    def _export_cloud(self, kf_id: int):
        """Export current cloud to PLY and .splat."""
        ply_path   = self.output_dir / "ply"   / f"cloud_{kf_id:05d}.ply"
        splat_path = self.output_dir / "splat" / f"cloud_{kf_id:05d}.splat"

        self._cloud.to_ply(str(ply_path))

        splat_bytes = self._cloud.to_splat_bytes()
        if splat_bytes:
            with open(splat_path, 'wb') as f:
                f.write(splat_bytes)
            log.info(f".splat exported: {splat_path} ({len(splat_bytes)/1024:.1f} KB)")

        # Always overwrite "latest" symlinks for live viewers
        latest_ply   = self.output_dir / "latest.ply"
        latest_splat = self.output_dir / "latest.splat"
        try:
            import shutil
            shutil.copy2(ply_path, latest_ply)
            if splat_bytes:
                shutil.copy2(splat_path, latest_splat)
        except Exception:
            pass

    def _final_export(self):
        if self._cloud and self._cloud.count > 0:
            self._export_cloud(self._keyframe_count)
            log.info(f"Final export: {self._cloud.count} Gaussians.")

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._latest_stats)

    def set_camera_intrinsics(self, K: np.ndarray):
        """Override default intrinsics with calibrated values."""
        self._K = K.astype(np.float64)
        if self._pose_estimator:
            self._pose_estimator.K = self._K
        log.info(f"Camera intrinsics updated: fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

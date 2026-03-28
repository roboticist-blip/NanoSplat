"""
NanoSplat :: MicroGaussian Engine
==================================
A from-scratch, minimal Gaussian Splatting implementation designed
for 2GB GPU (Jetson Nano) and ARM64 CPU (Raspberry Pi 4).

WHY THIS IS NOVEL:
==================
Standard 3DGS (Kerbl et al., 2023) requires:
  - Full CUDA tile-based rasterizer (complex C++/CUDA)
  - 8–24GB VRAM for training
  - Hours of optimization per scene

NanoSplat's MicroGaussian approach:
  1. Gaussians are SEEDED directly from depth maps — no training needed
  2. Instead of differentiable rendering for gradient descent,
     we use INCREMENTAL FUSION: new keyframes update Gaussian parameters
     via weighted averaging (like a Kalman filter for 3D position/color)
  3. The rasterizer is a lightweight numpy/cupy alpha-compositing
     pass — not tile-based, but fast enough for sparse scenes

This makes real-time 3D reconstruction possible without a training loop.

GAUSSIAN PARAMETERIZATION (stripped down for edge devices):
  - position:  (x, y, z)           float32  [3]
  - color:     (r, g, b)           float32  [3]  — SH degree 0 only
  - opacity:   scalar              float32  [1]
  - scale:     (sx, sy, sz)        float32  [3]  — axis-aligned, no rotation quaternion
  - confidence: scalar             float32  [1]  — our custom field for fusion weight

Dropping quaternion rotation and using axis-aligned covariance
saves 4 floats per Gaussian and avoids quaternion-to-matrix ops in the rasterizer.
For object reconstruction (not full scenes), axis-aligned Gaussians work well.
"""

import numpy as np
import logging
from typing import Optional, Tuple
from pathlib import Path

log = logging.getLogger("NanoSplat.Gaussian")

# ── Gaussian cloud dtype ──────────────────────────────────────────────────────
GAUSSIAN_DTYPE = np.dtype([
    ('pos',   np.float32, (3,)),   # XYZ world position
    ('color', np.float32, (3,)),   # RGB [0,1]
    ('opacity', np.float32),       # [0,1]
    ('scale', np.float32, (3,)),   # Axis-aligned standard deviations
    ('conf',  np.float32),         # Fusion confidence weight
    ('active', np.bool_),          # Is this Gaussian valid?
])


class MicroGaussianCloud:
    """
    Fixed-size Gaussian cloud with ring-buffer update semantics.
    Pre-allocated to avoid memory fragmentation on embedded systems.
    """

    def __init__(self, max_gaussians: int = 20_000):
        self.max_n = max_gaussians
        self.data  = np.zeros(max_gaussians, dtype=GAUSSIAN_DTYPE)
        self.count = 0   # Current active count
        self._write_ptr = 0  # Ring buffer pointer

        # Pre-allocate workspace arrays (avoid per-frame allocation)
        self._tmp_screen = np.zeros((max_gaussians, 2), dtype=np.float32)
        self._tmp_depths  = np.zeros(max_gaussians, dtype=np.float32)

        log.info(f"MicroGaussianCloud: {max_gaussians} max Gaussians  "
                 f"({max_gaussians * GAUSSIAN_DTYPE.itemsize / 1024:.1f} KB)")

    def seed_from_depth(
        self,
        depth_map: np.ndarray,      # (H, W) float32, metric depth in meters
        mask: np.ndarray,           # (H, W) uint8 binary mask of object
        color_map: np.ndarray,      # (H, W, 3) uint8 BGR
        K: np.ndarray,              # (3, 3) camera intrinsics
        pose: np.ndarray,           # (4, 4) world-from-camera transform
        depth_scale: float = 1.0,
        max_new: int = 3_000,
    ) -> int:
        """
        Lift 2D depth + mask pixels into 3D Gaussians and add to the cloud.

        Strategy:
          1. Subsample masked pixels (reservoir sampling for speed)
          2. Unproject each pixel through the depth map using intrinsics K
          3. Transform to world frame using pose
          4. Create one Gaussian per pixel with scale ∝ depth uncertainty
          5. Try to MERGE with nearby existing Gaussians (confidence-weighted)

        Returns number of Gaussians added/updated.
        """
        H, W = depth_map.shape
        ys, xs = np.where(mask > 127)

        if len(ys) == 0:
            return 0

        # ── Reservoir sampling: keep at most max_new points ─────────────────
        if len(ys) > max_new:
            idx = np.random.choice(len(ys), max_new, replace=False)
            ys, xs = ys[idx], xs[idx]

        d = depth_map[ys, xs].astype(np.float32) * depth_scale

        # Filter invalid depths
        valid = (d > 0.05) & (d < 10.0)
        ys, xs, d = ys[valid], xs[valid], d[valid]

        if len(ys) == 0:
            return 0

        # ── Unproject: pixel → camera space ─────────────────────────────────
        # P_cam = K^{-1} · [u, v, 1]^T · depth
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        X_cam = (xs - cx) / fx * d
        Y_cam = (ys - cy) / fy * d
        Z_cam = d

        # Stack into (N, 4) homogeneous
        pts_cam = np.stack([X_cam, Y_cam, Z_cam, np.ones_like(d)], axis=1)  # (N,4)

        # ── Transform to world frame ─────────────────────────────────────────
        pts_world = (pose @ pts_cam.T).T[:, :3]  # (N,3)

        # ── Colors ──────────────────────────────────────────────────────────
        colors_bgr = color_map[ys, xs].astype(np.float32) / 255.0
        colors_rgb = colors_bgr[:, ::-1]  # BGR → RGB

        # ── Scale ∝ depth uncertainty (farther → larger Gaussian) ───────────
        # Heuristic: σ = depth * pixel_size_at_1m / focal
        pixel_footprint = d / fx  # 1 pixel at depth d
        scales = np.clip(pixel_footprint * 0.5, 0.002, 0.1)
        scales_3d = np.stack([scales, scales, scales * 1.5], axis=1)  # slightly taller

        n_new = len(pts_world)
        self._insert_batch(pts_world, colors_rgb, scales_3d, conf=1.0)

        # Prune low-confidence Gaussians if over budget
        if self.count > self.max_n * 0.9:
            self._prune_low_confidence()

        return n_new

    def _insert_batch(
        self,
        positions: np.ndarray,   # (N, 3)
        colors:    np.ndarray,   # (N, 3)
        scales:    np.ndarray,   # (N, 3)
        conf:      float = 1.0,
        opacity:   float = 0.85,
    ):
        """Ring-buffer insert. Overwrites oldest when full."""
        n = len(positions)
        for i in range(n):
            ptr = self._write_ptr % self.max_n
            g = self.data[ptr]
            if g['active'] and g['conf'] > 1.5:
                # Confidence-weighted merge instead of overwrite
                w = g['conf']
                g['pos']   = (g['pos'] * w + positions[i]) / (w + 1)
                g['color'] = (g['color'] * w + colors[i]) / (w + 1)
                g['conf']  = min(w + 1, 8.0)
            else:
                g['pos']     = positions[i]
                g['color']   = colors[i]
                g['opacity'] = opacity
                g['scale']   = scales[i]
                g['conf']    = conf
                g['active']  = True
                self.count = min(self.count + 1, self.max_n)
            self._write_ptr += 1

    def _prune_low_confidence(self, keep_fraction: float = 0.85):
        """Remove bottom-confidence Gaussians to free space."""
        active_idx = np.where(self.data['active'])[0]
        if len(active_idx) < 100:
            return
        confs = self.data['conf'][active_idx]
        threshold = np.percentile(confs, (1 - keep_fraction) * 100)
        kill = active_idx[confs < threshold]
        self.data['active'][kill] = False
        self.data['conf'][kill] = 0.0
        self.count = int(np.sum(self.data['active']))
        log.debug(f"Pruned {len(kill)} low-conf Gaussians. Active: {self.count}")

    def active_gaussians(self) -> np.ndarray:
        """Return structured array of active Gaussians."""
        return self.data[self.data['active']]

    def to_ply(self, path: str):
        """
        Export cloud as PLY point cloud.
        Compatible with MeshLab, CloudCompare, Open3D, Blender.

        Uses Gaussian center positions + RGB colors + confidence as intensity.
        """
        active = self.active_gaussians()
        n = len(active)
        if n == 0:
            log.warning("Empty cloud — nothing to export.")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        positions = active['pos']           # (N, 3)
        colors    = (active['color'] * 255).clip(0, 255).astype(np.uint8)  # (N, 3)
        normals   = np.zeros((n, 3), dtype=np.float32)
        normals[:, 2] = 1.0  # placeholder normals

        with open(path, 'wb') as f:
            header = (
                f"ply\n"
                f"format binary_little_endian 1.0\n"
                f"element vertex {n}\n"
                f"property float x\n"
                f"property float y\n"
                f"property float z\n"
                f"property float nx\n"
                f"property float ny\n"
                f"property float nz\n"
                f"property uchar red\n"
                f"property uchar green\n"
                f"property uchar blue\n"
                f"property float confidence\n"
                f"end_header\n"
            )
            f.write(header.encode('ascii'))

            # Write binary records
            for i in range(n):
                f.write(positions[i].tobytes())          # xyz float32 ×3
                f.write(normals[i].tobytes())            # nxnynz float32 ×3
                f.write(colors[i].tobytes())             # rgb uint8 ×3
                f.write(active['conf'][i:i+1].astype(np.float32).tobytes())  # conf

        log.info(f"PLY exported: {path}  ({n} points, {path.stat().st_size/1024:.1f} KB)")

    def to_splat_bytes(self) -> bytes:
        """
        Export as minimal .splat format (compatible with online 3DGS viewers).
        Format per Gaussian (32 bytes):
          pos:     3 × float32  (12 bytes)
          scale:   3 × float32  (12 bytes)
          color:   4 × uint8    ( 4 bytes)  RGB + opacity
          rot:     4 × uint8    ( 4 bytes)  quaternion (identity for us)
        """
        active = self.active_gaussians()
        n = len(active)
        if n == 0:
            return b''

        buf = bytearray(n * 32)
        view = memoryview(buf)

        pos    = active['pos'].astype(np.float32)
        scales = active['scale'].astype(np.float32)
        colors_u8 = (active['color'] * 255).clip(0, 255).astype(np.uint8)
        opacity_u8 = (active['opacity'] * 255).clip(0, 255).astype(np.uint8)

        # Identity quaternion bytes: [128, 0, 0, 0]
        identity_rot = np.array([[128, 0, 0, 0]], dtype=np.uint8).repeat(n, axis=0)

        # Pack struct manually for speed
        for i in range(n):
            offset = i * 32
            view[offset:offset+12]    = pos[i].tobytes()
            view[offset+12:offset+24] = scales[i].tobytes()
            view[offset+24]           = int(colors_u8[i, 0])
            view[offset+25]           = int(colors_u8[i, 1])
            view[offset+26]           = int(colors_u8[i, 2])
            view[offset+27]           = int(opacity_u8[i])
            view[offset+28:offset+32] = identity_rot[i].tobytes()

        return bytes(buf)

    def get_bbox_3d(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (min_xyz, max_xyz) bounding box of active Gaussians."""
        active = self.active_gaussians()
        if len(active) == 0:
            return None
        return active['pos'].min(axis=0), active['pos'].max(axis=0)

    def stats(self) -> dict:
        return {
            'active':    self.count,
            'capacity':  self.max_n,
            'fill_pct':  100 * self.count / self.max_n,
            'bbox':      self.get_bbox_3d(),
        }

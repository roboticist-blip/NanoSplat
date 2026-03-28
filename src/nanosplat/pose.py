"""
NanoSplat :: Lightweight Pose Estimator
========================================
Estimates camera pose (R|t) between keyframes using ORB feature matching
and PnP solving. No ROS, no heavy SLAM library — pure OpenCV.

WHY POSE ESTIMATION MATTERS FOR 3D:
  Without knowing where the camera was when each frame was captured,
  depth-lifted point clouds from different frames cannot be fused into
  a consistent 3D structure. Each frame's cloud would float in its own
  independent camera space.

ARCHITECTURE:
  Frame N → ORB features → Match with Frame N-1 → Essential Matrix →
  R,t → Accumulate pose chain → World-from-camera transform (4×4)

LIMITATIONS ON EDGE DEVICES:
  - We use ORB (binary descriptor) not SIFT/SuperPoint — 100× faster
  - We skip loop closure (too expensive for real-time on Pi/Jetson)
  - We skip bundle adjustment (use incremental pose only)
  - Drift accumulates over time — acceptable for 10–60s object scans

FOR STATIC OBJECT SCANS (recommended use):
  User orbits camera around a stationary object.
  Pose chain gives the multi-view geometry needed for dense fusion.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List

log = logging.getLogger("NanoSplat.Pose")


class FramePose:
    """Represents the pose of a single keyframe."""
    def __init__(self, R: np.ndarray, t: np.ndarray, frame_id: int):
        self.R = R              # (3,3) rotation matrix
        self.t = t              # (3,1) translation vector
        self.frame_id = frame_id
        self._T = None          # Cached 4×4 transform

    @property
    def T_world_from_cam(self) -> np.ndarray:
        """4×4 world-from-camera homogeneous transform."""
        if self._T is None:
            self._T = np.eye(4, dtype=np.float64)
            self._T[:3, :3] = self.R
            self._T[:3, 3]  = self.t.ravel()
        return self._T

    @property
    def cam_center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.R.T @ self.t.ravel()


class LightweightPoseEstimator:
    """
    ORB + Essential Matrix + PnP pose estimator.
    Designed for 3–15 FPS keyframe processing on Jetson Nano / RPi 4.
    """

    def __init__(self, K: np.ndarray, max_features: int = 500):
        """
        Args:
            K: (3,3) camera intrinsics matrix
            max_features: ORB features per frame (lower = faster)
        """
        self.K = K.astype(np.float64)
        self.max_features = max_features

        # ORB detector — binary descriptor, arm64-optimised in OpenCV
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=4,
            edgeThreshold=15,
            patchSize=31,
        )

        # Hamming distance matcher (correct for binary ORB descriptors)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # State
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kp:   Optional[List]       = None
        self.prev_desc: Optional[np.ndarray] = None
        self.poses: List[FramePose] = []

        # Initialise at identity (camera at world origin)
        self.current_R = np.eye(3, dtype=np.float64)
        self.current_t = np.zeros((3, 1), dtype=np.float64)
        self.frame_id  = 0

        log.info(f"PoseEstimator: ORB {max_features} features, K={K[:2,:3]}")

    def process_frame(self, gray: np.ndarray) -> Optional[FramePose]:
        """
        Process a new grayscale frame.
        Returns FramePose if pose was successfully estimated, else None.
        """
        kp, desc = self.orb.detectAndCompute(gray, None)

        if self.prev_gray is None or desc is None or len(kp) < 20:
            # First frame or too few features — initialise at identity
            pose = FramePose(self.current_R.copy(), self.current_t.copy(), self.frame_id)
            self.poses.append(pose)
            self._update_prev(gray, kp, desc)
            self.frame_id += 1
            return pose

        if self.prev_desc is None or len(self.prev_kp) < 20:
            self._update_prev(gray, kp, desc)
            return None

        # ── Match descriptors ─────────────────────────────────────────────
        matches = self.matcher.knnMatch(self.prev_desc, desc, k=2)

        # Lowe's ratio test
        good = []
        for match_pair in matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 10:
            log.debug(f"Frame {self.frame_id}: too few good matches ({len(good)})")
            self._update_prev(gray, kp, desc)
            self.frame_id += 1
            pose = FramePose(self.current_R.copy(), self.current_t.copy(), self.frame_id)
            self.poses.append(pose)
            return pose

        # ── Extract matched point pairs ───────────────────────────────────
        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in good])

        # ── Essential matrix + recover pose ──────────────────────────────
        E, mask_e = cv2.findEssentialMat(
            pts_prev, pts_curr,
            self.K, method=cv2.RANSAC, prob=0.999, threshold=1.5,
        )

        if E is None:
            self._update_prev(gray, kp, desc)
            self.frame_id += 1
            return None

        _, R_rel, t_rel, mask_p = cv2.recoverPose(E, pts_prev, pts_curr, self.K)

        # ── Accumulate pose (simple chain multiply) ───────────────────────
        # T_world_curr = T_world_prev · T_prev_curr
        self.current_R = self.current_R @ R_rel
        self.current_t = self.current_t + self.current_R @ t_rel

        # Re-orthogonalise R periodically to fight floating point drift
        if self.frame_id % 30 == 0:
            U, _, Vt = np.linalg.svd(self.current_R)
            self.current_R = U @ Vt

        pose = FramePose(self.current_R.copy(), self.current_t.copy(), self.frame_id)
        self.poses.append(pose)
        self._update_prev(gray, kp, desc)
        self.frame_id += 1

        inliers = int(mask_p.sum()) if mask_p is not None else len(good)
        log.debug(f"Frame {self.frame_id}: {len(good)} matches, {inliers} inliers, "
                  f"t_norm={np.linalg.norm(t_rel):.3f}")
        return pose

    def _update_prev(self, gray, kp, desc):
        self.prev_gray = gray
        self.prev_kp   = kp
        self.prev_desc = desc

    def reset(self):
        """Reset to identity pose (start of new scan)."""
        self.current_R = np.eye(3, dtype=np.float64)
        self.current_t = np.zeros((3, 1), dtype=np.float64)
        self.prev_gray = None
        self.prev_kp   = None
        self.prev_desc = None
        self.poses.clear()
        self.frame_id = 0
        log.info("Pose estimator reset.")

    @staticmethod
    def default_K(width: int, height: int) -> np.ndarray:
        """
        Estimate camera intrinsics assuming 70° horizontal FoV.
        Use this when calibration data is unavailable.
        For production, calibrate with cv2.calibrateCamera() using a chessboard.
        """
        fx = width / (2 * np.tan(np.radians(35)))  # 70°/2 half-angle
        fy = fx
        cx = width  / 2.0
        cy = height / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

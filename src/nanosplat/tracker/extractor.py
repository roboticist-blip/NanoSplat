"""
NanoSplat :: Object Extractor
==============================
Integrates YOLOv8n detection, MOSSE tracking, K-means segmentation,
and temporal mask stabilization into a single cohesive pipeline stage.

This module produces per-frame:
  - Bounding box (xyxy)
  - Binary mask (ROI-sized)
  - Cropped BGR ROI
  - Stable mask (temporally blended)
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple

log = logging.getLogger("NanoSplat.Extractor")

# ── COCO alias map ────────────────────────────────────────────────────────────
ALIAS_MAP = {
    "ball": "sports ball", "football": "sports ball", "soccer": "sports ball",
    "basketball": "sports ball", "bottle": "bottle", "cup": "cup", "mug": "cup",
    "person": "person", "human": "person", "man": "person", "woman": "person",
    "car": "car", "vehicle": "car", "truck": "truck", "bus": "bus",
    "bike": "bicycle", "bicycle": "bicycle", "motorbike": "motorcycle",
    "dog": "dog", "cat": "cat", "bird": "bird", "laptop": "laptop",
    "phone": "cell phone", "mobile": "cell phone", "chair": "chair",
    "backpack": "backpack", "book": "book", "clock": "clock",
    "vase": "vase", "teddy": "teddy bear", "plant": "potted plant",
    "keyboard": "keyboard", "mouse": "mouse", "tv": "tv",
    "orange": "orange", "apple": "apple", "banana": "banana",
    "sandwich": "sandwich", "pizza": "pizza", "cake": "cake",
    "umbrella": "umbrella", "handbag": "handbag", "bag": "handbag",
    "suitcase": "suitcase", "kite": "kite", "frisbee": "frisbee",
}

def resolve_class(user_input: str) -> str:
    key = user_input.strip().lower()
    return ALIAS_MAP.get(key, key)


def create_tracker():
    for name, factory in [
        ("MOSSE", lambda: cv2.legacy.TrackerMOSSE_create()),
        ("KCF",   lambda: cv2.TrackerKCF_create()),
        ("CSRT",  lambda: cv2.TrackerCSRT_create()),
    ]:
        try:
            t = factory()
            log.info(f"Tracker: {name}")
            return t, name
        except AttributeError:
            continue
    raise RuntimeError("No OpenCV tracker found. Install opencv-contrib-python.")


class ObjectSegmenter:
    """HSV K-means + Canny + Morphology — lightweight, no neural segmentor."""

    def __init__(self, k: int = 2):
        self.k = k
        self.k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def segment(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        h, w = roi_bgr.shape[:2]
        if h < 10 or w < 10:
            return None

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        hs = hsv[:, :, :2].reshape(-1, 2).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
        _, labels, _ = cv2.kmeans(hs, self.k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels_2d = labels.reshape(h, w)

        center_label = int(labels_2d[h//2, w//2])
        mask = np.where(labels_2d == center_label, 255, 0).astype(np.uint8)

        gray  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edges_d = cv2.dilate(edges, self.k_open, iterations=1)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(edges_d)) | mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        return self._largest_cc(mask)

    @staticmethod
    def _largest_cc(mask):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return mask
        best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return np.where(labels == best, 255, 0).astype(np.uint8)


class TemporalStabilizer:
    """Farneback optical flow mask warping + weighted blend."""

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_mask: Optional[np.ndarray] = None
        self._fb = dict(pyr_scale=0.5, levels=2, winsize=13,
                        iterations=2, poly_n=5, poly_sigma=1.1, flags=0)

    def update(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_mask = mask.copy()
            return mask

        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, **self._fb)
        h, w = gray.shape
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
        mx = (gx + flow[..., 0]).clip(0, w-1)
        my = (gy + flow[..., 1]).clip(0, h-1)
        warped = cv2.remap(self.prev_mask.astype(np.float32), mx, my, cv2.INTER_LINEAR)
        blended = self.alpha * mask.astype(np.float32) + (1-self.alpha) * warped
        stable = np.where(blended > 127, 255, 0).astype(np.uint8)
        self.prev_gray = gray.copy()
        self.prev_mask = stable.copy()
        return stable

    def reset(self):
        self.prev_gray = None
        self.prev_mask = None


class ExtractionResult:
    """Holds all outputs for a single processed frame."""
    __slots__ = ['bbox', 'roi_bgr', 'mask_raw', 'mask_stable',
                 'full_frame', 'has_object', 'tracker_name']

    def __init__(self):
        self.bbox         = None   # (x1, y1, x2, y2) in frame coords
        self.roi_bgr      = None   # Cropped BGR image
        self.mask_raw     = None   # K-means mask
        self.mask_stable  = None   # Temporally stabilised mask
        self.full_frame   = None   # Annotated display frame
        self.has_object   = False
        self.tracker_name = "none"

    def to_full_mask(self, frame_shape: Tuple[int,int]) -> Optional[np.ndarray]:
        """Project ROI mask back into full frame coordinates."""
        if self.mask_stable is None or self.bbox is None:
            return None
        H, W = frame_shape
        full = np.zeros((H, W), dtype=np.uint8)
        x1, y1, x2, y2 = self.bbox
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(W,x2), min(H,y2)
        if x2 > x1 and y2 > y1:
            resized = cv2.resize(self.mask_stable, (x2-x1, y2-y1))
            full[y1:y2, x1:x2] = resized
        return full


class ObjectExtractor:
    """
    Main extraction pipeline:
      Camera frame → YOLO (sparse) → Tracker (dense) →
      Segmentation → Temporal stabilization → ExtractionResult
    """

    def __init__(self, target_label: str, hw_profile,
                 detect_interval: int = 15, kmeans_k: int = 2,
                 temporal_alpha: float = 0.6):

        self.target_class = resolve_class(target_label)
        self.hw = hw_profile
        self.detect_interval = detect_interval

        self.segmenter   = ObjectSegmenter(k=kmeans_k)
        self.stabilizer  = TemporalStabilizer(alpha=temporal_alpha)

        self._detector   = None   # Lazy-loaded
        self._tracker    = None
        self._tracker_name = "none"
        self._tracker_active = False
        self._tracking_xywh  = None

        self._frame_count = 0

        log.info(f"ObjectExtractor: target='{self.target_class}'"
                 f"  detect_every={detect_interval}f")

    def _ensure_detector(self):
        if self._detector is not None:
            return
        try:
            from ultralytics import YOLO
            self._detector = YOLO("yolov8n.pt")
            log.info("YOLOv8n loaded.")
        except ImportError:
            log.warning("ultralytics not installed — detection disabled.")
            self._detector = None

    def _detect(self, frame: np.ndarray) -> Optional[Tuple]:
        self._ensure_detector()
        if self._detector is None:
            return None

        H, W = frame.shape[:2]
        ds = self.hw.det_size
        sx, sy = W / ds, H / ds
        small = cv2.resize(frame, (ds, ds))

        name_to_id = {v: k for k, v in self._detector.names.items()}
        cid = name_to_id.get(self.target_class)
        if cid is None:
            log.warning(f"'{self.target_class}' not in COCO. Available: "
                        f"{list(name_to_id.keys())[:10]}...")
            return None

        results = self._detector(small, classes=[cid], conf=0.35,
                                 verbose=False, imgsz=ds)
        best, best_conf = None, 0.0
        for r in results:
            for box in r.boxes:
                c = float(box.conf[0])
                if c > best_conf:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    best = (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy), c)
                    best_conf = c
        return best

    def process(self, frame: np.ndarray) -> ExtractionResult:
        result = ExtractionResult()
        result.full_frame = frame.copy()
        H, W = frame.shape[:2]
        self._frame_count += 1

        # ── YOLO detection (sparse) ───────────────────────────────────────
        run_detect = (self._frame_count % self.detect_interval == 0
                      or not self._tracker_active)

        if run_detect:
            det = self._detect(frame)
            if det is not None:
                x1, y1, x2, y2, conf = det
                x1, y1 = max(0,x1), max(0,y1)
                x2, y2 = min(W,x2), min(H,y2)
                bw, bh = x2-x1, y2-y1
                if bw > 15 and bh > 15:
                    self._tracking_xywh = (x1, y1, bw, bh)
                    self._tracker, self._tracker_name = create_tracker()
                    self._tracker.init(frame, self._tracking_xywh)
                    self._tracker_active = True
                    self.stabilizer.reset()
                    log.debug(f"YOLO: '{self.target_class}' conf={conf:.2f}")

        result.tracker_name = self._tracker_name

        # ── Tracker update ────────────────────────────────────────────────
        if self._tracker_active:
            ok, new_bbox = self._tracker.update(frame)
            if ok:
                tx, ty, tw, th = [int(v) for v in new_bbox]
                tx2, ty2 = min(W, tx+tw), min(H, ty+th)
                tx, ty = max(0,tx), max(0,ty)
                if tx2 > tx and ty2 > ty:
                    result.bbox = (tx, ty, tx2, ty2)
                    roi = frame[ty:ty2, tx:tx2].copy()
                    result.roi_bgr = roi

                    # ── Segmentation ──────────────────────────────────────
                    raw_mask = self.segmenter.segment(roi)
                    result.mask_raw = raw_mask

                    if raw_mask is not None:
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        stable   = self.stabilizer.update(roi_gray, raw_mask)
                        result.mask_stable = stable

                    result.has_object = True
                    self._annotate(result)
            else:
                self._tracker_active = False
                self.stabilizer.reset()
                log.debug("Tracker lost object.")

        return result

    def _annotate(self, result: ExtractionResult):
        """Draw bounding box + mask overlay on full_frame."""
        if result.bbox is None:
            return
        x1, y1, x2, y2 = result.bbox
        f = result.full_frame
        cv2.rectangle(f, (x1,y1), (x2,y2), (0,230,100), 2)
        cv2.putText(f, f"{self.target_class} [{result.tracker_name}]",
                    (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,230,100), 2)

        if result.mask_stable is not None:
            rh, rw = y2-y1, x2-x1
            if rh > 0 and rw > 0:
                m_resized = cv2.resize(result.mask_stable, (rw, rh))
                overlay   = np.zeros_like(f[y1:y2, x1:x2])
                overlay[:,:,1] = 160
                alpha_ch = (m_resized[...,None]/255.0) * 0.38
                f[y1:y2, x1:x2] = (
                    f[y1:y2,x1:x2].astype(float)*(1-alpha_ch) +
                    overlay.astype(float)*alpha_ch
                ).astype(np.uint8)

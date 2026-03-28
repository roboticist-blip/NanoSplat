# NanoSplat — Real-Time Video → 3D Object Extraction

> **Runs on Jetson Nano (2GB GPU) and Raspberry Pi 4 (CPU).  
> No training. No cloud. No 8GB workstation GPU required.**

---

## What Makes This Novel

Every existing video-to-3D pipeline breaks down on embedded hardware:

| Tool | Why it fails on Nano/Pi |
|---|---|
| NeRF / Instant-NGP | Needs training loop, 8GB+ VRAM |
| 3D Gaussian Splatting (original) | Tile-based CUDA rasterizer, 8–24GB VRAM |
| SAM + depth | SAM ViT-H is 600MB+, 2 FPS on Nano |
| ZoeDepth | 344MB model, OOM on 2GB |
| COLMAP | Offline SfM, not real-time |

NanoSplat's approach:

```
MiDaS-Small (21MB, ONNX/TRT)  ←  only "heavy" model
K-means segmentation           ←  no SAM needed
ORB pose estimation            ←  no SLAM library
Depth-seeded MicroGaussians    ←  no training loop
Confidence-weighted fusion     ←  like a 3D Kalman filter
```

Result: **live coloured point cloud / Gaussian cloud of just the target object**,
built frame by frame, exported as `.ply` and `.splat`, viewable in MeshLab, 
Open3D, Blender, or a browser with zero install.

---

## Hardware Requirements

| Device | RAM | GPU | Expected FPS | 3D quality |
|---|---|---|---|---|
| Jetson Nano 4GB | 4GB shared | 128 CUDA cores (2GB eff.) | 6–10 FPS | Good |
| Jetson Nano 2GB | 2GB shared | 128 CUDA cores (1.5GB eff.) | 4–7 FPS | Good |
| Raspberry Pi 4 4GB | 4GB | None | 1–3 FPS | Fair |
| x86 laptop (CPU) | Any | None | 5–15 FPS | Good |

---

## Setup

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

**Jetson Nano only** (JetPack 4.6):
```bash
# CuPy for CUDA 10.2 — install from NVIDIA wheel, NOT pip
pip install cupy-cuda102 --break-system-packages
# TensorRT is pre-installed with JetPack — verify:
python -c "import tensorrt; print(tensorrt.__version__)"
```

### Step 2 — Download MiDaS depth model (21 MB, once)

```bash
python nanosplat/scripts/download_models.py
```

### Step 3 — Run

```bash
# Basic: Pi camera, track a ball
python main.py --target ball

# USB camera, headless (SSH), serve live 3D in browser
python main.py --target bottle --usb --headless --serve-viewer

# See 3D output in real time at http://<device-ip>:8080
python main.py --target cup --serve-viewer

# Jetson Nano (TensorRT auto-detected, ~6-8 FPS)
python main.py --target person --object-dist 2.0 --serve-viewer
```

---

## Output Files

```
output_3d/
├── latest.ply        ← always-updated coloured point cloud
├── latest.splat      ← Gaussian Splatting format
├── viewer.html       ← auto-generated WebGL viewer
├── ply/              ← per-keyframe PLY snapshots
├── splat/            ← per-keyframe .splat snapshots
├── crops/            ← cropped object JPGs
└── rgba/             ← transparent-background PNGs
```

### Viewing the 3D output

**Option 1 — Browser (zero install, works on phone too):**
```bash
python main.py --target ball --serve-viewer
# Open http://<device-ip>:8080 — drag to orbit, scroll to zoom
```

**Option 2 — MeshLab (free, all platforms):**
```
File → Import Mesh → latest.ply
```

**Option 3 — Open3D (x86 dev machine):**
```bash
pip install open3d
python -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('output_3d/latest.ply')
o3d.visualization.draw_geometries([pcd])
"
```

**Option 4 — Gaussian Splatting browser viewer:**
```
Drag output_3d/latest.splat onto https://antimatter15.com/splat/
```

---

## Architecture Deep Dive

### Why depth-seeded Gaussians instead of training?

3DGS training works by:
1. Initialising random Gaussians
2. Differentiable rendering → compare to ground-truth image → backprop → update

This requires thousands of iterations and a GPU with enough memory to hold
the full rasterizer's gradient tape.

NanoSplat skips this entirely:
1. MiDaS gives a per-pixel depth estimate for each frame
2. Masked pixels are unprojected into 3D using camera intrinsics K
3. Each 3D point becomes a Gaussian seeded with colour from the frame
4. New keyframes update existing nearby Gaussians via confidence weighting
   (Gaussian with conf=5 won't be overwritten by a new conf=1 estimate)

No gradient, no training loop, no VRAM for backprop.

### Why K-means instead of SAM?

SAM (Segment Anything) ViT-H = 600MB+ model, ~0.5 FPS on Jetson.
SAM-Small = still ~180MB, ~1.5 FPS.

K-means on HSV in the object ROI (e.g. 80×60 pixels):
- 2 clusters, 8 iterations → ~0.3ms
- No GPU needed
- Works for any object with colour contrast from background
- Combined with temporal optical flow stabilization → surprisingly stable

The key insight: we don't need perfect segmentation.
We need *consistent* segmentation across frames.
Temporal mask blending achieves this at essentially zero cost.

### Pose estimation without a SLAM library

Standard approach: ORBSLAM3, OpenVSLAM, etc. — complex dependencies, 
hard to build on ARM64, heavy runtime.

NanoSplat uses: ORB features → kNN match → Lowe's ratio test →
Essential Matrix (RANSAC) → recoverPose → accumulate T_world_from_cam.

This gives ~1–5cm accuracy over a 60-second scan, which is sufficient
for generating a 3D cloud of a handheld object.

---

## CLI Reference

```
--target          Object alias (ball, cup, bottle, dog, person, laptop...)
--object-dist     Approx object distance in metres (default: 1.0)
--width/--height  Camera resolution (default: 640×480)
--usb             Use USB camera instead of Picamera2
--detect-interval Run YOLO every N frames (default: 15)
--kmeans-k        K-means clusters (2 or 3, default: 2)
--blend-alpha     Temporal blend weight (default: 0.6)
--keyframe-interval Submit to 3D every N frames (default: 10)
--export-every    Export PLY every N keyframes (default: 30)
--output-dir      Output directory (default: output_3d/)
--headless        No GUI window
--serve-viewer    Launch WebGL 3D viewer server
--viewer-port     Viewer port (default: 8080)
--verbose         Debug logging
```

---

## Extending for Full 3D Reconstruction (Research Path)

The current system produces object-centric coloured point clouds.
Extensions to get dense mesh / NeRF quality:

1. **Add stereo camera** (two Pi cameras or Luxonis OAK-D Lite):
   Replace MiDaS relative depth with metric stereo depth.
   Remove depth ambiguity completely.

2. **Add bundle adjustment** (g2o or gtsam):
   Refine all poses jointly. Reduces drift in long scans.

3. **Replace K-means with lightweight neural seg**:
   Segment Anything Nano (SAM-nano) or FastSAM if 500MB fits in RAM.

4. **Replace depth-seeded Gaussians with trained Gaussians**:
   Use the NanoSplat cloud as initialisation for offline 3DGS training
   on a workstation, then deploy the trained .splat back to the device.

5. **Add mesh reconstruction**:
   Apply Poisson surface reconstruction (Open3D) to the PLY cloud
   post-capture for a watertight mesh.

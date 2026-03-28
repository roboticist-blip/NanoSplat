#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        NanoSplat v1.0                                      ║
║         Real-Time Video → 3D Object Extraction Pipeline                     ║
║         Designed for Jetson Nano (2GB GPU) & Raspberry Pi 4 (CPU)          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  NOVEL ARCHITECTURE:                                                         ║
║                                                                              ║
║  Video Frame ──► YOLOv8n (sparse) ──► MOSSE Tracker ──► K-means Seg        ║
║                                                              │               ║
║                                            Farneback OF ◄───┘               ║
║                                                 │                           ║
║                                         Stable Mask (2D)                   ║
║                                                 │                           ║
║            MiDaS-Small (ONNX/TRT) ──► Depth ───┤                           ║
║            ORB Feature Matching   ──► Pose  ───┤                           ║
║                                                 │                           ║
║                              Gaussian Seeding ◄─┘                          ║
║                                  (no training!)                             ║
║                                      │                                      ║
║              Confidence-Weighted Cloud Fusion                               ║
║                                      │                                      ║
║              PLY + .splat Export ────┘                                      ║
║              (MeshLab / Open3D / Browser)                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

QUICK START:
  python nanosplat/scripts/download_models.py    # once
  python main.py --target ball                   # run
  python main.py --target bottle --serve-viewer  # run + live browser 3D view
"""

import sys
import os
import time
import argparse
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nanosplat.core.hardware import detect_hardware, XP, Backend
from nanosplat.tracker.extractor import ObjectExtractor
from nanosplat.reconstruction.orchestrator import ReconstructionOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("NanoSplat")


# ── Camera abstraction ────────────────────────────────────────────────────────

class CameraSource:
    def __init__(self, use_picamera2: bool, src: int, width: int, height: int):
        self.width, self.height = width, height
        self._picam = None
        self._cap   = None

        if use_picamera2:
            try:
                from picamera2 import Picamera2
                self._picam = Picamera2()
                cfg = self._picam.create_preview_configuration(
                    main={"size": (width, height), "format": "BGR888"}
                )
                self._picam.configure(cfg)
                self._picam.start()
                log.info("Camera: Picamera2")
                return
            except Exception as e:
                log.warning(f"Picamera2 unavailable ({e}) — using USB camera.")

        self._cap = cv2.VideoCapture(src)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        log.info(f"Camera: USB/V4L2 (src={src})")

    def read(self):
        if self._picam:
            return self._picam.capture_array()
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self):
        if self._picam: self._picam.stop()
        if self._cap:   self._cap.release()


# ── HUD ──────────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, fps: float, stats: dict, hw_name: str):
    lines = [
        f"NanoSplat  FPS:{fps:.1f}",
        f"HW: {hw_name}",
        f"KF: {stats.get('keyframes', 0)}  Gauss: {stats.get('gaussians', 0)}",
        f"Fill: {stats.get('fill_pct', 0):.0f}%  "
        f"Lat: {stats.get('latency_ms', 0):.0f}ms",
    ]
    for i, line in enumerate(lines):
        y = 12 + 20 * (i + 1)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 255, 180), 1, cv2.LINE_AA)


# ── Minimal live WebGL viewer ─────────────────────────────────────────────────

VIEWER_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>NanoSplat Live 3D Viewer</title>
<style>
body { margin:0; background:#080c10; overflow:hidden;
       font-family:'Courier New',monospace; }
#hud { position:fixed; top:14px; left:14px; z-index:10;
       background:rgba(0,0,0,0.72); padding:10px 16px;
       border-left:3px solid #00ffb0; border-radius:2px; }
#hud h2 { margin:0 0 4px; color:#00ffb0; font-size:13px; }
#status  { font-size:11px; color:#888; }
#tip     { font-size:10px; color:#555; margin-top:6px; }
canvas   { display:block; }
</style>
</head>
<body>
<div id="hud">
  <h2>NanoSplat Live 3D</h2>
  <div id="status">Loading cloud…</div>
  <div id="tip">Drag to orbit · Scroll to zoom</div>
</div>
<canvas id="c"></canvas>
<script>
const c = document.getElementById('c');
c.width = window.innerWidth; c.height = window.innerHeight;
const gl = c.getContext('webgl');
if (!gl) { document.body.innerHTML = '<h2 style="color:red;padding:40px">WebGL not supported</h2>'; }

const VS=`attribute vec3 p;attribute vec3 col;uniform mat4 mvp;varying vec3 vc;
void main(){gl_Position=mvp*vec4(p,1);gl_PointSize=2.5;vc=col;}`;
const FS=`precision mediump float;varying vec3 vc;
void main(){float d=length(gl_PointCoord-vec2(.5));if(d>.5)discard;gl_FragColor=vec4(vc,1);}`;

function sh(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);return s;}
const prog=gl.createProgram();
gl.attachShader(prog,sh(gl.VERTEX_SHADER,VS));
gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,FS));
gl.linkProgram(prog);gl.useProgram(prog);
const ap=gl.getAttribLocation(prog,'p'),ac=gl.getAttribLocation(prog,'col');
const umvp=gl.getUniformLocation(prog,'mvp');
let pb=gl.createBuffer(),cb=gl.createBuffer(),nPts=0;

async function loadPLY(){
  try{
    const r=await fetch('/latest.ply?t='+Date.now());
    if(!r.ok){document.getElementById('status').textContent='Waiting for data…';return;}
    const buf=await r.arrayBuffer();
    const txt=new TextDecoder().decode(buf.slice(0,2048));
    const he=txt.indexOf('end_header\\n')+'end_header\\n'.length;
    const nm=txt.match(/element vertex (\\d+)/);
    if(!nm)return;
    const n=parseInt(nm[1]);
    // stride: pos(12)+normal(12)+rgb(3)+conf(4)=31 bytes
    const dv=new DataView(buf,he);
    const pos=new Float32Array(n*3),col=new Float32Array(n*3);
    for(let i=0;i<n;i++){
      const o=i*31;
      pos[i*3]=dv.getFloat32(o,true);pos[i*3+1]=dv.getFloat32(o+4,true);pos[i*3+2]=dv.getFloat32(o+8,true);
      col[i*3]=dv.getUint8(o+24)/255;col[i*3+1]=dv.getUint8(o+25)/255;col[i*3+2]=dv.getUint8(o+26)/255;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER,pb);gl.bufferData(gl.ARRAY_BUFFER,pos,gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER,cb);gl.bufferData(gl.ARRAY_BUFFER,col,gl.DYNAMIC_DRAW);
    nPts=n;
    document.getElementById('status').textContent=n+' Gaussians';
  }catch(e){document.getElementById('status').textContent='Error: '+e.message;}
}

let th=0.4,ph=0.25,rad=3,drag=false,lx=0,ly=0;
c.addEventListener('mousedown',e=>{drag=true;lx=e.clientX;ly=e.clientY;});
c.addEventListener('mouseup',()=>drag=false);
c.addEventListener('mousemove',e=>{if(!drag)return;th-=(e.clientX-lx)*0.005;ph-=(e.clientY-ly)*0.005;ph=Math.max(-1.5,Math.min(1.5,ph));lx=e.clientX;ly=e.clientY;});
c.addEventListener('wheel',e=>rad=Math.max(0.3,rad+e.deltaY*0.002));

// Touch support
c.addEventListener('touchstart',e=>{drag=true;lx=e.touches[0].clientX;ly=e.touches[0].clientY;});
c.addEventListener('touchend',()=>drag=false);
c.addEventListener('touchmove',e=>{e.preventDefault();th-=(e.touches[0].clientX-lx)*0.005;ph-=(e.touches[0].clientY-ly)*0.005;lx=e.touches[0].clientX;ly=e.touches[0].clientY;},{passive:false});

const norm=v=>{const l=Math.hypot(...v);return v.map(x=>x/l);};
const sub=(a,b)=>a.map((x,i)=>x-b[i]);
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const cross=([ax,ay,az],[bx,by,bz])=>[ay*bz-az*by,az*bx-ax*bz,ax*by-ay*bx];

function perspective(fov,asp,n,f){
  const t=1/Math.tan(fov/2),r=f-n;
  return new Float32Array([t/asp,0,0,0,0,t,0,0,0,0,-(f+n)/r,-1,0,0,-2*f*n/r,0]);
}
function lookAt(eye,ctr,up){
  const z=norm(sub(eye,ctr)),x=norm(cross(up,z)),y=cross(z,x);
  return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]);
}
function mul4(a,b){const o=new Float32Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[i+4*k]*b[k+4*j];o[i+4*j]=s;}return o;}

function render(){
  gl.viewport(0,0,c.width,c.height);
  gl.clearColor(0.03,0.05,0.07,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  if(nPts>0){
    const eye=[rad*Math.cos(ph)*Math.sin(th),rad*Math.sin(ph),rad*Math.cos(ph)*Math.cos(th)];
    const mvp=mul4(perspective(0.8,c.width/c.height,0.01,100),lookAt(eye,[0,0,0],[0,1,0]));
    gl.uniformMatrix4fv(umvp,false,mvp);
    gl.bindBuffer(gl.ARRAY_BUFFER,pb);gl.enableVertexAttribArray(ap);gl.vertexAttribPointer(ap,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,cb);gl.enableVertexAttribArray(ac);gl.vertexAttribPointer(ac,3,gl.FLOAT,false,0,0);
    gl.drawArrays(gl.POINTS,0,nPts);
  }
  requestAnimationFrame(render);
}

loadPLY();setInterval(loadPLY,3000);render();
window.addEventListener('resize',()=>{c.width=window.innerWidth;c.height=window.innerHeight;});
</script>
</body>
</html>
"""


def start_viewer_server(output_dir: str, port: int = 8080):
    import http.server
    import socketserver

    viewer_path = Path(output_dir) / "viewer.html"
    viewer_path.write_text(VIEWER_HTML)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=output_dir, **kwargs)
        def do_GET(self):
            if self.path.split("?")[0] in ("/", ""):
                self.path = "/viewer.html"
            super().do_GET()
        def log_message(self, *a):
            pass

    srv = socketserver.TCPServer(("", port), Handler)
    srv.allow_reuse_address = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    import socket
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "localhost"

    log.info(f"3D Viewer:  http://localhost:{port}  or  http://{ip}:{port}")
    return srv


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(args):
    hw = detect_hardware()

    cam = CameraSource(
        use_picamera2 = not args.usb,
        src=args.src, width=args.width, height=args.height,
    )

    extractor = ObjectExtractor(
        target_label    = args.target,
        hw_profile      = hw,
        detect_interval = args.detect_interval,
        kmeans_k        = args.kmeans_k,
        temporal_alpha  = args.blend_alpha,
    )

    recon = ReconstructionOrchestrator(
        hw_profile        = hw,
        output_dir        = args.output_dir,
        object_dist_m     = args.object_dist,
        keyframe_interval = args.keyframe_interval,
        auto_export_every = args.export_every,
    )
    recon.start(args.width, args.height)

    if args.serve_viewer:
        start_viewer_server(args.output_dir, args.viewer_port)

    fps_timer, fps_counter, fps = time.time(), 0, 0.0
    frame_id = 0

    log.info(f"\n{'='*62}")
    log.info(f"  NanoSplat  |  Target: '{args.target}'"
             f"  →  '{extractor.target_class}'")
    log.info(f"  Hardware : {hw.device_name}  [{hw.backend.name}]")
    log.info(f"  Output   : {args.output_dir}/")
    if args.serve_viewer:
        log.info(f"  3D View  : http://localhost:{args.viewer_port}")
    log.info(f"  Keys     : [q] quit  [r] reset tracker  [e] export now")
    log.info(f"{'='*62}\n")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_id    += 1
            fps_counter += 1
            result = extractor.process(frame)
            recon.submit_frame(frame, result, frame_id)

            elapsed = time.time() - fps_timer
            if elapsed >= 2.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer   = time.time()

            if not args.headless:
                draw_hud(result.full_frame, fps, recon.get_stats(), hw.device_name)
                cv2.imshow("NanoSplat", result.full_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    extractor.stabilizer.reset()
                    log.info("Tracker reset.")
                elif key == ord('e'):
                    recon._export_cloud(frame_id)
                    log.info("Manual export done.")
            else:
                if frame_id % max(1, int(fps or 3) * 10) == 0:
                    s = recon.get_stats()
                    log.info(f"FPS={fps:.1f}  KF={s.get('keyframes',0)}"
                             f"  G={s.get('gaussians',0)}"
                             f"  Fill={s.get('fill_pct',0):.0f}%")

    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        recon.stop()
        cam.release()
        cv2.destroyAllWindows()

        s = recon.get_stats()
        log.info(f"\n{'='*62}")
        log.info(f"  Done.")
        log.info(f"  Keyframes : {s.get('keyframes',0)}")
        log.info(f"  Gaussians : {s.get('gaussians',0)}")
        log.info(f"  PLY file  : {args.output_dir}/latest.ply")
        log.info(f"  Splat file: {args.output_dir}/latest.splat")
        log.info(f"\n  View in Open3D:")
        log.info(f"    python -c \"import open3d as o3d; "
                  f"o3d.visualization.draw_geometries(["
                  f"o3d.io.read_point_cloud('{args.output_dir}/latest.ply')])\"")
        log.info(f"\n  View .splat in browser:")
        log.info(f"    https://antimatter15.com/splat/")
        log.info(f"{'='*62}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="NanoSplat — video-to-3D object extraction on Jetson Nano / RPi 4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("Object")
    g.add_argument("--target",           default="bottle",
                   help="Object to extract (ball, cup, bottle, dog, person…)")
    g.add_argument("--object-dist",      type=float, default=1.0,
                   help="Approx. object distance in metres (scales depth)")

    g = p.add_argument_group("Camera")
    g.add_argument("--width",  type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--usb",    action="store_true", help="Force USB / V4L2 camera")
    g.add_argument("--src",    type=int, default=0)

    g = p.add_argument_group("Extraction tuning")
    g.add_argument("--detect-interval", type=int,   default=15)
    g.add_argument("--kmeans-k",        type=int,   default=2)
    g.add_argument("--blend-alpha",     type=float, default=0.6)

    g = p.add_argument_group("3D Reconstruction")
    g.add_argument("--keyframe-interval", type=int, default=10)
    g.add_argument("--export-every",      type=int, default=30,
                   help="Auto-export PLY + .splat every N keyframes")
    g.add_argument("--output-dir",        default="output_3d")

    g = p.add_argument_group("Display")
    g.add_argument("--headless",     action="store_true")
    g.add_argument("--serve-viewer", action="store_true",
                   help="Launch live WebGL 3D viewer in browser")
    g.add_argument("--viewer-port",  type=int, default=8080)
    g.add_argument("--verbose",      action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# --------------------------------------------------------
# Serve a directory of .glb files in a browser-based 3D viewer, so results
# produced on the cluster (e.g. by visualize_depth.py) can be inspected over an
# SSH port-forward without downloading anything.
#
# The server is a plain stdlib HTTP server -- no GPU, no project imports -- so
# run it on a login node. The viewer page (three.js: OrbitControls + a
# point-cloud-aware GLTFLoader) renders in your LOCAL browser, which fetches
# three.js from a CDN; the cluster side only serves the .glb bytes.
#
# Usage:
#   cd src
#   python serve_glb.py --glb-dir ../checkpoints/metric_depth_cond_<id>/viz
#   # (a run/viz dir works too -- it auto-descends into viz/ and glb/)
#
# Then open the port in your browser:
#   * VS Code Remote-SSH: the Ports panel auto-detects the port -> click the
#     forwarded localhost link (nothing else to do).
#   * Manual tunnel from your laptop:
#       ssh -L 8000:localhost:8000 jdosch@ssh.ccv.brown.edu
#     then browse to http://localhost:8000
#     (CCV has several login nodes: if localhost forwarding lands on a
#      different node, tunnel to the hostname this script prints instead.)
# --------------------------------------------------------
import argparse
import json
import socket
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# three.js version pinned so the importmap and addon paths stay in lockstep.
_THREE = "0.160.0"

INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GLB viewer</title>
<style>
  html, body { margin: 0; height: 100%; background: #1a1a1e; color: #ddd;
               font: 13px/1.4 system-ui, sans-serif; overflow: hidden; }
  #bar { position: fixed; top: 0; left: 0; right: 0; z-index: 10;
         display: flex; gap: 14px; align-items: center; padding: 8px 12px;
         background: rgba(20,20,24,.82); backdrop-filter: blur(6px); }
  #bar select, #bar input { font: inherit; }
  #bar label { display: flex; gap: 6px; align-items: center; color: #aaa; }
  #info { margin-left: auto; color: #888; }
  #err { color: #ff8080; }
  button { font: inherit; background:#333; color:#ddd; border:1px solid #555;
           border-radius:4px; padding:3px 9px; cursor:pointer; }
  button:hover { background:#444; }
  canvas { display: block; }
</style>
<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/three@__THREE__/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@__THREE__/examples/jsm/"
}}
</script>
</head>
<body>
<div id="bar">
  <label>scene <select id="file"></select></label>
  <label>point size <input id="size" type="range" min="1" max="20" step="0.5" value="4"></label>
  <button id="fit">fit view</button>
  <button id="bg">bg</button>
  <button id="cams">cameras</button>
  <label><input id="follow" type="checkbox"> follow GT cam</label>
  <span id="framectl" style="display:none; gap:6px; align-items:center;">
    <button id="prev">&#9664;</button>
    <input id="frame" type="range" min="0" max="0" step="1" value="0" style="width:160px">
    <button id="next">&#9654;</button>
    <button id="play">play</button>
    <label>fps <input id="fps" type="range" min="1" max="30" step="1" value="10" style="width:70px"></label>
    <label><input id="accum" type="checkbox" checked> accumulate</label>
    <span id="fnum" style="color:#aaa"></span>
  </span>
  <span id="info"></span>
</div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// antialias off: MSAA multiplies fill cost, and this renderer is fill-bound on
// millions of square point sprites that MSAA barely improves anyway.
const renderer = new THREE.WebGLRenderer({ antialias: false });
// Cap the pixel ratio: this is a fill-rate-bound point cloud, and rendering at
// devicePixelRatio 2 quadruples the fragments AND doubles the point width needed
// to close the inter-point gaps -- a ~4x cost for detail that a stippled cloud
// does not show off anyway.
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const bgColors = [0x1a1a1e, 0xffffff, 0x000000];
let bgIdx = 0;
scene.background = new THREE.Color(bgColors[bgIdx]);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.2));

const camera = new THREE.PerspectiveCamera(55, window.innerWidth/window.innerHeight, 0.01, 5000);
camera.position.set(2, 1.5, 2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
// Finer wheel steps, and dolly toward the POINTER rather than the orbit
// target -- so zooming into a detail does not require recentering on it first.
controls.zoomSpeed = 0.4;
controls.zoomToCursor = true;

// Render on demand rather than every rAF tick: a static 6.5M-point scene costs
// the same to redraw as a moving one, so continuously re-rendering an unchanged
// view is pure waste and makes interaction feel sluggish.
let needsRender = true;
function requestRender() { needsRender = true; }
controls.addEventListener('change', requestRender);

const loader = new GLTFLoader();
let current = null;                 // currently displayed gltf.scene
let pointMats = [];                  // point-cloud materials (for the size slider)
let pointObjs = [];                  // the point clouds themselves (fitView bounds THESE)
let camObjs = [];                    // camera frustum meshes (toggleable, never framed)
let camsVisible = true;
let camPoses = new Map();            // frame index -> Object3D carrying GT cam->world
let frameGroups = new Map();         // frame index -> [Object3D] (per-frame point clouds)
let frameMax = -1;                   // highest frame index, or -1 if not a per-frame scene
let playTimer = null;
const sizeEl = document.getElementById('size');
const infoEl = document.getElementById('info');
const frameCtl = document.getElementById('framectl');
const frameEl = document.getElementById('frame');
const fnumEl = document.getElementById('fnum');
const accumEl = document.getElementById('accum');
const followEl = document.getElementById('follow');
let sceneSpan = 1;                   // point-cloud extent, sets the follow-cam target distance

function stopPlay() {
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  document.getElementById('play').textContent = 'play';
}

// Parsed scenes are cached (name -> gltf.scene) and NEVER disposed, so GPU
// buffers stay resident and dropdown flips (base <-> finetuned A/B) are
// instant instead of re-downloading + re-parsing a ~100MB GLB each time.
const sceneCache = new Map();

function detachCurrent() {
  if (!current) return;
  scene.remove(current);   // no dispose: the cache owns the buffers
  current = null; pointMats = []; pointObjs = []; camObjs = [];
  frameGroups = new Map(); frameMax = -1; camPoses = new Map();
}

function fitView() {
  if (!current) return;
  // Frame the POINT CLOUDS only. The camera frusta trace the trajectory, which
  // necessarily sits outside the geometry it observes -- including them in the
  // bounds zooms out until the reconstruction is a speck in the middle.
  const box = new THREE.Box3();
  if (pointObjs.length) pointObjs.forEach(o => box.expandByObject(o));
  else box.setFromObject(current);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  sceneSpan = maxDim;
  const dist = 1.6 * maxDim / (2 * Math.tan((camera.fov * Math.PI/180) / 2));
  camera.near = Math.max(dist/1000, 1e-3);
  camera.far = dist * 1000;
  // zoom clamps scaled to the scene: deep enough to inspect single points,
  // capped so a runaway scroll cannot fling the view into the far plane
  controls.minDistance = maxDim * 0.002;
  controls.maxDistance = dist * 8;
  camera.position.copy(center).add(new THREE.Vector3(dist, dist*0.6, dist));
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function attach(root) {
    detachCurrent();
    current = root;
    let nPts = 0;
    current.traverse(o => {
      // GT camera-pose markers: not scene content. Capture and hide them BEFORE
      // the isPoints branch, else they join pointObjs and skew fitView.
      const cm = /^campose_(\d+)/.exec(o.name || '');
      if (cm) {
        o.visible = false;
        camPoses.set(parseInt(cm[1], 10), o);
        return;
      }
      if (o.isPoints) {
        // trimesh exports vertex colors as COLOR_0 -> GLTFLoader already sets
        // vertexColors; screen-space size (sizeAttenuation off) keeps points
        // legible regardless of the scene's metric scale.
        o.material.size = pointPx();
        o.material.sizeAttenuation = false;
        o.material.needsUpdate = true;
        pointMats.push(o.material);
        pointObjs.push(o);
        nPts += o.geometry.attributes.position.count;
      } else if (o.isMesh) {
        // camera frusta: triangle meshes, not part of any frame_NNN group
        camObjs.push(o);
        o.visible = camsVisible;
      }
      // per-frame point clouds are named frame_000, frame_001, ... by the
      // exporter; group them so the slider can toggle frame visibility
      const m = /^frame_(\\d+)/.exec(o.name || '');
      if (m) {
        const idx = parseInt(m[1], 10);
        if (!frameGroups.has(idx)) frameGroups.set(idx, []);
        frameGroups.get(idx).push(o);
        if (idx > frameMax) frameMax = idx;
      }
    });
    scene.add(current);
    fitView();  // fit once to the whole clip so stepping frames doesn't jump
    infoEl.textContent = nPts.toLocaleString() + ' points';
    if (frameMax >= 0) {
      frameEl.max = frameMax;
      frameEl.value = frameMax;        // default: show all frames (accumulate)
      frameCtl.style.display = 'inline-flex';
      applyFrames();
    } else {
      frameCtl.style.display = 'none'; // fused/legacy scene: no per-frame nodes
    }
    requestRender();
}

function load(name) {
  stopPlay();
  const cached = sceneCache.get(name);
  if (cached) { attach(cached); return; }   // instant: buffers already on GPU
  infoEl.textContent = 'loading ' + name + ' ...';
  infoEl.classList.remove('err'); infoEl.id = 'info';
  loader.load(encodeURIComponent(name), (gltf) => {
    sceneCache.set(name, gltf.scene);
    attach(gltf.scene);
  }, undefined, (e) => {
    infoEl.textContent = 'failed to load ' + name + ' (' + e + ')';
    infoEl.className = 'err';
  });
}

function applyFrames() {
  if (frameMax < 0) return;
  const k = parseInt(frameEl.value, 10);
  const accumulate = accumEl.checked;
  for (const [idx, objs] of frameGroups) {
    const vis = accumulate ? idx <= k : idx === k;
    objs.forEach(o => { o.visible = vis; });
  }
  fnumEl.textContent = 'frame ' + k + '/' + frameMax;
  if (followEl.checked) gotoCam(k);
  requestRender();   // covers the slider, the play timer and the arrow keys
}

// Place the view camera at frame k's GT camera. Viewing the reconstruction from
// the pose that produced it is what makes frame-to-frame drift legible: the
// scene should track rigidly as you scrub, and anything that swims is the depth
// disagreeing between frames rather than your own viewpoint moving.
const CV_TO_GL = new THREE.Quaternion().setFromRotationMatrix(
  new THREE.Matrix4().makeBasis(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, -1, 0),
    new THREE.Vector3(0, 0, -1)));

function gotoCam(k) {
  const node = camPoses.get(k);
  if (!node) return;
  node.updateWorldMatrix(true, false);
  const pos = new THREE.Vector3(), quat = new THREE.Quaternion(), scl = new THREE.Vector3();
  node.matrixWorld.decompose(pos, quat, scl);
  // marker holds OpenCV cam->world (X right, Y down, Z forward); three.js
  // cameras look down -Z with +Y up, hence the basis flip.
  camera.position.copy(pos);
  camera.quaternion.copy(quat.clone().multiply(CV_TO_GL));
  // controls.update() below re-derives the orientation with lookAt against
  // camera.up, which would discard the GT camera's ROLL and leave the frame
  // tilted (and its corners clipped outside the viewport). Take up from the
  // pose instead: OpenCV +Y points DOWN, so world up is R @ (0,-1,0).
  camera.up.set(0, -1, 0).applyQuaternion(quat);
  camera.updateProjectionMatrix();
  // OrbitControls orbits about .target, so park it ahead of the camera along
  // the view axis -- otherwise the next drag snaps the view somewhere else.
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(pos).addScaledVector(fwd, sceneSpan * 0.5);
  controls.update();
}

function stepFrame(delta) {
  const k = Math.min(frameMax, Math.max(0, parseInt(frameEl.value, 10) + delta));
  frameEl.value = k;
  applyFrames();
}

// gl_PointSize is in FRAMEBUFFER pixels while sizeAttenuation is off, and the
// renderer runs at devicePixelRatio -- so on a HiDPI screen a raw slider value
// paints half (or a third) as wide as it reads. Scale by the pixel ratio so the
// number means CSS pixels. One point per depth pixel means a 518x392 cloud
// viewed head-on needs canvasWidth/518 px per point (~7 on a 3840px canvas)
// before the gaps close; below that the background shows through as a halftone.
function pointPx() {
  return parseFloat(sizeEl.value) * renderer.getPixelRatio();
}

sizeEl.addEventListener('input', () => {
  const s = pointPx();
  pointMats.forEach(m => { m.size = s; m.needsUpdate = true; });
});
document.getElementById('fit').addEventListener('click', fitView);
document.getElementById('cams').addEventListener('click', () => {
  camsVisible = !camsVisible;
  camObjs.forEach(o => { o.visible = camsVisible; });
});
document.getElementById('bg').addEventListener('click', () => {
  bgIdx = (bgIdx + 1) % bgColors.length;
  scene.background = new THREE.Color(bgColors[bgIdx]);
});

frameEl.addEventListener('input', () => { stopPlay(); applyFrames(); });
accumEl.addEventListener('change', applyFrames);
followEl.addEventListener('change', () => {
  if (followEl.checked) {
    gotoCam(parseInt(frameEl.value, 10));
  } else {
    camera.up.set(0, 1, 0);   // restore world up so free orbiting is level again
    fitView();
  }
});
document.getElementById('prev').addEventListener('click', () => { stopPlay(); stepFrame(-1); });
document.getElementById('next').addEventListener('click', () => { stopPlay(); stepFrame(1); });
const fpsEl = document.getElementById('fps');
function startPlay() {
  document.getElementById('play').textContent = 'stop';
  playTimer = setInterval(() => {
    const k = parseInt(frameEl.value, 10);
    frameEl.value = k >= frameMax ? 0 : k + 1;
    applyFrames();
  }, 1000 / parseFloat(fpsEl.value));
}
document.getElementById('play').addEventListener('click', () => {
  if (playTimer) { stopPlay(); return; }
  if (frameMax < 0) return;
  startPlay();
});
// retime a running playback when the fps slider moves
fpsEl.addEventListener('input', () => {
  if (playTimer) { clearInterval(playTimer); startPlay(); }
});
window.addEventListener('keydown', (e) => {
  if (frameMax < 0) return;
  if (e.key === 'ArrowRight') { stopPlay(); stepFrame(1); }
  else if (e.key === 'ArrowLeft') { stopPlay(); stepFrame(-1); }
});

// Any control in the bar changes something visible; delegate rather than
// threading requestRender() through every individual handler.
document.getElementById('bar').addEventListener('input', requestRender);
document.getElementById('bar').addEventListener('click', requestRender);

const fileEl = document.getElementById('file');
fileEl.addEventListener('change', () => load(fileEl.value));

fetch('api/list').then(r => r.json()).then(files => {
  if (!files.length) { infoEl.textContent = 'no .glb files in this directory'; return; }
  for (const f of files) {
    const opt = document.createElement('option');
    opt.value = f; opt.textContent = f; fileEl.appendChild(opt);
  }
  load(files[0]);
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  requestRender();
});
(function animate() {
  requestAnimationFrame(animate);
  // controls.update() returns true while damping is still settling
  const moving = controls.update();
  if (moving || needsRender) {
    renderer.render(scene, camera);
    needsRender = false;
  }
})();
</script>
</body>
</html>
""".replace("__THREE__", _THREE)


def resolve_glb_dir(path: str) -> Path:
    """Accept either the directory that holds the .glb files, or a parent (a
    run dir / its viz dir): descend through viz/ then glb/ when present, so
    `--glb-dir <run>` and `--glb-dir <run>/viz` both work."""
    p = Path(path).resolve()
    if not p.is_dir():
        raise SystemExit(f"--glb-dir is not a directory: {p}")
    for sub in ("viz", "glb"):
        # only descend if this level has no .glb of its own but the child does
        here = any(f.suffix == ".glb" for f in p.iterdir())
        child = p / sub
        if not here and child.is_dir():
            p = child
    return p


def make_handler(glb_dir: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(glb_dir), **k)

        def do_GET(self):  # noqa: N802 (stdlib naming)
            if self.path in ("/", "/index.html") or self.path.startswith("/?"):
                body = INDEX_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/list":
                files = sorted(f.name for f in glb_dir.iterdir() if f.suffix == ".glb")
                body = json.dumps(files).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()  # static .glb bytes from glb_dir

        def log_message(self, fmt, *args):  # keep the console quiet-ish
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--glb-dir",
        required=True,
        help="directory of .glb files (a run dir / viz dir also works)",
    )
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default 127.0.0.1: reachable only via the tunnel)",
    )
    args = ap.parse_args()

    glb_dir = resolve_glb_dir(args.glb_dir)
    n = len(sorted(f for f in glb_dir.iterdir() if f.suffix == ".glb"))
    hostname = socket.gethostname()

    print(f"Serving {n} .glb file(s) from {glb_dir}")
    print(f"Listening on http://{args.host}:{args.port}")
    print("-" * 64)
    print("VS Code Remote-SSH: open the Ports panel and click the forwarded")
    print(f"  localhost:{args.port} link (usually auto-detected).")
    print("Manual tunnel (run on your laptop):")
    print(f"  ssh -L {args.port}:localhost:{args.port} jdosch@ssh.ccv.brown.edu")
    print(f"  # if that lands on the wrong login node, use host {hostname}:")
    print(f"  ssh -L {args.port}:{hostname}:{args.port} jdosch@ssh.ccv.brown.edu")
    print(f"Then browse to http://localhost:{args.port}   (Ctrl-C to stop)")
    print("-" * 64)
    # serve_forever blocks; flush so the banner (esp. the tunnel command) shows
    # immediately even when stdout is redirected to a file / nohup.
    sys.stdout.flush()

    with ThreadingHTTPServer((args.host, args.port), make_handler(glb_dir)) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()

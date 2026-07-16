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
  <label>point size <input id="size" type="range" min="1" max="8" step="0.5" value="2.5"></label>
  <button id="fit">fit view</button>
  <button id="bg">bg</button>
  <span id="framectl" style="display:none; gap:6px; align-items:center;">
    <button id="prev">&#9664;</button>
    <input id="frame" type="range" min="0" max="0" step="1" value="0" style="width:160px">
    <button id="next">&#9654;</button>
    <button id="play">play</button>
    <label><input id="accum" type="checkbox" checked> accumulate</label>
    <span id="fnum" style="color:#aaa"></span>
  </span>
  <span id="info"></span>
</div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
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

const loader = new GLTFLoader();
let current = null;                 // currently displayed gltf.scene
let pointMats = [];                  // point-cloud materials (for the size slider)
let frameGroups = new Map();         // frame index -> [Object3D] (per-frame point clouds)
let frameMax = -1;                   // highest frame index, or -1 if not a per-frame scene
let playTimer = null;
const sizeEl = document.getElementById('size');
const infoEl = document.getElementById('info');
const frameCtl = document.getElementById('framectl');
const frameEl = document.getElementById('frame');
const fnumEl = document.getElementById('fnum');
const accumEl = document.getElementById('accum');

function stopPlay() {
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  document.getElementById('play').textContent = 'play';
}

function clearCurrent() {
  if (!current) return;
  scene.remove(current);
  current.traverse(o => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => m.dispose());
  });
  current = null; pointMats = [];
  frameGroups = new Map(); frameMax = -1;
}

function fitView() {
  if (!current) return;
  const box = new THREE.Box3().setFromObject(current);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const dist = 1.6 * maxDim / (2 * Math.tan((camera.fov * Math.PI/180) / 2));
  camera.near = Math.max(dist/1000, 1e-3);
  camera.far = dist * 1000;
  camera.position.copy(center).add(new THREE.Vector3(dist, dist*0.6, dist));
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function load(name) {
  stopPlay();
  infoEl.textContent = 'loading ' + name + ' ...';
  infoEl.classList.remove('err'); infoEl.id = 'info';
  loader.load(encodeURIComponent(name), (gltf) => {
    clearCurrent();
    current = gltf.scene;
    let nPts = 0;
    current.traverse(o => {
      if (o.isPoints) {
        // trimesh exports vertex colors as COLOR_0 -> GLTFLoader already sets
        // vertexColors; screen-space size (sizeAttenuation off) keeps points
        // legible regardless of the scene's metric scale.
        o.material.size = parseFloat(sizeEl.value);
        o.material.sizeAttenuation = false;
        o.material.needsUpdate = true;
        pointMats.push(o.material);
        nPts += o.geometry.attributes.position.count;
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
}

function stepFrame(delta) {
  const k = Math.min(frameMax, Math.max(0, parseInt(frameEl.value, 10) + delta));
  frameEl.value = k;
  applyFrames();
}

sizeEl.addEventListener('input', () => {
  const s = parseFloat(sizeEl.value);
  pointMats.forEach(m => { m.size = s; m.needsUpdate = true; });
});
document.getElementById('fit').addEventListener('click', fitView);
document.getElementById('bg').addEventListener('click', () => {
  bgIdx = (bgIdx + 1) % bgColors.length;
  scene.background = new THREE.Color(bgColors[bgIdx]);
});

frameEl.addEventListener('input', () => { stopPlay(); applyFrames(); });
accumEl.addEventListener('change', applyFrames);
document.getElementById('prev').addEventListener('click', () => { stopPlay(); stepFrame(-1); });
document.getElementById('next').addEventListener('click', () => { stopPlay(); stepFrame(1); });
document.getElementById('play').addEventListener('click', (e) => {
  if (playTimer) { stopPlay(); return; }
  if (frameMax < 0) return;
  e.target.textContent = 'stop';
  playTimer = setInterval(() => {
    const k = parseInt(frameEl.value, 10);
    frameEl.value = k >= frameMax ? 0 : k + 1;
    applyFrames();
  }, 600);
});
window.addEventListener('keydown', (e) => {
  if (frameMax < 0) return;
  if (e.key === 'ArrowRight') { stopPlay(); stepFrame(1); }
  else if (e.key === 'ArrowLeft') { stopPlay(); stepFrame(-1); }
});

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
});
(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
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

"""Post a mesh file to a running EllipSDF API and pull back the ellipsoids.

Start EllipSDF with the server first:
    .venv/Scripts/python main.py --server

Then, in another terminal:
    .venv/Scripts/python tools/api_post_mesh.py meshes/bunny.obj
    .venv/Scripts/python tools/api_post_mesh.py meshes/bunny.obj --num 25 --steps 1500
    .venv/Scripts/python tools/api_post_mesh.py meshes/bunny.obj --out result.json

The mesh is sent in its raw (un-normalized) coordinates — EllipSDF normalizes it
internally and returns the ellipsoids mapped back into the file's space.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh_io import as_trimesh_scene, scene_to_single_mesh  # noqa: E402


def _req(url: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())
    except urllib.error.URLError as e:
        print(f"Cannot reach server: {e}\n"
              f"Is EllipSDF running with --server?")
        raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Post a mesh to EllipSDF.")
    ap.add_argument("mesh", help="Path to a mesh file (obj/stl/ply/glb/…).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--num", type=int, default=20,
                    help="Initial ellipsoid count (num_ellipsoids).")
    ap.add_argument("--max", type=int, default=60,
                    help="Max ellipsoid count (max_ellipsoids).")
    ap.add_argument("--steps", type=int, default=2000, help="num_steps.")
    ap.add_argument("--symmetry", action="store_true")
    ap.add_argument("--out", help="Write the result JSON to this path.")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"

    # ── Load the mesh into raw vertex/face arrays ──
    mesh = scene_to_single_mesh(as_trimesh_scene(args.mesh))
    verts = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    print(f"Loaded {args.mesh}: {len(verts)} verts, {len(faces)} faces")

    # ── Check the server is up ──
    code, body = _req(f"{base}/ping")
    print("ping:", body)
    if body.get("busy"):
        print("Server is busy with another fit — try again shortly.")
        return 1

    # ── Post the fit ──
    payload = {
        "vertices": verts,
        "faces": faces,
        "options": {
            "num_ellipsoids": args.num,
            "max_ellipsoids": args.max,
            "num_steps": args.steps,
            "symmetry": args.symmetry,
        },
    }
    code, body = _req(f"{base}/fit", payload)
    if code != 202:
        print("fit rejected:", code, body)
        return 1
    job_id = body["job_id"]
    print(f"job {job_id} started — watch the fit in the EllipSDF window")

    # ── Poll until done ──
    last = -1
    while True:
        time.sleep(0.5)
        _, st = _req(f"{base}/fit/{job_id}/status")
        if st["state"] == "running" and st["step"] != last:
            last = st["step"]
            total = st["total"] or "?"
            print(f"  step {st['step']}/{total}  loss={st['loss']:.6f}  "
                  f"count={st['count']}")
        if st["state"] in ("done", "error"):
            break

    if st["state"] == "error":
        print("FIT FAILED:", st["error"])
        return 1

    # ── Fetch the result ──
    _, result = _req(f"{base}/fit/{job_id}/result")
    print(f"\nDone — {result['count']} ellipsoids")
    for e in result["ellipsoids"][:5]:
        print(f"  {e['name']}: center={e['center']} radii={e['radii']}")
    if result["count"] > 5:
        print(f"  … (+{result['count'] - 5} more)")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

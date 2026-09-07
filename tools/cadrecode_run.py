import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(points_npy, out_prefix):
    import trimesh
    import cadquery as cq
    from cadrecode_bench import execute, farthest_points, generate, load_model
    pts = np.load(points_npy).astype(np.float64)
    centre = (pts.min(0) + pts.max(0)) / 2; scale = 2.0 / max(float(np.ptp(pts, 0).max()), 1e-9)
    cloud = farthest_points((pts - centre) * scale, 256) if len(pts) > 256 else (pts - centre) * scale
    tokenizer, model = load_model()
    code = generate(tokenizer, model, cloud)
    mesh = execute(code)
    k = float(np.ptp(pts, 0).max() / max(np.ptp(mesh.vertices, axis=0).max(), 1e-9))
    mid = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
    mesh.apply_translation(-mid); mesh.apply_scale(k); mesh.apply_translation(centre)
    g = {"cq": cq}; exec(code, g)
    shape = g["r"].val().translate(cq.Vector(*(-mid))).scale(k).translate(cq.Vector(*centre))
    cq.exporters.export(cq.Workplane().add(shape), out_prefix + ".step")
    mesh.export(out_prefix + ".stl")
    open(out_prefix + ".cq.py", "w").write(code)
    json.dump({"volume_mm3": float(mesh.volume), "extents_mm": [float(x) for x in mesh.extents], "code_lines": code.count("\n") + 1}, open(out_prefix + ".json", "w"))
    print(json.dumps({"step": out_prefix + ".step", "extents_mm": [round(float(x), 2) for x in mesh.extents]}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

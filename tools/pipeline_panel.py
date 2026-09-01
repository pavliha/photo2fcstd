"""Photo, the sketch we build, the solid it makes, and the truth beside it."""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageOps

sys.path.insert(0, "src")
from photo2fcstd.bench import photos_of, truth_of
from photo2fcstd.gallery import draw_sketch

VIEW = (24, -58)


def show_mesh(ax, path, colour):
    mesh = trimesh.load(path)
    if len(mesh.faces) > 4000:
        mesh = mesh.simplify_quadric_decimation(4000)
    mesh.apply_translation(-mesh.centroid)
    scale = max(mesh.extents)
    mesh.apply_scale(1.0 / scale if scale else 1.0)
    tri = mesh.vertices[mesh.faces]
    normals = mesh.face_normals
    light = np.array([0.4, -0.7, 0.6])
    light = light / np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.clip(normals @ light, 0, 1)
    facecolors = np.array([np.array(matplotlib.colors.to_rgb(colour)) * s for s in shade])
    ax.add_collection3d(Poly3DCollection(tri, facecolors=facecolors, edgecolors="none"))
    ax.set_xlim(-0.6, 0.6), ax.set_ylim(-0.6, 0.6), ax.set_zlim(-0.6, 0.6)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(*VIEW)
    ax.set_axis_off()


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/sk5"
    parts = sys.argv[2].split(",") if len(sys.argv) > 2 else ["00245", "01619", "00806", "00205"]
    solids = sys.argv[3] if len(sys.argv) > 3 else "runs/final"
    fig = plt.figure(figsize=(15, 3.6 * len(parts)))
    fig.patch.set_facecolor("#faf8f5")
    for r, part in enumerate(parts):
        photo = sorted(photos_of(part))[:1]
        ax = fig.add_subplot(len(parts), 4, 4 * r + 1)
        if photo:
            ax.imshow(ImageOps.exif_transpose(Image.open(photo[0])))
        ax.set_title("%s  photo" % part, fontsize=9)
        ax.axis("off")
        ax = fig.add_subplot(len(parts), 4, 4 * r + 2)
        spec = os.path.join(run, "out", part + ".spec.json")
        if os.path.exists(spec):
            draw_sketch(ax, json.load(open(spec)))
        ax.set_aspect("equal"), ax.axis("off")
        ours = os.path.join(solids, "out", part + ".stl")
        ax = fig.add_subplot(len(parts), 4, 4 * r + 3, projection="3d")
        if os.path.exists(ours):
            show_mesh(ax, ours, "#2f6f9f")
        ax.set_title("built solid", fontsize=9)
        ax = fig.add_subplot(len(parts), 4, 4 * r + 4, projection="3d")
        show_mesh(ax, truth_of(part), "#8a8177")
        ax.set_title("ground truth", fontsize=9)
    fig.suptitle("photo, sketch, solid, truth", fontsize=13)
    fig.tight_layout()
    out = "tools/pipeline_panel.png"
    fig.savefig(out, dpi=100, facecolor=fig.get_facecolor())
    print(out)


if __name__ == "__main__":
    main()

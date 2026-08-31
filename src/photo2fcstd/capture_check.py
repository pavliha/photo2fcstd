"""Drive the real capture path with rendered board photos, so it can be measured without a camera.

`carve.from_photos` detects the ChArUco target, solves each pose and carves. None of that had
ever run end to end, because measuring it needs photos with the board in them and the datasets
have none. Rendering the board is exact - it is planar, so a homography maps it into any view
without approximation - which tests detection, pose and the carve together, leaving only optics
and matting untested.
"""
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from photo2fcstd import capture, carve as C, carve_check as CC, make_target

PX_PER_MM = 12


def board_texture(px_per_mm=PX_PER_MM):
    w, h = board_mm()
    art = make_target.board().generateImage((int(w * px_per_mm), int(h * px_per_mm)))
    return cv2.cvtColor(art, cv2.COLOR_GRAY2BGR) if art.ndim == 2 else art


def board_mm():
    return make_target.COLS * make_target.SQUARE_MM, make_target.ROWS * make_target.SQUARE_MM


def board_to_world():
    """The board lies pattern-up with the camera above it.

    Its printed frame runs x right and y down, which is left-handed in 2D, so laying it along
    world +x and +y and shooting from above renders it mirrored - the detector then finds three
    phantom markers out of thirty-five. Turning it over about its x axis is a real rotation
    rather than a reflection, and it leaves the part above the board where `carve` expects it.
    """
    _, h_mm = board_mm()
    rvec, _ = cv2.Rodrigues(np.diag([1.0, -1.0, -1.0]))
    return rvec, np.array([[0.0], [h_mm], [0.0]])


def world_from_board(rvec_bc, tvec_bc):
    rv, tv = board_to_world()
    R, _ = cv2.Rodrigues(rv)
    inv, _ = cv2.Rodrigues(R.T)
    r, t = cv2.composeRT(inv, -R.T @ tv, np.asarray(rvec_bc, float), np.asarray(tvec_bc, float))[:2]
    return r, t


def board_quad_world():
    w_mm, h_mm = board_mm()
    rv, tv = board_to_world()
    R, _ = cv2.Rodrigues(rv)
    corners = np.float32([[0, 0, 0], [w_mm, 0, 0], [w_mm, h_mm, 0], [0, h_mm, 0]])
    return (corners @ R.T + tv.ravel()).astype(np.float32)


def render_view(view, texture, mesh=None, shape=(CC.H, CC.W)):
    """Warp the board into the view, then paint the part's silhouette over it."""
    ph, pw = texture.shape[:2]
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    plane = board_quad_world()
    dst, _ = cv2.projectPoints(plane, view["rvec"], view["tvec"], view["K"], view["dist"])
    H = cv2.getPerspectiveTransform(src, dst.reshape(-1, 2).astype(np.float32))
    img = cv2.warpPerspective(texture, H, (shape[1], shape[0]),
                              borderValue=(255, 255, 255))
    if mesh is None:
        return img, None
    sil = CC.silhouette(mesh, view)
    img[sil] = (35, 35, 35)
    return img, sil


def board_views(n, radius, elevations=(30.0, 42.0, 55.0, 68.0)):
    """Cameras that see both the part and enough of the board to solve a pose.

    The board's printed frame runs x right and y down, which is left-handed in 2D, so with its
    corners at z=0 the camera that reads the pattern the right way round sits at negative z.
    Placing the cameras above instead renders every view mirrored, and the detector then finds
    three phantom markers out of thirty-five.
    """
    w_mm, h_mm = board_mm()
    centre = np.array([w_mm / 2, h_mm / 2, 0.0])
    out = []
    for i in range(n):
        az = 2 * np.pi * i / n
        el = np.radians(elevations[i % len(elevations)])
        eye = centre + radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        f = centre - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        d = np.cross(f, r)
        R = np.stack([r, d, f])
        rvec, _ = cv2.Rodrigues(R)
        out.append({"rvec": rvec, "tvec": (-R @ eye).reshape(3, 1),
                    "K": CC.K_of(), "dist": np.zeros(5)})
    return out


def angle_between(a, b):
    Ra, _ = cv2.Rodrigues(np.asarray(a, float))
    Rb, _ = cv2.Rodrigues(np.asarray(b, float))
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def recover(views, texture=None, mesh=None):
    """Render each view, detect the board, solve the pose, and report how far off it is."""
    texture = board_texture() if texture is None else texture
    out = []
    for v in views:
        img, sil = render_view(v, texture, mesh)
        got = capture.pose(img, v["K"], v["dist"])
        if got is not None:
            r, t = world_from_board(got["rvec"], got["tvec"])
            got = {**got, "rvec": r, "tvec": t}
        out.append({"truth": v, "got": got, "mask": sil,
                    "deg": None if got is None else angle_between(v["rvec"], got["rvec"]),
                    "mm": None if got is None else float(np.linalg.norm(
                        np.asarray(got["tvec"]).ravel() - np.asarray(v["tvec"]).ravel()))})
    return out


DETECTABLE_ELEV = (15.0, 22.0, 35.0, 55.0)


def demo():
    import trimesh
    w_mm, h_mm = board_mm()
    mesh = trimesh.creation.box((26.0, 16.0, 7.0))
    mesh.apply_translation([w_mm / 2, h_mm / 2, 3.5])
    views = board_views(16, radius=max(w_mm, h_mm) * 1.6, elevations=DETECTABLE_ELEV)
    rows = recover(views, mesh=mesh)
    found = [r for r in rows if r["got"]]
    print("rendered %d board views, solved %d" % (len(rows), len(found)))
    assert len(found) >= len(rows) - 1, "board detection failed on too many views"
    deg = np.array([r["deg"] for r in found])
    mm = np.array([r["mm"] for r in found])
    print("  rotation error  median %.4f deg, worst %.4f" % (np.median(deg), deg.max()))
    print("  position error  median %.4f mm, worst %.4f" % (np.median(mm), mm.max()))
    assert np.median(deg) < 0.5, deg
    carved = C.carve([r["got"] for r in found], [r["mask"] for r in found], voxel_mm=0.4,
                     bounds=((w_mm / 2 - 20, w_mm / 2 + 20),
                             (h_mm / 2 - 15, h_mm / 2 + 15), (0.0, 14.0)))
    got = np.sort(carved["extents_mm"])
    true = np.sort(mesh.extents)
    print("  carved extents  %s against a true %s" % (np.round(got, 2), true))
    assert np.all(got[1:] - true[1:] < 2.0), got
    floor = true[1] / 2 * np.tan(np.radians(DETECTABLE_ELEV[0]))
    print("  height is over by %.1f mm against a predicted %.1f: a silhouette taken %.0f degrees"
          % (got[0] - true[0], floor, DETECTABLE_ELEV[0]))
    print("  above the table bounds a %.0f mm wide part's height no better than that, and the"
          % true[1])
    print("  board stops being detectable below about %.0f degrees." % DETECTABLE_ELEV[0])
    assert got[0] - true[0] < 3.5, got
    print("the capture path works end to end on rendered board photos")


if __name__ == "__main__":
    demo()

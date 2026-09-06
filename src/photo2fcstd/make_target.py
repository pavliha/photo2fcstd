import cv2
import numpy as np

import os

from photo2fcstd import settings

DICT = cv2.aruco.DICT_4X4_50
COLS, ROWS = 7, 10
NOMINAL_SQUARE_MM = 15.0
SQUARE_MM = float(os.environ.get("P2F_SQUARE_MM", NOMINAL_SQUARE_MM))
MARKER_MM = SQUARE_MM * 11.0 / 15.0
DPI = 300
MM_PER_INCH = 25.4


CLEAR_X = (0.28, 0.72)
CLEAR_Y = (0.33, 0.67)


def clear_rect_mm():
    """The plain patch in the middle of the target, in board millimetres.

    A part standing on a checkerboard cannot be separated from it by appearance: a saliency
    segmenter returns the whole board, differencing against a render reaches 0.04 IoU and parallax
    between real views reaches 0.00, because every method is defeated by the pattern's edges under
    sub-pixel misalignment. Leaving the middle blank removes the problem rather than fighting it -
    the border markers still solve every pose, and the part sits on plain paper where a threshold
    reaches 0.92.
    """
    w, h = COLS * SQUARE_MM, ROWS * SQUARE_MM
    return (CLEAR_X[0] * w, CLEAR_Y[0] * h, CLEAR_X[1] * w, CLEAR_Y[1] * h)


def board():
    return cv2.aruco.CharucoBoard((COLS, ROWS), SQUARE_MM, MARKER_MM,
                                  cv2.aruco.getPredefinedDictionary(DICT))


def clear_centre(img, w_px, h_px):
    x0, y0, x1, y1 = (int(CLEAR_X[0] * w_px), int(CLEAR_Y[0] * h_px),
                      int(CLEAR_X[1] * w_px), int(CLEAR_Y[1] * h_px))
    img[y0:y1, x0:x1] = 255
    return img


def render(path):
    px = lambda mm: int(round(mm * DPI / MM_PER_INCH))
    img = board().generateImage((px(COLS * SQUARE_MM), px(ROWS * SQUARE_MM)))
    img = clear_centre(img, img.shape[1], img.shape[0])
    margin = px(10.0)
    sheet = np.full((img.shape[0] + 2 * margin, img.shape[1] + 2 * margin), 255, np.uint8)
    sheet[margin:margin + img.shape[0], margin:margin + img.shape[1]] = img
    cv2.imwrite(path, sheet)
    return sheet.shape


if __name__ == "__main__":
    shape = render(settings.charuco_target())
    print("target %dx%d px at %d dpi = %.0f x %.0f mm  (%d x %d squares of %.1f mm)"
          % (shape[1], shape[0], DPI, shape[1] * MM_PER_INCH / DPI,
             shape[0] * MM_PER_INCH / DPI, COLS, ROWS, SQUARE_MM))


A4_MM = (210.0, 297.0)
A3_MM = (297.0, 420.0)


def measured_cell_px(art):
    from photo2fcstd.capture import board_corners
    corners, ids = board_corners(cv2.cvtColor(art, cv2.COLOR_GRAY2RGB))
    if corners is None:
        return None
    xs = np.unique(np.round(corners[:, 0] / 2.0) * 2.0)
    return float(np.median(np.diff(xs))) if len(xs) > 1 else None


def board_image(cell_px, tries=6):
    want = (int(round(COLS * cell_px)), int(round(ROWS * cell_px)))
    for _ in range(tries):
        art = board().generateImage(want)
        got = measured_cell_px(art)
        if not got:
            return art
        error = got / cell_px
        if abs(error - 1.0) < 0.005:
            return art
        want = (int(round(want[0] / error)), int(round(want[1] / error)))
    return art


def pdf(path, page_mm=A4_MM, dpi=DPI):
    from PIL import Image, ImageDraw
    px = lambda mm: int(round(mm * dpi / MM_PER_INCH))
    art = Image.fromarray(board_image(px(SQUARE_MM)))
    page = Image.new("L", (px(page_mm[0]), px(page_mm[1])), 255)
    left = (page.width - art.width) // 2
    top = (page.height - art.height) // 2 + px(6.0)
    page.paste(art, (left, top))
    draw = ImageDraw.Draw(page)
    draw.text((left, top - px(11.0)), "photo2fcstd calibration target - print at 100%, no fit to page", fill=0)
    draw.text((left, top - px(5.0)), "check with a caliper: one square must be %.1f mm" % SQUARE_MM, fill=0)
    base = top + art.height + px(8.0)
    draw.line([(left, base), (left + px(50.0), base)], fill=0, width=max(1, px(0.4)))
    for i in range(6):
        x = left + px(10.0 * i)
        draw.line([(x, base - px(2.0)), (x, base + px(2.0))], fill=0, width=max(1, px(0.4)))
    draw.text((left, base + px(4.0)), "50 mm check bar", fill=0)
    page.save(path, "PDF", resolution=float(dpi))
    return path


def verify(path, dpi=DPI, page_mm=A4_MM):
    from PIL import Image
    from photo2fcstd.capture import board_corners
    page = np.array(Image.open(path).convert("L")) if not path.endswith(".pdf") else None
    if page is None:
        import subprocess, tempfile, os
        png = os.path.join(tempfile.mkdtemp(), "page.png")
        subprocess.run(["sips", "-s", "format", "png", "--resampleWidth", "2480", path, "--out", png],
                       capture_output=True)
        page = np.array(Image.open(png).convert("L"))
    corners, ids = board_corners(cv2.cvtColor(page, cv2.COLOR_GRAY2RGB))
    if corners is None:
        return None
    ppmm = page.shape[1] / page_mm[0]
    xs = np.unique(np.round(corners[:, 0] / 2.0) * 2.0)
    return float(np.median(np.diff(xs)) / ppmm)


def run():
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "charuco_target_A4.pdf"
    page_mm = A3_MM if "A3" in os.path.basename(out) else A4_MM
    pdf(out, page_mm=page_mm)
    square = verify(out, page_mm=page_mm)
    if square is None:
        print("wrote %s but could not verify it - check the print by hand" % out)
        return
    print("wrote %s: squares measure %.2f mm on the page (want %.1f)" % (out, square, SQUARE_MM))
    print("print at 100% with no scaling, then confirm one square with your caliper")


def screen(path="charuco_target_screen.png", pixels=1600):
    from PIL import Image
    art = Image.fromarray(board_image(pixels // COLS))
    art.save(path)
    return path, art.width // COLS


def screen_run():
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "charuco_target_screen.png"
    path, cell_px = screen(out)
    print("wrote %s" % path)
    print("show it full screen on a monitor, phone or tablet lying flat, then measure one square")
    print("with your caliper and pass that number: photo2fcstd ... --rectify --square-mm=<measured>")

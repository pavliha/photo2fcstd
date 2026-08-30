import cv2
import numpy as np

DICT = cv2.aruco.DICT_4X4_50
COLS, ROWS = 7, 10
SQUARE_MM = 15.0
MARKER_MM = 11.0
DPI = 300
MM_PER_INCH = 25.4


def board():
    return cv2.aruco.CharucoBoard((COLS, ROWS), SQUARE_MM, MARKER_MM,
                                  cv2.aruco.getPredefinedDictionary(DICT))


def render(path):
    px = lambda mm: int(round(mm * DPI / MM_PER_INCH))
    img = board().generateImage((px(COLS * SQUARE_MM), px(ROWS * SQUARE_MM)))
    margin = px(10.0)
    sheet = np.full((img.shape[0] + 2 * margin, img.shape[1] + 2 * margin), 255, np.uint8)
    sheet[margin:margin + img.shape[0], margin:margin + img.shape[1]] = img
    cv2.imwrite(path, sheet)
    return sheet.shape


if __name__ == "__main__":
    shape = render("/Users/pavliha/3DPrint/tools/charuco_target.png")
    print("target %dx%d px at %d dpi = %.0f x %.0f mm  (%d x %d squares of %.1f mm)"
          % (shape[1], shape[0], DPI, shape[1] * MM_PER_INCH / DPI,
             shape[0] * MM_PER_INCH / DPI, COLS, ROWS, SQUARE_MM))

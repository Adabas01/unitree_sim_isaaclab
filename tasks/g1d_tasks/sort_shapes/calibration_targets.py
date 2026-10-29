"""Textured ChArUco board and ArUco markers (USD) for comparing sim camera images with the real calibration.

Each target is a flat, double-sided quad with a UsdPreviewSurface texture, written once to a cache folder and
referenced by the scene. Geometry follows OpenCV (see layout._calibration_targets): the board's quad spans its
squares plus a white margin, the board origin is the top-left corner of the squares; a marker's origin is its
centre. Textures are generated exactly as for the printed targets (borderBits = 1, no margin inside the board).
"""

import hashlib
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

CACHE = Path(tempfile.gettempdir()) / "g1d_calibration_targets"
PX_PER_SQUARE = 120  # board texture resolution
MARKER_PX = 240
BOARD_MARGIN = 0.5  # white margin around the board, in squares
MARKER_MARGIN = 0.25  # white quiet zone around a marker, in marker sides

USDA = """#usda 1.0
(
    defaultPrim = "Target"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Target"
{{
    def Mesh "Quad" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [{points}]
        normal3f[] normals = [(0, 0, {normal}), (0, 0, {normal}), (0, 0, {normal}), (0, 0, {normal})] (
            interpolation = "vertex"
        )
        texCoord2f[] primvars:st = [{st}] (
            interpolation = "vertex"
        )
        uniform bool doubleSided = 1
        uniform token subdivisionScheme = "none"
        rel material:binding = </Target/Material>
    }}

    def Material "Material"
    {{
        token outputs:surface.connect = </Target/Material/Surface.outputs:surface>

        def Shader "Surface"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </Target/Material/Texture.outputs:rgb>
            float inputs:roughness = 0.9
            float inputs:metallic = 0
            token outputs:surface
        }}

        def Shader "Texture"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @./{texture}@
            float2 inputs:st.connect = </Target/Material/Reader.outputs:result>
            token inputs:sourceColorSpace = "sRGB"
            token inputs:wrapS = "clamp"
            token inputs:wrapT = "clamp"
            float3 outputs:rgb
        }}

        def Shader "Reader"
        {{
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }}
    }}
}}
"""


def _write(name, image, corners, st, normal):
    """Quad with the given corner points (counter-clockwise seen from the textured side) and texture coords."""
    key = hashlib.sha1(image.tobytes() + json.dumps([corners, st]).encode()).hexdigest()[:12]
    folder = CACHE / f"{name}_{key}"
    usd = folder / "target.usda"
    if not usd.exists():
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "texture.png"), image)
        usd.write_text(USDA.format(
            points=", ".join(f"({x:.6f}, {y:.6f}, 0)" for x, y in corners),
            st=", ".join(f"({s:.6f}, {t:.6f})" for s, t in st), normal=normal, texture="texture.png",
        ))
    return str(usd)


def board_usd(dictionary, squares, square_m, marker_ratio):
    cols, rows = squares
    aruco = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    board = cv2.aruco.CharucoBoard((cols, rows), square_m, square_m * marker_ratio, aruco)
    image = board.generateImage((cols * PX_PER_SQUARE, rows * PX_PER_SQUARE), marginSize=0, borderBits=1)
    pad = round(BOARD_MARGIN * PX_PER_SQUARE)
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    m = BOARD_MARGIN * square_m
    x0, x1, y0, y1 = -m, cols * square_m + m, -m, rows * square_m + m
    # Board frame: x right, y down, seen from -z. Image row 0 is at y0 (top); texture t runs upwards.
    corners = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    st = [(0, 1), (0, 0), (1, 0), (1, 1)]
    return _write("board", image, corners, st, -1)


def marker_usd(dictionary, marker_id, size_m):
    aruco = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    image = cv2.aruco.generateImageMarker(aruco, marker_id, MARKER_PX, borderBits=1)
    pad = round(MARKER_MARGIN * MARKER_PX)
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    h = size_m / 2 + MARKER_MARGIN * size_m
    # Marker frame: x right, y up, seen from +z. Image row 0 is at +h (top).
    corners = [(-h, -h), (h, -h), (h, h), (-h, h)]
    st = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return _write(f"marker{marker_id}", image, corners, st, 1)


def target_assets(targets):
    """[(prim name, parent link, usd path, position, quat_wxyz)] for the scene's calibration_targets."""
    if not targets:
        return []
    dictionary = targets["dictionary"]
    assets = []
    board = targets.get("board")
    if board:
        usd = board_usd(dictionary, board["squares"], board["square_m"], board["marker_ratio"])
        assets.append(("calibration_board", board["parent"], usd, board["position"], board["quat_wxyz"]))
    for marker in targets["markers"]:
        usd = marker_usd(dictionary, marker["id"], marker["size_m"])
        assets.append((f"calibration_marker_{marker['id']}", marker["parent"], usd, marker["position"],
                       marker["quat_wxyz"]))
    return assets


def board_texture_check(dictionary, squares, square_m, marker_ratio):
    """Detected ChArUco corner ids on the generated board texture (all inner corners when correct)."""
    cols, rows = squares
    aruco = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    board = cv2.aruco.CharucoBoard((cols, rows), square_m, square_m * marker_ratio, aruco)
    image = cv2.imread(str(Path(board_usd(dictionary, squares, square_m, marker_ratio)).with_name("texture.png")),
                       cv2.IMREAD_GRAYSCALE)
    _, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(image)
    return np.sort(ids.reshape(-1)) if ids is not None else np.zeros(0, int)

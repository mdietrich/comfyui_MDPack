"""
Image Hand/Foot Montage — 2-Pass-Zoom für die VLM-Prüfung.

Findet Hände (MediaPipe Hands) und Füße (MediaPipe Pose: Knöchel/Ferse/Fußspitze),
schneidet die Regionen mit Kontext aus, vergrößert sie und setzt sie zusammen mit
dem verkleinerten Originalbild zu EINEM beschrifteten Montage-Bild zusammen.

Zweck: Dieses Montage-Bild an das VLM geben. Fingerzahl/Zehen sind auf den
vergrößerten Crops viel besser beurteilbar als auf dem Gesamtbild — das ist der
wirksamste Hebel gegen die schwache Finger-Erkennung.

Hinweis: MediaPipe Pose ist single-person (ein 33-Punkt-Skelett). Bei mehreren
Personen werden Füße nur für die dominant erkannte Person gefunden; Hände (bis
max_num_hands) auch mehrerer Personen.
"""

import json
import os

# Headless-GL-Fix (siehe nodes.py) — muss vor mediapipe/cv2 stehen.
os.environ.pop("DISPLAY", None)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GLOG_minloglevel", "2")

import numpy as np
import torch

_MP = {}

# Pose-Landmark-Indizes für Füße
_FOOT_IDX = {
    "Left": [27, 29, 31],   # left_ankle, left_heel, left_foot_index
    "Right": [28, 30, 32],  # right_ankle, right_heel, right_foot_index
}


def _first_frame_uint8(image):
    arr = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _load_hands():
    if "hands" not in _MP:
        import mediapipe as mp

        _MP["hands"] = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=6, min_detection_confidence=0.4
        )
    return _MP["hands"]


def _load_pose():
    if "pose" not in _MP:
        import mediapipe as mp

        _MP["pose"] = mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=1, min_detection_confidence=0.4
        )
    return _MP["pose"]


def _square_bbox(pts, w, h, pad_ratio):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    side = max(max(xs) - min(xs), max(ys) - min(ys), 8.0) * (1.0 + 2.0 * pad_ratio)
    half = side / 2.0
    x0 = int(max(0, cx - half))
    y0 = int(max(0, cy - half))
    x1 = int(min(w, cx + half))
    y1 = int(min(h, cy + half))
    return x0, y0, x1, y1


def _find_hands(rgb, pad_ratio):
    h, w = rgb.shape[:2]
    res = _load_hands().process(rgb)
    out = []
    if not res.multi_hand_landmarks:
        return out
    labels = None
    if getattr(res, "multi_handedness", None):
        labels = [d.classification[0].label for d in res.multi_handedness]
    for i, hand in enumerate(res.multi_hand_landmarks):
        pts = [(lm.x * w, lm.y * h) for lm in hand.landmark]
        side = labels[i] if labels and i < len(labels) else "?"
        out.append((f"Hand {i + 1} ({side})", _square_bbox(pts, w, h, pad_ratio)))
    return out


def _find_feet(rgb, pad_ratio, min_vis):
    h, w = rgb.shape[:2]
    res = _load_pose().process(rgb)
    out = []
    if not res.pose_landmarks:
        return out
    lms = res.pose_landmarks.landmark
    for side, idxs in _FOOT_IDX.items():
        pts, vis = [], []
        for i in idxs:
            lm = lms[i]
            vis.append(lm.visibility)
            pts.append((lm.x * w, lm.y * h))
        if max(vis) >= min_vis:
            out.append((f"Foot ({side})", _square_bbox(pts, w, h, pad_ratio)))
    return out


def _crop_resize(rgb, box, size):
    from PIL import Image

    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.zeros((size, size, 3), dtype=np.uint8)
    return Image.fromarray(crop).convert("RGB").resize((size, size), Image.LANCZOS)


def _font(px):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=px)
    except Exception:
        return ImageFont.load_default()


def _tile(crop_img, size, label):
    from PIL import Image, ImageDraw

    bar = max(18, size // 12)
    canvas = Image.new("RGB", (size, size + bar), (25, 25, 25))
    canvas.paste(crop_img, (0, bar))
    ImageDraw.Draw(canvas).text((5, 2), label, fill=(240, 240, 240), font=_font(bar - 6))
    return canvas


def _compose(original_uint8, tiles, orig_max_w, size):
    from PIL import Image, ImageDraw

    orig = Image.fromarray(original_uint8).convert("RGB")
    ow, oh = orig.size
    scale = min(1.0, orig_max_w / ow)
    dw, dh = max(1, int(ow * scale)), max(1, int(oh * scale))
    orig_disp = orig.resize((dw, dh), Image.LANCZOS)

    if not tiles:
        return np.asarray(orig_disp)

    bar = max(18, size // 12)
    tile_h = size + bar
    gap = 14
    header = 26
    cols = max(1, min(len(tiles), max(1, (dw + gap) // (size + gap))))
    rows = (len(tiles) + cols - 1) // cols
    grid_w = cols * size + (cols - 1) * gap
    grid_h = rows * tile_h + (rows - 1) * gap

    cw = max(dw, grid_w)
    ch = dh + gap + header + grid_h
    canvas = Image.new("RGB", (cw, ch), (35, 35, 35))
    canvas.paste(orig_disp, ((cw - dw) // 2, 0))
    ImageDraw.Draw(canvas).text(
        (6, dh + gap + 4), "Enlarged hands & feet (judge fingers/toes here):",
        fill=(255, 220, 120), font=_font(16),
    )

    y = dh + gap + header
    x = (cw - grid_w) // 2
    # tiles ist eine Liste aus (label, PIL-Tile)
    for i, (_, tile_img) in enumerate(tiles):
        r, c = divmod(i, cols)
        px = x + c * (size + gap)
        py = y + r * (tile_h + gap)
        canvas.paste(tile_img, (px, py))
    return np.asarray(canvas)


class ImageHandFootMontage:
    """
    Findet Hände & Füße, vergrößert sie und baut ein Montage-Bild
    (Original + vergrößerte, beschriftete Crops) für die VLM-Beurteilung.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "include_hands": ("BOOLEAN", {"default": True}),
                "include_feet": ("BOOLEAN", {"default": True}),
                "tile_size": ("INT", {"default": 448, "min": 128, "max": 1024, "step": 32}),
                "padding": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "original_max_width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
                "min_foot_visibility": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("montage", "found", "num_crops")
    FUNCTION = "build"
    CATEGORY = "MDPack/quality"

    def build(self, image, include_hands, include_feet, tile_size, padding,
              original_max_width, min_foot_visibility):
        rgb = _first_frame_uint8(image)

        regions = []
        if include_hands:
            regions += _find_hands(rgb, padding)
        if include_feet:
            regions += _find_feet(rgb, padding, min_foot_visibility)

        tiles = [(label, _tile(_crop_resize(rgb, box, tile_size), tile_size, label))
                 for label, box in regions]

        montage = _compose(rgb, tiles, original_max_width, tile_size)
        tensor = torch.from_numpy(montage.astype(np.float32) / 255.0).unsqueeze(0)

        found = json.dumps({
            "num_crops": len(tiles),
            "hands": sum(1 for l, _ in regions if l.startswith("Hand")),
            "feet": sum(1 for l, _ in regions if l.startswith("Foot")),
            "labels": [l for l, _ in regions],
        }, ensure_ascii=False)
        return (tensor, found, len(tiles))


NODE_CLASS_MAPPINGS = {"ImageHandFootMontage": ImageHandFootMontage}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageHandFootMontage": "Image Hand/Foot Montage (2-Pass)"}

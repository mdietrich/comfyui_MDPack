"""
Image Quality Gate — Stufe 1 (lokale CV-Filter) für die automatische
Qualitätsprüfung KI-generierter Bilder.

Erkennt billig & schnell die OFFENSICHTLICHEN Ausschüsse:
  - niedriger Aesthetic-Score  (Artefakte / "billig wirkend")
  - verschwommene Augen        (Laplacian-Varianz auf der Augenregion)
  - globale Unschärfe          (Laplacian-Varianz Gesamtbild)
  - falsche Anzahl Gesichter   (MediaPipe FaceDetection)
  - falsche Anzahl Hände       (MediaPipe Hands)
  - falsche Anzahl Personen    (optional: Ultralytics YOLO-Pose)

WICHTIG — Grenzen von Stufe 1:
  Subtile Anatomiefehler wie "6 Finger", "verschmolzene Finger", "doppelter Arm
  am selben Körper" oder "Textfehler" werden hier NICHT zuverlässig erkannt.
  MediaPipe liefert für eine deformierte Hand trotzdem 21 Landmarks und für eine
  Person ein festes 33-Punkt-Skelett — es *zählt* Instanzen, es *bewertet* keine
  Anatomie. Diese Fälle gehen an Stufe 2 (VLM-Richter, siehe Jira-Ticket).

Stufe 1 ist also der billige Grobfilter, der ~70-80 % der offensichtlichen
Ausschüsse abfängt, bevor das teurere VLM überhaupt anläuft.

Alle Checks sind einzeln abschaltbar und degradieren sauber: fehlt eine Library
oder ein Gewicht, wird der Check mit einem Hinweis übersprungen statt zu crashen.
"""

import json
import os

# --------------------------------------------------------------------------- #
#  Headless-GL-Fix (RunPod / Docker / Server ohne echtes Display)
#  MediaPipe versucht sonst einen GLX-Kontext über X11 zu erzeugen und crasht mit
#  "X Error ... BadAccess ... GLX X_GLXMakeCurrent". Ist DISPLAY nicht gesetzt,
#  nutzt MediaPipe den Headless-EGL-/CPU-Pfad (das EGL-Init klappt bereits laut Log).
#  MUSS vor dem ersten mediapipe-/cv2-Import passieren -> daher ganz oben.
# --------------------------------------------------------------------------- #
os.environ.pop("DISPLAY", None)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GLOG_minloglevel", "2")  # MediaPipe-Log-Spam dämpfen

import numpy as np
import torch

# Modelle werden lazy geladen und prozessweit gecached.
_MODELS = {}
_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
_AES_WEIGHT = os.path.join(_WEIGHTS_DIR, "sac+logos+ava1-l14-linearMSE.pth")


# --------------------------------------------------------------------------- #
#  Hilfsfunktionen
# --------------------------------------------------------------------------- #
def _first_frame_uint8(image):
    """ComfyUI IMAGE-Tensor [B,H,W,C] float 0..1  ->  HxWxC uint8 RGB (erstes Bild)."""
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _laplacian_var(gray):
    """Schärfe-Proxy: Varianz der Laplacian. Niedrig = unscharf."""
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# --------------------------------------------------------------------------- #
#  Aesthetic (CLIP ViT-L/14 + LAION-Linear-Head)
# --------------------------------------------------------------------------- #
class _AestheticMLP(torch.nn.Module):
    """MLP von improved-aesthetic-predictor. Trotz Dateiname 'linearMSE' ist das
    Gewicht ein mehrschichtiges MLP mit State-Dict-Keys layers.0/2/4/6/7."""

    def __init__(self, input_size=768):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024),  # layers.0
            torch.nn.Dropout(0.2),              # layers.1
            torch.nn.Linear(1024, 128),         # layers.2
            torch.nn.Dropout(0.2),              # layers.3
            torch.nn.Linear(128, 64),           # layers.4
            torch.nn.Dropout(0.1),              # layers.5
            torch.nn.Linear(64, 16),            # layers.6
            torch.nn.Linear(16, 1),             # layers.7
        )

    def forward(self, x):
        return self.layers(x)


def _load_aesthetic():
    if "aes" in _MODELS:
        return _MODELS["aes"]
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model = model.to(device).eval()

    head = _AestheticMLP(768)
    state = torch.load(_AES_WEIGHT, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    head.load_state_dict(state)
    head = head.to(device).eval()

    _MODELS["aes"] = (model, preprocess, head, device)
    return _MODELS["aes"]


def _aesthetic_score(pil_img):
    model, preprocess, head, device = _load_aesthetic()
    with torch.no_grad():
        x = preprocess(pil_img).unsqueeze(0).to(device)
        feat = model.encode_image(x).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return float(head(feat).item())


# --------------------------------------------------------------------------- #
#  MediaPipe-Singletons
# --------------------------------------------------------------------------- #
def _load_facemesh():
    if "fm" not in _MODELS:
        import mediapipe as mp

        _MODELS["fm"] = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=5, refine_landmarks=True
        )
    return _MODELS["fm"]


def _load_face_detection():
    if "fd" not in _MODELS:
        import mediapipe as mp

        _MODELS["fd"] = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
    return _MODELS["fd"]


def _load_hands():
    if "hands" not in _MODELS:
        import mediapipe as mp

        _MODELS["hands"] = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=6, min_detection_confidence=0.5
        )
    return _MODELS["hands"]


# FaceMesh-Indizes (refine_landmarks=True) für linkes/rechtes Auge
_LEFT_EYE = [33, 133, 159, 145, 158, 153]
_RIGHT_EYE = [362, 263, 386, 374, 385, 380]


def _eye_sharpness(rgb):
    """Minimale Laplacian-Varianz über beide Augenregionen. None wenn kein Auge."""
    import cv2

    h, w = rgb.shape[:2]
    fm = _load_facemesh()
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None, 0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharpness = []
    faces = res.multi_face_landmarks
    for face in faces:
        for eye_idx in (_LEFT_EYE, _RIGHT_EYE):
            xs = [face.landmark[i].x * w for i in eye_idx]
            ys = [face.landmark[i].y * h for i in eye_idx]
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            r = max(6, int((max(xs) - min(xs)) * 0.9))
            x0, x1 = max(0, cx - r), min(w, cx + r)
            y0, y1 = max(0, cy - r), min(h, cy + r)
            crop = gray[y0:y1, x0:x1]
            if crop.size >= 16:
                sharpness.append(_laplacian_var(crop))
    if not sharpness:
        return None, len(faces)
    return min(sharpness), len(faces)


def _count_faces(rgb):
    res = _load_face_detection().process(rgb)
    return len(res.detections) if res.detections else 0


def _count_hands(rgb):
    res = _load_hands().process(rgb)
    return len(res.multi_hand_landmarks) if res.multi_hand_landmarks else 0


def _count_persons(rgb):
    """Optional via Ultralytics YOLO-Pose. None wenn nicht installiert."""
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    if "yolo" not in _MODELS:
        _MODELS["yolo"] = YOLO("yolov8n-pose.pt")
    res = _MODELS["yolo"](rgb, verbose=False)
    if not res:
        return 0
    boxes = getattr(res[0], "boxes", None)
    return 0 if boxes is None else int(len(boxes))


# --------------------------------------------------------------------------- #
#  ComfyUI-Node
# --------------------------------------------------------------------------- #
class ImageQualityGate:
    """
    Stufe-1-Grobfilter. Nimmt ein IMAGE, gibt es unverändert zurück plus:
      passed  (BOOLEAN)  -> True wenn ALLE aktiven Checks bestanden
      score   (FLOAT)    -> Aesthetic-Score (0..~10), -1 wenn nicht berechnet
      report  (STRING)   -> JSON mit allen Einzelergebnissen (für Java/Logging)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "check_aesthetic": ("BOOLEAN", {"default": True}),
                "aesthetic_min": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "check_eyes": ("BOOLEAN", {"default": True}),
                "eye_blur_min": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 2000.0, "step": 1.0}),
                "check_global_blur": ("BOOLEAN", {"default": True}),
                "global_blur_min": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 2000.0, "step": 1.0}),
                "check_faces": ("BOOLEAN", {"default": True}),
                "expected_faces": ("INT", {"default": 1, "min": 0, "max": 10}),
                "check_hands": ("BOOLEAN", {"default": True}),
                "max_hands": ("INT", {"default": 2, "min": 0, "max": 10}),
                "check_persons": ("BOOLEAN", {"default": False}),
                "expected_persons": ("INT", {"default": 1, "min": 0, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "passed", "score", "report")
    FUNCTION = "assess"
    CATEGORY = "MDPack/quality"
    # OUTPUT_NODE, damit der JSON-Report in der ComfyUI-History (API /history)
    # auftaucht und vom Java-Backend ausgelesen werden kann.
    OUTPUT_NODE = True

    def assess(
        self,
        image,
        check_aesthetic,
        aesthetic_min,
        check_eyes,
        eye_blur_min,
        check_global_blur,
        global_blur_min,
        check_faces,
        expected_faces,
        check_hands,
        max_hands,
        check_persons,
        expected_persons,
    ):
        rgb = _first_frame_uint8(image)
        checks = {}
        reasons = []
        aesthetic = -1.0

        # --- Aesthetic ---
        if check_aesthetic:
            try:
                from PIL import Image

                aesthetic = _aesthetic_score(Image.fromarray(rgb))
                ok = aesthetic >= aesthetic_min
                checks["aesthetic"] = {"ok": ok, "value": round(aesthetic, 3), "min": aesthetic_min}
                if not ok:
                    reasons.append(f"aesthetic {aesthetic:.2f} < {aesthetic_min}")
            except Exception as e:  # noqa: BLE001
                checks["aesthetic"] = {"skipped": str(e)}

        # --- Augen-Schärfe + Gesichts-Anzahl (FaceMesh liefert beides) ---
        if check_eyes:
            try:
                eye_sharp, n_faces_mesh = _eye_sharpness(rgb)
                if eye_sharp is None:
                    checks["eyes"] = {"skipped": "no face/eye detected"}
                else:
                    ok = eye_sharp >= eye_blur_min
                    checks["eyes"] = {"ok": ok, "value": round(eye_sharp, 1), "min": eye_blur_min}
                    if not ok:
                        reasons.append(f"eye blur {eye_sharp:.0f} < {eye_blur_min}")
            except Exception as e:  # noqa: BLE001
                checks["eyes"] = {"skipped": str(e)}

        # --- Globale Schärfe ---
        if check_global_blur:
            try:
                import cv2

                gv = _laplacian_var(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
                ok = gv >= global_blur_min
                checks["global_blur"] = {"ok": ok, "value": round(gv, 1), "min": global_blur_min}
                if not ok:
                    reasons.append(f"global blur {gv:.0f} < {global_blur_min}")
            except Exception as e:  # noqa: BLE001
                checks["global_blur"] = {"skipped": str(e)}

        # --- Gesichts-Anzahl ---
        if check_faces:
            try:
                n = _count_faces(rgb)
                ok = n == expected_faces
                checks["faces"] = {"ok": ok, "count": n, "expected": expected_faces}
                if not ok:
                    reasons.append(f"faces {n} != {expected_faces}")
            except Exception as e:  # noqa: BLE001
                checks["faces"] = {"skipped": str(e)}

        # --- Hand-Anzahl ---
        if check_hands:
            try:
                n = _count_hands(rgb)
                ok = n <= max_hands
                checks["hands"] = {"ok": ok, "count": n, "max": max_hands}
                if not ok:
                    reasons.append(f"hands {n} > {max_hands}")
            except Exception as e:  # noqa: BLE001
                checks["hands"] = {"skipped": str(e)}

        # --- Personen-Anzahl (optional) ---
        if check_persons:
            n = _count_persons(rgb)
            if n is None:
                checks["persons"] = {"skipped": "ultralytics not installed"}
            else:
                ok = n == expected_persons
                checks["persons"] = {"ok": ok, "count": n, "expected": expected_persons}
                if not ok:
                    reasons.append(f"persons {n} != {expected_persons}")

        failed = [k for k, v in checks.items() if v.get("ok") is False]
        passed = len(failed) == 0

        report = json.dumps(
            {
                "passed": passed,
                "aesthetic": round(aesthetic, 3),
                "failed_checks": failed,
                "reasons": reasons,
                "checks": checks,
                "note": "Stufe 1 (Grobfilter). Feine Anatomie/Finger/Text -> Stufe 2 (VLM).",
            },
            ensure_ascii=False,
        )
        return {
            "ui": {"text": [report], "passed": [passed]},
            "result": (image, passed, float(aesthetic), report),
        }


NODE_CLASS_MAPPINGS = {"ImageQualityGate": ImageQualityGate}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageQualityGate": "Image Quality Gate (Stufe 1)"}

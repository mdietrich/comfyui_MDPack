"""End-to-end check of the partial re-run path, with the model stubbed out."""
import json, os, re, sys

COMFY = "/home/mdietrich/ComfyUI_winows_portable/ComfyUI"
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes", "comfyui_MDPack"))
os.chdir(COMFY)

import folder_paths  # noqa: F401
import video_to_h3_prompts as V

CALLS = []


def fake_ask(self, video_path, question, sample_fps, max_frames, prefill,
             max_new_tokens, temperature, seed):
    """Return canned JSON and record what context the clip was asked with."""
    index = len(CALLS) + 1
    CALLS.append({"question": question, "segment": os.path.basename(video_path)})
    if "STYLE BLOCK" not in question:          # the style pass
        return "A synthetic test pattern fills the frame."
    tag = re.search(r"clip (\d+) of", question)
    n = int(tag.group(1)) if tag else index
    body = {"prompt": "Run%s clip %d action." % (RUN, n),
            "closing_state": "Run%s clip %d closing." % (RUN, n),
            "used": ["element-%d" % n]}
    return json.dumps(body)[len(V.PREFILL_CLIP):]


V.VideoToH3ClipPrompts.ask = fake_ask
V.VideoToH3ClipPrompts.load_model = lambda self, *a, **k: None
V.VideoToH3ClipPrompts.unload = lambda self: None

common = dict(video="synthetic_45s.mp4", chunk_seconds=15.0, fps=24,
              model="x", quantization="none", attention="sdpa",
              keep_model_loaded=False, sample_fps=2.0, frame_height=256,
              style_words=40, clip_words=40, max_new_tokens=200,
              temperature=0.6, seed=1, whisper_model="off",
              transcribe_language="auto",
              camera_discipline=V.DEFAULT_CAMERA_DISCIPLINE,
              style_instruction=V.DEFAULT_STYLE_INSTRUCTION,
              clip_instruction=V.DEFAULT_CLIP_INSTRUCTION,
              retry_overlap=1.01, chain_name="pytest_resume")

node = V.VideoToH3ClipPrompts()

RUN = "A"
style_a, clips_a, count, _, stats_a = node.describe(
    reset_state=True, start_clip=1, end_clip=0, **common)
print("LAUF A (alle %d Clips):" % count)
print(clips_a)

CALLS.clear()
RUN = "B"
style_b, clips_b, _, _, stats_b = node.describe(
    reset_state=False, start_clip=2, end_clip=2, **common)
print()
print("LAUF B (nur Clip 2 neu):")
print(clips_b)

print()
print("--- PRUEFUNGEN ---")
got = {int(m.group(1)): m.group(2).strip() for m in re.finditer(
    r'(?ms)^\s*\[(\d+)\]\s*(.*?)(?=^\s*\[\d+\]\s*|\Z)', clips_b)}
checks = [
    ("Clip 1 unveraendert aus dem Zustand", got.get(1) == "RunA clip 1 action."),
    ("Clip 3 unveraendert aus dem Zustand", got.get(3) == "RunA clip 3 action."),
    ("Clip 2 wurde neu geschrieben", got.get(2) == "RunB clip 2 action."),
    ("nur EIN Modellaufruf in Lauf B", len(CALLS) == 1),
    ("Style-Block wiederverwendet", style_b == style_a),
    ("Airlock kennt Clip-1-Schluss",
     "RunA clip 1 closing." in CALLS[0]["question"]),
    ("Sperrliste kennt element-1 und element-3",
     "element-1" in CALLS[0]["question"] and "element-3" in CALLS[0]["question"]),
]
for name, ok in checks:
    print("%-42s %s" % (name, "OK" if ok else "FEHLGESCHLAGEN"))
print()
print("ERGEBNIS:", "alle Pruefungen bestanden" if all(c[1] for c in checks)
      else "FEHLER")

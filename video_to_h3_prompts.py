"""Turn an existing video into MiniMax H3 Auto Chain prompts with a local VLM.

The node walks a source video segment by segment, describes each segment with a
locally loaded Qwen3-VL model and emits the two strings the
``MiniMaxH3AutoChainAudio`` node expects: a constant ``style_prompt`` and a
``clip_prompts`` block tagged ``[1] ... [2] ...``.

Everything runs on the local GPU; no API key and no cloud call is involved.
"""

import json
import math
import os
import re
import subprocess
import sys
import time

try:
    import folder_paths
except ImportError:  # running the module directly (tests)
    folder_paths = None

try:
    import comfy.model_management as model_management
except ImportError:  # running the module directly (tests)
    model_management = None


# Repository owner per model: the abliterated builds live under huihui-ai,
# the stock instruct builds under qwen.
MODEL_REPOS = {
    "Huihui-Qwen3-VL-8B-Instruct-abliterated": "huihui-ai",
    "Huihui-Qwen3-VL-32B-Instruct-abliterated": "huihui-ai",
}

MODEL_CHOICES = [
    "Huihui-Qwen3-VL-8B-Instruct-abliterated",
    "Huihui-Qwen3-VL-32B-Instruct-abliterated",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-8B-Instruct-FP8",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-4B-Instruct-FP8",
    "Qwen3-VL-8B-Thinking",
    "Qwen3-VL-4B-Thinking",
]

WHISPER_CHOICES = ["off", "tiny", "base", "small", "medium", "large-v3"]

PREFILL_STYLE = ""
PREFILL_CLIP = '{"prompt": "'

DEFAULT_CAMERA_DISCIPLINE = (
    "The camera holds a steady medium presenter distance for the whole chain. "
    "The entire head, hair, shoulders and upper body stay comfortably visible "
    "with generous space around the head, and the framing stays as wide as it "
    "was established."
)

DEFAULT_STYLE_INSTRUCTION = """\
You are writing the STYLE BLOCK for a MiniMax H3 video chain. This block is
prepended verbatim to every single clip prompt of the chain, so it may only
contain things that stay constant for the whole video.

Watch the attached video and write ONE paragraph of {style_words} words covering,
in this order:
1. The main subject: apparent age range, build, hair (colour, length, style),
   eyebrows, eye colour, skin tone, make-up, wardrobe with colours and materials.
2. The location and background with its distinctive objects.
3. The lighting: direction, softness, colour temperature.

Write plain declarative present-tense prose. Describe only what is visible.
No camera movement, no actions, no dialogue, no story, no quality words such as
"cinematic", "4k", "masterpiece" or "beautiful".

Output the paragraph and nothing else."""

DEFAULT_CLIP_INSTRUCTION = """\
Write the prompt for this clip of a MiniMax H3 chain.

- One paragraph of {clip_words} words, present tense, English.
- {opening}
- Describe what physically CHANGES during this segment: a gesture, a weight
  shift, an object that ends up somewhere else. Make it specific enough that
  this clip could not be confused with any other clip of the chain.
- Describe the camera behaviour you actually observe in the segment, in plain
  terms (static, slow push-in, lateral arc, subtle reframe).
- Dialogue comes only from the transcript. Quote a transcribed line in double
  quotes, word for word. Never invent speech, and never add a murmur, a hum, a
  sigh, a gasp or any other vocal sound that is not in the transcript - the
  model renders every one of them as audible voice.
- End with the scene settled and the line finished, holding a stable
  arrangement for about two seconds.
- Never write a sentence about something being absent, still or unchanged.
  A clause like "she does not speak", "nothing moves" or "the room is
  unchanged" conditions the model on exactly the word you tried to exclude.
- Write what happens, never what does not happen: this model renders negations,
  so "the camera holds the established distance" works and "do not zoom in"
  does not. Phrases about stillness freeze the whole frame, so keep the camera
  steady in words but keep the performer breathing and moving.
- Write complete sentences, each with a subject and a verb. Never string
  observations together as comma-separated fragments.
- The subject's appearance, the room and the lighting are already in the style
  block. Do not describe them again; write only what happens.
- No headings, no clip numbers, no quality words, no backstory."""


# --------------------------------------------------------------------- helpers

def probe_duration(path):
    """Duration of a media file in seconds, via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def plan_clips(duration, chunk_seconds, fps):
    """Split a duration into the same clip grid the Auto Chain node uses.

    The chain builds a whole-frame timeline and cuts it into chunks of
    ``chunk_seconds``; the final chunk may be shorter. Returns a list of
    ``(index, start, end)`` tuples with 1-based indices.
    """
    fps = max(1.0, float(fps))
    total_frames = max(1, int(round(float(duration) * fps)))
    per_clip = max(1, int(round(float(chunk_seconds) * fps)))
    count = max(1, math.ceil(total_frames / per_clip))
    clips = []
    for index in range(count):
        start_frame = index * per_clip
        end_frame = min(total_frames, start_frame + per_clip)
        clips.append((index + 1, start_frame / fps, end_frame / fps))
    return clips


def stub_clip_warning(clips, chunk_seconds, minimum=2.0):
    """Warn when the clip grid ends in a stub too short to render or describe.

    The grid has to stay identical to the Auto Chain node, so the stub is not
    merged away here; the message names the chunk_seconds that divides the
    source evenly instead.
    """
    if len(clips) < 2:
        return ""
    index, start, end = clips[-1]
    length = end - start
    if length >= float(minimum):
        return ""
    total = clips[-1][2]
    even = total / max(1, len(clips) - 1)
    return ("WARNING: clip %d is only %.2fs long. Set chunk_seconds to %.2f "
            "on this node and on the Auto Chain to get %d even clips instead."
            % (index, length, even, len(clips) - 1))


def transcript_window(segments, start, end):
    """Transcript text of every segment that overlaps the [start, end) window."""
    parts = []
    for seg in segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        if seg_end > float(start) and seg_start < float(end):
            text = str(seg["text"]).strip()
            if text:
                parts.append(text)
    return " ".join(parts)


def strip_meta(text):
    """Remove headings, clip tags, markdown fences and stray quotes."""
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    text = text.strip()
    text = re.sub(r"^\s*(?:\[\d+\]|Clip\s*\d+\s*[:.-]|Shot\s*\d+\s*[:.-]|"
                  r"Prompt\s*[:.-]|\*\*[^*]{0,40}\*\*\s*[:.-]?)\s*", "",
                  text, flags=re.IGNORECASE)
    text = text.strip().strip('"').strip()
    return re.sub(r"\s*\n\s*", " ", text).strip()


def parse_clip_answer(raw, prefill=PREFILL_CLIP):
    """Return ``(prompt, closing_state, used)`` from a model answer.

    The assistant turn is prefilled with the beginning of the JSON object, so
    the answer is parsed as JSON first. A truncated or malformed object falls
    back to treating the whole text as the prompt. ``used`` is the model's own
    list of body parts and objects it consumed, which the following clips get
    handed back as a ban list.
    """
    text = re.sub(r"```(?:json)?", "", str(prefill) + str(raw))
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    brace = text.find("{")
    if brace >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, brace)
            if isinstance(obj, dict):
                prompt = strip_meta(str(obj.get("prompt", "")))
                closing = strip_meta(str(obj.get("closing_state", "")))
                used = [str(u).strip().lower() for u in obj.get("used", [])
                        if str(u).strip()]
                if prompt:
                    return prompt, closing, used
        except json.JSONDecodeError:
            pass
        # Truncated object: salvage the prompt string literal, drop the rest.
        match = re.search(r'"prompt"\s*:\s*"(.*?)(?<!\\)"', text, flags=re.DOTALL)
        if match:
            try:
                prompt = json.loads('"%s"' % match.group(1))
            except json.JSONDecodeError:
                prompt = match.group(1)
            closing = ""
            tail = re.search(r'"closing_state"\s*:\s*"(.*?)(?<!\\)"', text,
                             flags=re.DOTALL)
            if tail:
                closing = strip_meta(tail.group(1))
            return strip_meta(prompt), closing, []
    return strip_meta(text), "", []


def closing_from_prompt(text):
    """Last sentence of a clip prompt, used when the model omits closing_state.

    Carrying the previous clip's state forward instead would make two clips
    open on the same arrangement, which reads as a stalled chain.
    """
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text))
                 if part.strip()]
    return sentences[-1] if sentences else ""


BANNED_PHRASES = ("shifts her weight", "settles back", "holds the pose",
                  "breathing steadily", "the camera remains static",
                  "for two seconds")


def recent_actions(prompts, used, count=2):
    """Directive that keeps the writer off ground the chain already covered.

    Two mechanisms: the last few prompts verbatim so the sentence rhythm
    varies, and the cumulative list of body parts and objects the model itself
    reported as consumed. The cumulative list is what stops clip 11 from
    repeating clip 9 - a window of recent text alone cannot reach that far
    back without spending the whole context on it.
    """
    parts = []
    recent = [p for p in prompts if p.strip()][-int(count):]
    if recent:
        parts.append("THE CLIPS IMMEDIATELY BEFORE THIS ONE:\n%s"
                     % "\n".join("- %s" % p.strip() for p in recent))
    consumed = sorted({str(u).strip().lower() for u in used if str(u).strip()})
    if consumed:
        parts.append(
            "ALREADY USED EARLIER IN THIS CHAIN - do not build this clip's "
            "action around any of them:\n%s" % ", ".join(consumed))
    if not parts:
        return ""
    parts.append(
        "Pick a body part, an object or a direction that is not listed above. "
        "These phrases are used up and are banned: %s."
        % ", ".join(BANNED_PHRASES))
    return "\n\n".join(parts)


ABSENCE_PATTERNS = (
    r"\b(?:she|he|they|the (?:woman|man|subject)|nobody|no one)\s+"
    r"(?:does|do)\s+not\s+(?:speak|talk|say)",
    r"\b(?:she|he|they|the (?:woman|man|subject))\s+(?:remains?|stays?|is)\s+"
    r"(?:silent|quiet|motionless|still)",
    r"\b(?:there\s+is|there's)\s+no\s+(?:sound|speech|dialogue|audio|noise)",
    r"\bno\s+(?:one|body)\s+speaks",
    r"\bnothing\s+(?:moves|happens|changes)",
    r"\bthe\s+(?:clip|shot|scene|segment)\s+is\s+silent",
    r"\bwithout\s+(?:speaking|speech|sound|dialogue)",
)


def fill_placeholders(template, **values):
    """Substitute the named placeholders and leave every other brace alone.

    str.format would raise KeyError or ValueError on any stray brace a user
    types into the editable instruction widgets, which surfaces as a failed
    run rather than as the typo it is.
    """
    result = str(template)
    for key, value in values.items():
        result = result.replace("{%s}" % key, str(value))
    return result


def strip_absence_sentences(text):
    """Drop sentences that state what is absent.

    H3 runs at CFG 1.0 with no negative branch, so "she does not speak" puts
    *speak* into the conditioning. The writer keeps producing these even when
    the instruction forbids them, so they are removed here rather than asked
    about.
    """
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", str(text))
                 if part.strip()]
    kept = [part for part in sentences
            if not any(re.search(pattern, part, re.IGNORECASE)
                       for pattern in ABSENCE_PATTERNS)]
    if not kept:
        return str(text).strip()
    return " ".join(part.strip() for part in kept)


def overlap_ratio(used, consumed):
    """Share of a clip's elements that earlier clips already built on.

    The writer reports its own elements, and it reports them honestly even in
    the runs where it ignores the ban list, so this is a usable duplicate
    detector: 1.0 means the clip is built entirely on used-up ground.
    """
    fresh = {str(u).strip().lower() for u in used if str(u).strip()}
    if not fresh:
        return 0.0
    old_set = {str(u).strip().lower() for u in consumed if str(u).strip()}
    return len(fresh & old_set) / len(fresh)


def retry_directive(used, consumed):
    """Harder instruction for the second attempt at a duplicated clip."""
    repeated = sorted({str(u).strip().lower() for u in used}
                      & {str(u).strip().lower() for u in consumed})
    return (
        "YOUR PREVIOUS ATTEMPT WAS REJECTED. It built the action on %s, which "
        "earlier clips in this chain already used, so this clip would be "
        "indistinguishable from them. Write a different clip: choose another "
        "body part or another object in the room, and give it an action that "
        "leaves something in a place it has not been in yet."
        % ", ".join(repeated))


def chain_state_path(directory, chain_name):
    """File the per-clip state of one prompt chain is kept in."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(chain_name)).strip("._")
    return os.path.join(directory, "%s.json" % (safe or "h3_prompt_chain"))


def load_chain_state(path):
    """Read a saved chain, or an empty one when there is nothing to resume.

    A damaged file is treated as absent rather than raised on: the state is a
    convenience for partial re-runs, and losing it costs a re-run, not data.
    """
    empty = {"style": "", "clips": {}}
    if not path or not os.path.exists(path):
        return empty
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (ValueError, OSError):
        return empty
    if not isinstance(data, dict):
        return empty
    clips = data.get("clips")
    if not isinstance(clips, dict):
        clips = {}
    return {"style": str(data.get("style", "")),
            "clips": {str(k): v for k, v in clips.items()
                      if isinstance(v, dict)}}


def save_chain_state(path, style, clips):
    """Persist the chain so a later partial run can resume the continuity."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"style": str(style),
               "clips": {str(k): v for k, v in clips.items()}}
    with open(path, "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    return payload


def state_clip(state, index):
    """One clip's saved record, normalised to prompt/closing/used."""
    entry = state.get("clips", {}).get(str(int(index))) or {}
    return {"prompt": str(entry.get("prompt", "")),
            "closing": str(entry.get("closing", "")),
            "used": [str(u) for u in entry.get("used", []) if str(u).strip()]}


def assemble_clip_prompts(prompts):
    """Join per-clip prompts into the ``[n]`` block the Auto Chain node parses."""
    return "\n\n".join("[%d] %s" % (index, text.strip())
                       for index, text in enumerate(prompts, start=1))


def cut_segment(source, destination, start, end, sample_fps, height):
    """Cut ``[start, end)`` out of ``source`` into a small silent mp4."""
    duration = max(0.04, float(end) - float(start))
    # qwen_vl_utils refuses a clip with fewer than two frames, so a very short
    # tail segment is sampled denser rather than dropped.
    rate = max(float(sample_fps), 2.5 / duration)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error",
         "-ss", "%.3f" % float(start), "-i", str(source), "-t", "%.3f" % duration,
         "-an", "-vf", "fps=%.3f,scale=-2:%d" % (rate, int(height)),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
         str(destination)],
        check=True, capture_output=True)
    return destination


def extract_audio(source, destination):
    """Extract mono 16 kHz PCM audio for the ASR pass."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
         str(destination)],
        check=True, capture_output=True)
    return destination


def transcribe(path, model_size, language, device="cuda"):
    """Transcribe with faster-whisper in a child process.

    ctranslate2 aborts the interpreter when it cannot load cuDNN, so the call
    is isolated; a failing GPU run is retried on the CPU.
    """
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "whisper_worker.py")
    attempts = [device] if device == "cpu" else [device, "cpu"]
    last_error = ""
    for attempt in attempts:
        process = subprocess.run(
            [sys.executable, worker, str(path), str(model_size),
             str(language), attempt],
            capture_output=True, text=True)
        if process.returncode == 0 and process.stdout.strip():
            payload = json.loads(process.stdout)
            return payload["segments"], payload.get("language", language)
        last_error = (process.stderr.strip().splitlines() or [""])[-1]
        print("VideoToH3ClipPrompts: whisper on %s failed (%s)"
              % (attempt, last_error or "rc=%d" % process.returncode))
    raise RuntimeError("faster-whisper failed: %s" % last_error)


# ------------------------------------------------------------------------ node

class VideoToH3ClipPrompts:
    """Describe an existing video as a MiniMax H3 Auto Chain prompt set."""

    def __init__(self):
        self.processor = None
        self.model = None
        self.loaded_key = None

    @classmethod
    def INPUT_TYPES(cls):
        videos = []
        if folder_paths is not None:
            input_dir = folder_paths.get_input_directory()
            videos = [f for f in os.listdir(input_dir)
                      if os.path.isfile(os.path.join(input_dir, f))]
            videos = sorted(folder_paths.filter_files_content_types(
                videos, ["video"]))
        return {
            "required": {
                "video": (videos, {
                    "video_upload": True,
                    "tooltip": "Source video from the ComfyUI input folder. "
                               "Overridden by video_path when that is set.",
                }),
                "chunk_seconds": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 600.0, "step": 0.1,
                    "tooltip": "Clip length of the chain. Must match "
                               "chunk_seconds on the H3 Auto Chain Audio node, "
                               "otherwise the [n] tags address the wrong clips.",
                }),
                "fps": ("INT", {
                    "default": 24, "min": 1, "max": 240,
                    "tooltip": "Frame rate of the chain timeline. Must match "
                               "the Auto Chain node.",
                }),
                "model": (MODEL_CHOICES, {"default": "Huihui-Qwen3-VL-8B-Instruct-abliterated"}),
                "quantization": (["none", "4bit", "8bit"], {"default": "none"}),
                "attention": (["sdpa", "eager", "flash_attention_2"],
                              {"default": "sdpa"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "sample_fps": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 16.0, "step": 0.1,
                    "tooltip": "Frames per second handed to the VLM for each "
                               "clip segment. Higher catches faster motion and "
                               "costs VRAM.",
                }),
                "frame_height": ("INT", {
                    "default": 448, "min": 128, "max": 1080, "step": 8,
                    "tooltip": "Segments are downscaled to this height before "
                               "they reach the VLM.",
                }),
                "style_words": ("INT", {"default": 100, "min": 30, "max": 400}),
                "clip_words": ("INT", {"default": 110, "min": 40, "max": 400}),
                "max_new_tokens": ("INT", {
                    "default": 900, "min": 128, "max": 8192}),
                "temperature": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xFFFFFFFFFFFFFFFF}),
                "whisper_model": (WHISPER_CHOICES, {
                    "default": "small",
                    "tooltip": "Local ASR for the spoken lines. 'off' skips "
                               "the audio pass and produces silent prompts.",
                }),
                "transcribe_language": ("STRING", {
                    "default": "auto",
                    "tooltip": "ISO code such as en or de, or 'auto'.",
                }),
                "camera_discipline": ("STRING", {
                    "multiline": True, "default": DEFAULT_CAMERA_DISCIPLINE,
                    "tooltip": "Framing rules appended to the style block "
                               "verbatim and shown to the writer for every clip.",
                }),
                "style_instruction": ("STRING", {
                    "multiline": True, "default": DEFAULT_STYLE_INSTRUCTION}),
                "clip_instruction": ("STRING", {
                    "multiline": True, "default": DEFAULT_CLIP_INSTRUCTION}),
            },
            "optional": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to a video outside the input "
                               "folder. Takes precedence over the dropdown.",
                }),
                "style_prompt_override": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Skip the style pass and use this text as the "
                               "style block. Useful to re-run the clip pass "
                               "with a hand-edited appearance description.",
                }),
                "first_clip_is_establishing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clip 1 may contain the one camera move that "
                               "arrives at the presenter distance. Every later "
                               "clip holds that distance.",
                }),
                "start_clip": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "end_clip": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "0 means all clips to the end of the video.",
                }),
                "retry_overlap": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.01, "step": 0.05,
                    "tooltip": "Rewrite a clip once when this share of its "
                               "elements was already used by earlier clips. "
                               "1.01 disables the retry.",
                }),
                "chain_name": ("STRING", {
                    "default": "h3_prompt_chain",
                    "tooltip": "Name this prompt chain is saved under. Lets a "
                               "later run rewrite single clips while keeping "
                               "the continuity of the others.",
                }),
                "reset_state": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True starts a fresh chain and overwrites the "
                               "saved one. False resumes: clips outside "
                               "start_clip..end_clip are taken from the saved "
                               "chain, so the airlock and the ban list stay "
                               "intact and only the requested clips are "
                               "rewritten.",
                }),
                "keep_segments": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep the downscaled segment clips that were "
                               "handed to the model, under "
                               "output/h3_prompt_segments/<run>. They show "
                               "exactly what the model saw, which is what you "
                               "check when a clip prompt describes something "
                               "that is not there.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("style_prompt", "clip_prompts", "clip_count",
                    "transcript", "stats")
    FUNCTION = "describe"
    CATEGORY = "LLM"
    DESCRIPTION = ("Reads an existing video and writes a MiniMax H3 chain "
                   "prompt set: one constant style block plus one [n] tagged "
                   "prompt per clip, sized to the Auto Chain clip grid.")

    # ----------------------------------------------------------------- runtime

    def load_model(self, model, quantization, attention, min_pixels, max_pixels):
        import torch
        from transformers import (AutoProcessor, BitsAndBytesConfig,
                                  Qwen3VLForConditionalGeneration)

        key = (model, quantization, attention)
        if self.loaded_key == key and self.model is not None:
            return

        self.unload()
        checkpoint = os.path.join(folder_paths.models_dir, "prompt_generator",
                                  model)
        if not os.path.exists(checkpoint):
            from huggingface_hub import snapshot_download
            repo = "%s/%s" % (MODEL_REPOS.get(model, "qwen"), model)
            print("VideoToH3ClipPrompts: downloading %s" % repo)
            snapshot_download(repo_id=repo, local_dir=checkpoint,
                              local_dir_use_symlinks=False)

        self.processor = AutoProcessor.from_pretrained(
            checkpoint, min_pixels=min_pixels, max_pixels=max_pixels)
        config = None
        if quantization == "4bit":
            config = BitsAndBytesConfig(load_in_4bit=True)
        elif quantization == "8bit":
            config = BitsAndBytesConfig(load_in_8bit=True)
        device = (model_management.get_torch_device() if model_management
                  else "cuda")
        bf16 = (torch.cuda.is_available()
                and torch.cuda.get_device_capability(device)[0] >= 8)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint, dtype=torch.bfloat16 if bf16 else torch.float16,
            device_map="auto", attn_implementation=attention,
            quantization_config=config)
        self.loaded_key = key

    def unload(self):
        import torch
        self.processor = None
        self.model = None
        self.loaded_key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def ask(self, video_path, question, sample_fps, max_frames, prefill,
            max_new_tokens, temperature, seed):
        """One VLM call over one video file."""
        import torch
        from qwen_vl_utils import process_vision_info

        torch.manual_seed(int(seed))
        messages = [
            {"role": "system", "content":
                "You write prompts for MiniMax Hailuo H3, a video model with "
                "native audio. You answer with the prompt only."},
            {"role": "user", "content": [
                {"type": "video", "video": str(video_path),
                 "fps": float(sample_fps), "max_frames": int(max_frames)},
                {"type": "text", "text": question},
            ]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True) + prefill
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos,
                                padding=True, return_tensors="pt")
        device = (model_management.get_torch_device() if model_management
                  else "cuda")
        inputs = inputs.to(device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=int(max_new_tokens),
                do_sample=float(temperature) > 0.0,
                temperature=float(temperature) if temperature > 0 else None,
                top_p=0.9)
        trimmed = [out[len(inp):] for inp, out
                   in zip(inputs.input_ids, generated)]
        answer = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return answer[0] if answer else ""

    def describe(self, video, chunk_seconds, fps, model, quantization,
                 attention, keep_model_loaded, sample_fps, frame_height,
                 style_words, clip_words, max_new_tokens, temperature, seed,
                 whisper_model, transcribe_language, camera_discipline,
                 style_instruction, clip_instruction, video_path="",
                 style_prompt_override="", first_clip_is_establishing=True,
                 start_clip=1, end_clip=0, retry_overlap=0.75,
                 chain_name="h3_prompt_chain", reset_state=True,
                 keep_segments=False):

        source = (video_path.strip() or
                  folder_paths.get_annotated_filepath(video))
        if not os.path.exists(source):
            raise RuntimeError("VideoToH3ClipPrompts: no such video: %s" % source)

        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        run_id = "h3prompt_%d" % (time.time() * 1000)

        segment_dir = temp_dir
        if keep_segments:
            segment_dir = os.path.join(folder_paths.get_output_directory(),
                                       "h3_prompt_segments", run_id)
            os.makedirs(segment_dir, exist_ok=True)

        state_dir = os.path.join(folder_paths.get_output_directory(),
                                 "h3_prompt_state")
        state_file = chain_state_path(state_dir, chain_name)
        state = ({"style": "", "clips": {}} if reset_state
                 else load_chain_state(state_file))

        duration = probe_duration(source)
        clips = plan_clips(duration, chunk_seconds, fps)
        first = max(1, int(start_clip))
        last = len(clips) if int(end_clip) <= 0 else min(len(clips),
                                                         int(end_clip))
        if not reset_state:
            print("VideoToH3ClipPrompts: resuming chain %s with %d saved clips"
                  % (chain_name, len(state["clips"])))
        stats = ["source=%s" % os.path.basename(source),
                 "duration=%.2fs, fps=%d, chunk=%.2fs -> %d clips "
                 "(writing %d..%d)" % (duration, fps, chunk_seconds,
                                       len(clips), first, last)]
        print("VideoToH3ClipPrompts: %s" % stats[-1])
        stub = stub_clip_warning(clips, chunk_seconds)
        if stub:
            stats.append(stub)
            print("VideoToH3ClipPrompts: %s" % stub)

        # ------------------------------------------------------------- audio
        segments = []
        transcript_text = ""
        if whisper_model != "off":
            wav = os.path.join(temp_dir, "%s.wav" % run_id)
            started = time.time()
            try:
                extract_audio(source, wav)
                segments, detected = transcribe(wav, whisper_model,
                                                transcribe_language)
                transcript_text = "\n".join(
                    "[%6.2f - %6.2f] %s" % (s["start"], s["end"],
                                            s["text"].strip())
                    for s in segments)
                stats.append("asr: %s, %d segments, language=%s, %.1fs"
                             % (whisper_model, len(segments), detected,
                                time.time() - started))
            except Exception as error:
                stats.append("asr FAILED (%s) - continuing without dialogue"
                             % error)
            finally:
                if os.path.exists(wav):
                    os.remove(wav)
            print("VideoToH3ClipPrompts: %s" % stats[-1])

        # ------------------------------------------------------------- model
        max_pixels = int(frame_height * frame_height * 16 / 9)
        self.load_model(model, quantization, attention, 256 * 28 * 28,
                        max_pixels)
        max_frames_clip = max(4, int(math.ceil(chunk_seconds * sample_fps)) + 4)

        try:
            # --------------------------------------------------- style pass
            if style_prompt_override.strip():
                style_prompt = style_prompt_override.strip()
                stats.append("style: taken from override")
            elif state["style"].strip():
                style_prompt = state["style"].strip()
                stats.append("style: reused from chain %s" % chain_name)
            else:
                overview = os.path.join(segment_dir, "%s_overview.mp4" % run_id)
                cut_segment(source, overview, 0.0, duration,
                            min(1.0, 48.0 / max(duration, 1.0)), frame_height)
                started = time.time()
                raw = self.ask(overview, fill_placeholders(
                    style_instruction, style_words=style_words),
                    1.0, 48, PREFILL_STYLE, max_new_tokens, temperature, seed)
                style_prompt = strip_meta(raw)
                discipline = camera_discipline.strip()
                if discipline:
                    style_prompt = (style_prompt + " " + discipline).strip()
                if os.path.exists(overview) and not keep_segments:
                    os.remove(overview)
                stats.append("style: %d words, %.1fs"
                             % (len(style_prompt.split()), time.time() - started))
                if keep_segments:
                    stats.append("segments kept in %s" % segment_dir)
                print("VideoToH3ClipPrompts: %s" % stats[-1])

            # ---------------------------------------------------- clip pass
            prompts = []
            used_elements = []
            records = {}
            closing_state = ""

            # Clips that come AFTER the rewritten range are already written,
            # so their elements have to be banned up front - the loop would
            # only reach them once the rewrite is done, too late to matter.
            for later, _, _ in clips:
                if later > last:
                    used_elements.extend(state_clip(state, later)["used"])
            for index, start, end in clips:
                if index < first or index > last:
                    saved = state_clip(state, index)
                    prompts.append(saved["prompt"])
                    if saved["prompt"]:
                        # Feeding the saved clip back in is what keeps a
                        # partial re-run continuous: the next generated clip
                        # opens on this one and avoids what it already used.
                        used_elements.extend(saved["used"])
                        closing_state = saved["closing"] or closing_from_prompt(
                            saved["prompt"])
                        records[index] = saved
                    continue
                segment = os.path.join(segment_dir, "%s_%03d.mp4"
                                       % (run_id, index))
                cut_segment(source, segment, start, end, sample_fps,
                            frame_height)

                if index == 1 and first_clip_is_establishing:
                    opening = ("This is the opening clip. It may contain one "
                               "single, slow, smooth camera movement that "
                               "arrives at the presenter distance and then "
                               "holds there for the rest of the clip.")
                elif closing_state:
                    opening = ("Open by holding exactly this arrangement from "
                               "the previous clip for about two seconds of "
                               "quiet - same position, same framing, only "
                               "breathing and a small weight shift - before "
                               "anything else happens: \"%s\"" % closing_state)
                else:
                    opening = ("Open by holding the arrangement the previous "
                               "clip closed on for about two seconds of quiet "
                               "before anything else happens.")

                line = transcript_window(segments, start, end)
                if line:
                    dialogue = ("SPOKEN IN THIS SEGMENT, quote it word for "
                                "word and add nothing: \"%s\"" % line)
                else:
                    dialogue = ("Nobody speaks in this segment. Write no "
                                "quoted line and no vocal sound - no murmur, "
                                "hum, sigh or gasp. Do not mention the silence "
                                "either: a sentence like \"she does not speak\" "
                                "puts speech into the conditioning and renders "
                                "as speech. Leave sound out of the prompt "
                                "entirely, or name a non-vocal ambient sound.")

                question = "\n\n".join(part for part in [
                    "STYLE BLOCK - already prepended to your prompt "
                    "automatically, never repeat it:\n%s" % style_prompt,
                    "This is clip %d of %d of the chain and covers %.2fs to "
                    "%.2fs of the source. The attached video is exactly that "
                    "segment." % (index, len(clips), start, end),
                    dialogue,
                    "FRAMING RULES:\n%s" % camera_discipline.strip(),
                    recent_actions(prompts, used_elements),
                    fill_placeholders(clip_instruction,
                                      clip_words=clip_words, opening=opening),
                    'Answer as JSON: {"prompt": "...", "closing_state": "one '
                    'sentence naming the exact arrangement this clip ends on, '
                    'so the next clip can open holding it", "used": ["the body '
                    'parts and objects this clip built its action around, two '
                    'to five short entries"]}',
                ] if part)

                started = time.time()
                attempt_note = ""
                raw = self.ask(segment, question, sample_fps, max_frames_clip,
                               PREFILL_CLIP, max_new_tokens, temperature,
                               int(seed) + index)
                prompt, closing, used = parse_clip_answer(raw, PREFILL_CLIP)

                # One retry when the clip is built entirely on ground earlier
                # clips already covered - a chain of interchangeable clips
                # stops progressing, which is the failure this guards against.
                overlap = overlap_ratio(used, used_elements)
                if overlap >= float(retry_overlap) and used_elements:
                    retry_question = question + "\n\n" + retry_directive(
                        used, used_elements)
                    raw_retry = self.ask(
                        segment, retry_question, sample_fps, max_frames_clip,
                        PREFILL_CLIP, max_new_tokens,
                        min(2.0, float(temperature) + 0.2),
                        int(seed) + index + 9973)
                    retry_prompt, retry_closing, retry_used = parse_clip_answer(
                        raw_retry, PREFILL_CLIP)
                    retry_overlap_value = overlap_ratio(retry_used,
                                                        used_elements)
                    if retry_prompt and retry_overlap_value < overlap:
                        prompt, closing, used = (retry_prompt, retry_closing,
                                                 retry_used)
                        attempt_note = (", retried (overlap %.2f -> %.2f)"
                                        % (overlap, retry_overlap_value))
                    else:
                        attempt_note = (", retry kept original (overlap %.2f)"
                                        % overlap)

                if os.path.exists(segment) and not keep_segments:
                    os.remove(segment)
                if not prompt:
                    raise RuntimeError(
                        "VideoToH3ClipPrompts: clip %d produced no usable "
                        "prompt. Raw head: %r" % (index, raw[:200]))
                cleaned = strip_absence_sentences(prompt)
                if cleaned != prompt:
                    attempt_note += ", absence sentence removed"
                    prompt = cleaned
                prompts.append(prompt)
                used_elements.extend(used)
                closing_state = closing or closing_from_prompt(prompt)
                records[index] = {"prompt": prompt, "closing": closing_state,
                                  "used": used}
                stats.append("clip %d: %.2f-%.2fs, %d words, %.1fs, used=%s%s"
                             % (index, start, end, len(prompt.split()),
                                time.time() - started,
                                "/".join(used) if used else "-", attempt_note))
                print("VideoToH3ClipPrompts: %s" % stats[-1])
        finally:
            if not keep_model_loaded:
                self.unload()

        tagged = "\n\n".join(
            "[%d] %s" % (index, text.strip())
            for index, text in enumerate(prompts, start=1) if text.strip())

        save_chain_state(state_file, style_prompt, records)
        stats.append("chain %s saved with %d clips -> %s"
                     % (chain_name, len(records), state_file))
        print("VideoToH3ClipPrompts: %s" % stats[-1])

        return (style_prompt, tagged, len(clips), transcript_text,
                "\n".join(stats))


NODE_CLASS_MAPPINGS = {
    "VideoToH3ClipPrompts": VideoToH3ClipPrompts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoToH3ClipPrompts": "Video to H3 Clip Prompts (local VLM)",
}

"""Out-of-process faster-whisper transcription.

ctranslate2 loads its own cuDNN. When that fails it aborts the interpreter
instead of raising, which would take the whole ComfyUI server down, so the
transcription always runs in a child process. The child also frees its GPU
memory by exiting, which matters right before a VLM is loaded.

Usage: python whisper_worker.py <audio> <model_size> <language|auto> <device>
Prints a JSON document {"language": ..., "segments": [...]} on stdout.
"""

import ctypes
import json
import os
import sys


def preload_cudnn():
    """Put the pip-installed cuDNN 9 libraries on the loader path.

    ctranslate2 links against libcudnn_ops.so.9 but does not search the
    nvidia/* wheel directories, so the symbols have to be made global first.
    """
    try:
        import nvidia
    except ImportError:
        return False
    base = os.path.dirname(nvidia.__file__)
    loaded = False
    for package in ("cublas", "cudnn"):
        lib_dir = os.path.join(base, package, "lib")
        if not os.path.isdir(lib_dir):
            continue
        # graph before ops before the rest: later libraries need the symbols.
        names = sorted(os.listdir(lib_dir),
                       key=lambda n: ("graph" not in n, "ops" not in n, n))
        for name in names:
            if ".so" not in name:
                continue
            try:
                ctypes.CDLL(os.path.join(lib_dir, name), mode=ctypes.RTLD_GLOBAL)
                loaded = True
            except OSError:
                pass
    return loaded


def main():
    audio, model_size, language, device = sys.argv[1:5]
    if device == "cuda":
        preload_cudnn()

    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_size, device=device,
        compute_type="float16" if device == "cuda" else "int8")
    segments, info = model.transcribe(
        audio, language=None if language == "auto" else language,
        vad_filter=True, beam_size=5)
    result = {
        "language": getattr(info, "language", language),
        "segments": [{"start": s.start, "end": s.end, "text": s.text}
                     for s in segments],
    }
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()

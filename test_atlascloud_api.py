"""Offline tests for the Atlas Cloud nodes: no network, no API key needed.

Run inside the ComfyUI venv:
    python custom_nodes/comfyui_MDPack/test_atlascloud_api.py
"""

import contextlib
import io
import json
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, __file__.rsplit("/", 2)[0])
import atlascloud_api as ac  # noqa: E402

failures = []


def check(name, condition, detail=""):
    print(("PASS " if condition else "FAIL ") + name + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


# --- response parsing -------------------------------------------------------
check("prediction id from data.id",
      ac.extract_prediction_id({"data": {"id": "abc"}}) == "abc")
check("prediction id from request_id",
      ac.extract_prediction_id({"request_id": "xyz"}) == "xyz")

check("outputs from data.outputs list",
      ac.extract_outputs({"status": "completed", "outputs": ["https://a/1.png"]}) == ["https://a/1.png"])
check("outputs from output.image_url",
      ac.extract_outputs({"output": {"image_url": "https://a/2.png"}}) == ["https://a/2.png"])
check("outputs from plain output string",
      ac.extract_outputs({"output": "https://a/3.mp4"}) == ["https://a/3.mp4"])
check("outputs from images list of dicts",
      ac.extract_outputs({"images": [{"url": "https://a/4.png"}, {"url": "https://a/5.png"}]})
      == ["https://a/4.png", "https://a/5.png"])
check("outputs deduplicated",
      ac.extract_outputs({"outputs": ["https://a/6.png", "https://a/6.png"]}) == ["https://a/6.png"])
check("no outputs for empty body", ac.extract_outputs({"status": "processing"}) == [])
check("outputs from output.download_url",
      ac.extract_outputs({"output": {"download_url": "https://a/7.png"}}) == ["https://a/7.png"])

# --- uploadMedia response shapes -------------------------------------------
_real_request_json = ac.request_json


def upload_returning(body):
    def stub(method, url, headers, timeout, **kwargs):
        return body
    return stub


# The live uploadMedia endpoint answers with data.download_url, not data.url.
ac.request_json = upload_returning({
    "code": 200, "message": "success",
    "data": {"type": "image", "download_url": "https://cdn.test/uploaded.png",
             "filename": "uploaded.png", "size": 123},
})
check("uploadMedia accepts data.download_url",
      ac.upload_media("key", b"x", "a.png", "image/png", 5) == "https://cdn.test/uploaded.png")
ac.request_json = upload_returning({"url": "https://cdn.test/plain.png"})
check("uploadMedia accepts top-level url",
      ac.upload_media("key", b"x", "a.png", "image/png", 5) == "https://cdn.test/plain.png")
ac.request_json = _real_request_json

# --- extra_params -----------------------------------------------------------
check("extra_params empty", ac.parse_extra_params("  ") == {})
check("extra_params object", ac.parse_extra_params('{"cfg": 3.5}') == {"cfg": 3.5})
try:
    ac.parse_extra_params("{nope}")
    check("extra_params invalid JSON raises", False)
except ValueError:
    check("extra_params invalid JSON raises", True)
try:
    ac.parse_extra_params("[1,2]")
    check("extra_params non-object raises", False)
except ValueError:
    check("extra_params non-object raises", True)

# --- missing api key --------------------------------------------------------
try:
    ac.build_headers("")
    check("empty api_key raises", False)
except ValueError:
    check("empty api_key raises", True)

# --- request payload built by the image node --------------------------------
captured = {}


def fake_request_json(method, url, headers, timeout, **kwargs):
    captured.setdefault("calls", []).append((method, url, kwargs.get("json")))
    if url == ac.IMAGE_ENDPOINT or url == ac.VIDEO_ENDPOINT:
        return {"data": {"id": "pred-1"}}
    if url == ac.UPLOAD_ENDPOINT:
        uploads = captured.setdefault("uploads", [])
        uploads.append(url)
        return {"url": f"https://cdn.test/uploaded_{len(uploads)}.png"}
    return {"data": {"status": "completed", "outputs": ["https://cdn.test/out.png"]}}


def image_of(value, size=8):
    """Distinct image content per value, so the upload cache can be exercised."""
    return torch.full((1, size, size, 3), float(value) / 255.0)


def fake_download_bytes(url, timeout, label="Atlas Cloud"):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 32), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


ac.request_json = fake_request_json
ac.download_bytes = fake_download_bytes

# Everything below this point exercises the node's own logic (media routing,
# seed/num_images merging, the upload cache) rather than the Atlas Cloud
# catalogue, so the schema lookup is stubbed to always report "no schema
# known" -- the same shape schema_for_model returns for a model it has never
# indexed. The node's own fallback (see resolve_model_schema in
# atlascloud_api.py) still routes the prompt to the conventional "prompt"
# field in that case, which is why the prompt-related checks below keep
# passing unchanged. Schema-driven field names and size derivation are
# covered separately in the schema-driven block near the end of this file.
ac.schema_for_model = lambda model_identifier, **kwargs: {
    "model": model_identifier, "prompt_field": "", "image_fields": [], "fields": [],
}

node = ac.AtlasCloudImage()
images, urls, info = node.generate(
    api_key="k", model="bytedance/seedream-v5.0-pro/edit", prompt="a cat",
    output_type="Image", input_type="Image",
    width=1024, height=0, num_images=1, seed=42, poll_interval=0.01, timeout=30,
    image_field="images", image_transport="upload", upload_cache_minutes=0,
    extra_params='{"guidance": 3.5}', params_json="{}", input_map="",
    image=torch.zeros((1, 8, 8, 3)),
)
payload = [call[2] for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0]
check("image payload keeps model/prompt",
      payload["model"] == "bytedance/seedream-v5.0-pro/edit" and payload["prompt"] == "a cat")
# A lone width (no height) used to be sent verbatim -- exactly the class of
# bug this feature fixes (Seedream declares a "size" string and silently
# ignored "width"). With no schema known for this call the dimensions are
# now correctly dropped instead of guessed; the schema-aware size derivation
# itself is covered in the schema-driven block below.
check("width alone is dropped when no schema declares width or size",
      "width" not in payload and "height" not in payload and "size" not in payload,
      json.dumps(payload))
check("num_images omitted when 1", "num_images" not in payload)
check("seed sent when >= 0", payload.get("seed") == 42)
check("uploaded image lands in the configured plural field",
      payload.get("images") == ["https://cdn.test/uploaded_1.png"], json.dumps(payload))
check("extra_params merged", payload.get("guidance") == 3.5)
check("image tensor shape (1,32,64,3)", tuple(images.shape) == (1, 32, 64, 3), str(images.shape))
check("image tensor float 0..1",
      images.dtype == torch.float32 and float(images.max()) <= 1.0)
check("URLs returned", urls == "https://cdn.test/out.png")
check("info mentions prediction", "pred-1" in info)

captured.clear()
images, urls, info = node.generate(
    api_key="k", model="m", prompt="p", output_type="Image", input_type="Image",
    width=0, height=0, num_images=3, seed=-1,
    poll_interval=0.01, timeout=30, image_field="images", image_transport="upload",
    upload_cache_minutes=0, extra_params="", params_json="{}", input_map="",
    image=torch.cat([image_of(11), image_of(12)]),
)
payload = [call[2] for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0]
check("every image of a batch becomes one reference",
      len(payload.get("images", [])) == 2, json.dumps(payload))
check("seed omitted when -1", "seed" not in payload)
check("num_images sent when > 1", payload.get("num_images") == 3)

# --- several reference inputs of different sizes -----------------------------
captured.clear()
node.generate(
    api_key="k", model="bytedance/seedream-v5.0-pro/edit", prompt="p",
    output_type="Image", input_type="Image",
    width=0, height=0, num_images=1, seed=1, poll_interval=0.01, timeout=30,
    image_field="images", image_transport="upload", upload_cache_minutes=0,
    extra_params="", params_json="{}", input_map="",
    image=image_of(21, size=8),
    image_2=image_of(22, size=16),
    image_3=torch.cat([image_of(23, size=32), image_of(24, size=32)]),
    reference_urls="https://cdn.test/hosted_a.png\n\nhttps://cdn.test/hosted_b.png",
)
payload = [call[2] for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0]
check("differently sized inputs are concatenated in order",
      payload["images"][:4] == [f"https://cdn.test/uploaded_{i}.png" for i in range(1, 5)],
      json.dumps(payload))
check("reference_urls appended after the uploads",
      payload["images"][4:] == ["https://cdn.test/hosted_a.png", "https://cdn.test/hosted_b.png"],
      json.dumps(payload))
check("six references in total", len(payload["images"]) == 6)

# --- singular field name keeps only the first reference ----------------------
captured.clear()
node.generate(
    api_key="k", model="atlascloud/step1x-edit", prompt="p",
    output_type="Image", input_type="Image",
    width=0, height=0, num_images=1, seed=1, poll_interval=0.01, timeout=30,
    image_field="image", image_transport="upload", upload_cache_minutes=0,
    extra_params="", params_json="{}", input_map="",
    image=image_of(31), image_2=image_of(32),
)
payload = [call[2] for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0]
check("singular image_field gets a plain string",
      isinstance(payload.get("image"), str) and "images" not in payload, json.dumps(payload))

# --- base64 transport does not upload ---------------------------------------
captured.clear()
node.generate(
    api_key="k", model="m", prompt="p", output_type="Image", input_type="Image",
    width=0, height=0, num_images=1, seed=1,
    poll_interval=0.01, timeout=30, image_field="images", image_transport="base64",
    upload_cache_minutes=0, extra_params="", params_json="{}", input_map="",
    image=image_of(41),
)
payload = [call[2] for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0]
check("base64 transport inlines a data URI",
      payload["images"][0].startswith("data:image/png;base64,"), payload["images"][0][:40])
check("base64 transport skips uploadMedia",
      not any(call[1] == ac.UPLOAD_ENDPOINT for call in captured["calls"]))

# --- upload cache ------------------------------------------------------------
ac._upload_cache.clear()
captured.clear()
captured["uploads"] = []
for run in range(3):
    node.generate(
        api_key="k", model="m", prompt=f"prompt {run}",
        output_type="Image", input_type="Image", width=0, height=0,
        num_images=1, seed=-1, poll_interval=0.01, timeout=30, image_field="images",
        image_transport="upload", upload_cache_minutes=120, extra_params="",
        params_json="{}", input_map="",
        image=torch.cat([image_of(51), image_of(52)]),
    )
check("identical references upload once across runs",
      len(captured["uploads"]) == 2, f"{len(captured['uploads'])} uploads")
payload = [call[2] for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][-1]
check("cached run still sends both references", len(payload["images"]) == 2)

captured["uploads"] = []
node.generate(
    api_key="k", model="m", prompt="p", output_type="Image", input_type="Image",
    width=0, height=0, num_images=1, seed=-1,
    poll_interval=0.01, timeout=30, image_field="images", image_transport="upload",
    upload_cache_minutes=120, extra_params="", params_json="{}", input_map="",
    image=image_of(53),
)
check("changed reference content uploads again", len(captured["uploads"]) == 1)

ac._upload_cache.clear()
captured["uploads"] = []
for run in range(2):
    node.generate(
        api_key="k", model="m", prompt="p", output_type="Image", input_type="Image",
        width=0, height=0, num_images=1, seed=-1,
        poll_interval=0.01, timeout=30, image_field="images", image_transport="upload",
        upload_cache_minutes=0, extra_params="", params_json="{}", input_map="",
        image=image_of(54),
    )
check("cache_minutes=0 uploads every run", len(captured["uploads"]) == 2)

# --- video node reference handling ------------------------------------------
captured.clear()
video_node = ac.AtlasCloudVideo()
ac.download_bytes = lambda url, timeout: b""
try:
    video_node.generate(
        api_key="k", output_type="Video", input_type="Any",
        model="kling-v2.0", prompt="p", duration=5, aspect_ratio="16:9",
        resolution="", seed=-1, poll_interval=0.01, timeout=30, max_frames=0,
        frame_stride=1, image_field="image_url", image_transport="upload",
        upload_cache_minutes=0, params_json="{}", input_map="",
        extra_params="", image=image_of(61),
    )
except Exception:
    pass  # decoding empty bytes fails; only the payload matters here
payload = [call[2] for call in captured["calls"] if call[1] == ac.VIDEO_ENDPOINT][0]
check("video singular image_url gets a string",
      isinstance(payload.get("image_url"), str), json.dumps(payload))
check("video duration and aspect_ratio sent",
      payload.get("duration") == 5 and payload.get("aspect_ratio") == "16:9")
check("video resolution omitted when empty", "resolution" not in payload)
ac.download_bytes = fake_download_bytes

# --- IS_CHANGED -------------------------------------------------------------
check("IS_CHANGED is NaN without seed", ac.AtlasCloudImage.IS_CHANGED(seed=-1) != ac.AtlasCloudImage.IS_CHANGED(seed=-1))
check("IS_CHANGED stable with seed", ac.AtlasCloudVideo.IS_CHANGED(seed=7) == 7)

# --- failed prediction ------------------------------------------------------
def failing_request_json(method, url, headers, timeout, **kwargs):
    if url == ac.VIDEO_ENDPOINT:
        return {"data": {"id": "pred-2"}}
    return {"data": {"status": "failed", "error": "content policy"}}


ac.request_json = failing_request_json
try:
    ac.submit_and_poll(ac.VIDEO_ENDPOINT, "k", {"model": "m"}, 0.01, 5, 30, "test")
    check("failed status raises", False)
except RuntimeError as e:
    check("failed status raises", "content policy" in str(e), str(e))

# --- timeout ----------------------------------------------------------------
def pending_request_json(method, url, headers, timeout, **kwargs):
    if url == ac.VIDEO_ENDPOINT:
        return {"data": {"id": "pred-3"}}
    return {"data": {"status": "processing"}}


ac.request_json = pending_request_json
try:
    ac.submit_and_poll(ac.VIDEO_ENDPOINT, "k", {"model": "m"}, 0.05, 1, 30, "test")
    check("timeout raises", False)
except RuntimeError as e:
    check("timeout raises", "timeout" in str(e).lower(), str(e))

# --- video decoding ---------------------------------------------------------
try:
    import av

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p"
        for index in range(12):
            picture = np.full((32, 64, 3), index * 10, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(picture, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    video_bytes = buffer.getvalue()

    frames, fps, path = ac.video_bytes_to_frames(video_bytes, 0, 1, "test")
    check("all frames decoded", tuple(frames.shape) == (12, 32, 64, 3), str(frames.shape))
    check("fps detected", abs(fps - 24.0) < 0.01, str(fps))
    check("frames float 0..1", frames.dtype == torch.float32 and float(frames.max()) <= 1.0)

    frames, fps, path = ac.video_bytes_to_frames(video_bytes, 4, 1, "test")
    check("max_frames honored", frames.shape[0] == 4, str(frames.shape))

    frames, fps, path = ac.video_bytes_to_frames(video_bytes, 0, 2, "test")
    check("frame_stride honored", frames.shape[0] == 6 and abs(fps - 12.0) < 0.01,
          f"{frames.shape} fps={fps}")
except ImportError:
    check("PyAV available for video tests", False, "PyAV missing")

# --- schema-driven payload on the node --------------------------------------
import atlascloud_models as am  # noqa: E402

# The timeout/failed-prediction tests above left ac.request_json pointing at
# their own stand-ins; restore the general-purpose fake before driving the
# node again.
ac.request_json = fake_request_json

STUB_EDIT_SCHEMA = {
    "model": "bytedance/seedream-v5.0-pro/edit",
    "prompt_field": "prompt",
    "image_fields": [{"name": "images", "is_list": True, "max_items": 10}],
    "fields": [
        {"name": "size", "type": "string", "default": "2048*2048",
         "enum": ["2048*2048", "1328*1776"], "required": False,
         "description": "", "is_media": False, "is_list": False},
    ],
}
am.schema_for_model = lambda model_identifier, **kwargs: STUB_EDIT_SCHEMA
ac.schema_for_model = am.schema_for_model

captured.clear()
node = ac.AtlasCloudImage()
images, urls, info = node.generate(
    api_key="key", model="bytedance/seedream-v5.0-pro/edit", prompt="a cat",
    output_type="Image", input_type="Image",
    width=1328, height=1776, num_images=1, seed=7,
    poll_interval=0.0, timeout=30, image_field="", image_transport="upload",
    upload_cache_minutes=0, params_json="{}", input_map="", extra_params="",
    image=image_of(11),
)
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("node sends size instead of width and height",
      submitted.get("size") == "1328*1776" and "width" not in submitted, str(submitted))
check("node writes references into the schema media field",
      isinstance(submitted.get("images"), list) and len(submitted["images"]) == 1,
      str(submitted))

# image_field's declared default must be empty so a freshly added node takes
# the schema-driven assign_media() path by default -- "images" as a default
# would make the override branch win on every fresh node and leave
# assign_media() dead code for anyone who never touches the widget. A
# workflow saved before this change keeps its saved "images" value and is
# unaffected either way (ComfyUI restores the saved value, not the class
# default).
image_field_widget = ac.AtlasCloudImage.INPUT_TYPES()["required"]["image_field"]
check("image_field defaults to empty so new nodes use the schema by default",
      image_field_widget[1]["default"] == "", str(image_field_widget))

# A schema with two singular media fields (no plural "images" at all) proves
# assign_media() honours the schema's own field names in order rather than
# the node hardcoding "images" anywhere.
TWO_SINGULAR_FIELDS_SCHEMA = {
    "model": "test/two-image-fields",
    "prompt_field": "prompt",
    "image_fields": [
        {"name": "image", "is_list": False, "max_items": 1},
        {"name": "last_image", "is_list": False, "max_items": 1},
    ],
    "fields": [],
}
ac.schema_for_model = lambda model_identifier, **kwargs: TWO_SINGULAR_FIELDS_SCHEMA
captured.clear()
images, urls, info = node.generate(
    api_key="key", model="test/two-image-fields", prompt="a cat",
    output_type="Image", input_type="Image",
    width=0, height=0, num_images=1, seed=7,
    poll_interval=0.0, timeout=30, image_field="", image_transport="upload",
    upload_cache_minutes=0, params_json="{}", input_map="", extra_params="",
    image=image_of(15), image_2=image_of(16),
)
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("default image_field fills two singular schema fields by name, in order",
      submitted.get("image") == "https://cdn.test/uploaded_1.png"
      and submitted.get("last_image") == "https://cdn.test/uploaded_2.png",
      str(submitted))
ac.schema_for_model = am.schema_for_model

captured.clear()
images, urls, info = node.generate(
    api_key="key", model="bytedance/seedream-v5.0-pro/edit", prompt="a cat",
    output_type="Image", input_type="Image",
    width=0, height=0, num_images=1, seed=7,
    poll_interval=0.0, timeout=30, image_field="", image_transport="upload",
    upload_cache_minutes=0, params_json='{"size": "2048*2048"}', input_map="",
    extra_params='{"size": "1024*1024"}', image=image_of(12),
)
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("params_json feeds the payload and extra_params still wins",
      submitted.get("size") == "1024*1024", str(submitted))

# --- FIX 2: width/height set alongside a params_json-supplied size ----------
# This is Finding 1 from the final review reproduced end to end: the web
# extension writes every schema field's default into params_json the moment
# a model is selected (before FIX 2b, that included "size"), so a user who
# then set width/height on the node got them silently discarded because
# build_payload saw "size" already in the payload. Precedence (params_json
# wins) is unchanged; what changed is that it is no longer silent.
collision_output = io.StringIO()
captured.clear()
with contextlib.redirect_stdout(collision_output):
    node.generate(
        api_key="key", model="bytedance/seedream-v5.0-pro/edit", prompt="a cat",
        output_type="Image", input_type="Image",
        width=1328, height=1776, num_images=1, seed=7,
        poll_interval=0.0, timeout=30, image_field="", image_transport="upload",
        upload_cache_minutes=0, params_json='{"size": "2048*2048"}', input_map="",
        extra_params="", image=image_of(19),
    )
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("width/height colliding with an existing params_json size keeps the explicit size",
      submitted.get("size") == "2048*2048", str(submitted))
check("the collision prints one diagnostic naming the model and both sizes",
      "2048*2048" in collision_output.getvalue() and "1328" in collision_output.getvalue()
      and "1776" in collision_output.getvalue()
      and "bytedance/seedream-v5.0-pro/edit" in collision_output.getvalue(),
      collision_output.getvalue())

captured.clear()
images, urls, info = node.generate(
    api_key="key", model="bytedance/seedream-v5.0-pro/edit", prompt="a cat",
    output_type="Image", input_type="Image",
    width=0, height=0, num_images=1, seed=7,
    poll_interval=0.0, timeout=30, image_field="image_url", image_transport="upload",
    upload_cache_minutes=0, params_json="{}", input_map="", extra_params="",
    image=image_of(13),
)
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("explicit image_field overrides the schema media field",
      "image_url" in submitted and "images" not in submitted, str(submitted))
check("info names the override field, not the schema's own media field",
      "field=image_url" in info and "field=images" not in info, info)

captured.clear()
images, urls, info = node.generate(
    api_key="key", model="bytedance/seedream-v5.0-pro/edit", prompt="a cat",
    output_type="Image", input_type="Image",
    width=0, height=0, num_images=1, seed=7,
    poll_interval=0.0, timeout=30, image_field="", image_transport="upload",
    upload_cache_minutes=0, params_json="{}", input_map="", extra_params="",
    image=image_of(21), image_5=image_of(22), image_10=image_of(23),
)
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("image slots 5 to 10 are collected too", len(submitted.get("images", [])) == 3,
      str(submitted))

# --- schema lookup failure falls back instead of crashing --------------------
captured.clear()


def failing_schema_lookup(model_identifier, **kwargs):
    raise RuntimeError("network is down")


ac.schema_for_model = failing_schema_lookup
images, urls, info = node.generate(
    api_key="key", model="totally/unknown-model", prompt="a cat",
    output_type="Image", input_type="Image",
    width=0, height=0, num_images=1, seed=7,
    poll_interval=0.0, timeout=30, image_field="", image_transport="upload",
    upload_cache_minutes=0, params_json="{}", input_map="", extra_params="",
    image=image_of(14),
)
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("a failed schema lookup does not crash the node and still sends the prompt",
      submitted.get("prompt") == "a cat", str(submitted))
check("a failed schema lookup falls back to the images field for references",
      submitted.get("images") == ["https://cdn.test/uploaded_1.png"], str(submitted))
check("a failed schema lookup is named in the returned info",
      "schema=unknown" in info, info)

# The old, pre-schema node sent width/height unconditionally for any model.
# When nothing at all is known about a model's schema (lookup failed, or the
# model simply is not in the Atlas catalogue yet), that is still strictly
# better than dropping the user's dimensions -- so the fallback schema must
# carry its own width/height fields and build_payload must send them raw.
captured.clear()
images, urls, info = node.generate(
    api_key="key", model="totally/unknown-model", prompt="a cat",
    output_type="Image", input_type="Image",
    width=768, height=1024, num_images=1, seed=7,
    poll_interval=0.0, timeout=30, image_field="", image_transport="upload",
    upload_cache_minutes=0, params_json="{}", input_map="", extra_params="",
    image=image_of(17),
)
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("a fallback schema still sends raw width/height",
      submitted.get("width") == 768 and submitted.get("height") == 1024
      and "size" not in submitted, str(submitted))
check("info names the fallback when width/height are involved too",
      "schema=unknown" in info, info)
ac.schema_for_model = am.schema_for_model

# --- a real (non-fallback) schema with zero image fields but images connected
NO_IMAGE_FIELDS_SCHEMA = {
    "model": "test/no-image-fields",
    "prompt_field": "prompt",
    "image_fields": [],
    "fields": [
        {"name": "style", "type": "string", "default": "vivid", "enum": None,
         "required": False, "description": "", "is_media": False, "is_list": False},
    ],
}
ac.schema_for_model = lambda model_identifier, **kwargs: NO_IMAGE_FIELDS_SCHEMA
captured.clear()
images, urls, info = node.generate(
    api_key="key", model="test/no-image-fields", prompt="a cat",
    output_type="Image", input_type="Image",
    width=0, height=0, num_images=1, seed=7,
    poll_interval=0.0, timeout=30, image_field="", image_transport="upload",
    upload_cache_minutes=0, params_json="{}", input_map="", extra_params="",
    image=image_of(18),
)
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("a schema with no image fields sends no reference at all",
      "images" not in submitted and "image" not in submitted, str(submitted))
check("dropping connected references because the schema has no image field "
      "is named in the returned info",
      "references dropped" in info and "test/no-image-fields" in info, info)
ac.schema_for_model = am.schema_for_model

# --- FIX 4: a schema whose only media field is video/audio-kind ------------
# 25 live video models declare their first (sometimes only) media field as
# `video`/`audio` (e.g. kling-v2.6-pro/avatar). A connected IMAGE input used
# to be uploaded and silently written into that field; now it is treated the
# same as "no image field at all" from the node's point of view (references
# dropped, named in the info string), but with a message that says which
# non-image field(s) actually block it.
VIDEO_ONLY_MEDIA_SCHEMA = {
    "model": "test/video-only-media",
    "prompt_field": "prompt",
    "image_fields": [{"name": "video", "is_list": False, "max_items": 1, "kind": "video"}],
    "fields": [],
}
ac.schema_for_model = lambda model_identifier, **kwargs: VIDEO_ONLY_MEDIA_SCHEMA
video_only_output = io.StringIO()
captured.clear()
with contextlib.redirect_stdout(video_only_output):
    images, urls, info = node.generate(
        api_key="key", model="test/video-only-media", prompt="a cat",
        output_type="Image", input_type="Image",
        width=0, height=0, num_images=1, seed=7,
        poll_interval=0.0, timeout=30, image_field="", image_transport="upload",
        upload_cache_minutes=0, params_json="{}", input_map="", extra_params="",
        image=image_of(22),
    )
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("a connected image is never written into a video-kind field",
      "video" not in submitted, str(submitted))
check("the connected reference is named as dropped in the returned info",
      "references dropped" in info and "test/video-only-media" in info, info)
check("the console diagnostic names the model and the blocking non-image field",
      "test/video-only-media" in video_only_output.getvalue()
      and "video" in video_only_output.getvalue()
      and "non-image" in video_only_output.getvalue(),
      video_only_output.getvalue())
ac.schema_for_model = am.schema_for_model

# --- video node: schema-driven payload, mirroring the image node above ------
STUB_KLING_SCHEMA = {
    "model": "kwaivgi/kling-v2.5-turbo-pro/image-to-video",
    "prompt_field": "prompt",
    "image_fields": [{"name": "image", "is_list": False, "max_items": 1},
                     {"name": "last_image", "is_list": False, "max_items": 1}],
    "fields": [
        {"name": "duration", "type": "integer", "default": 5, "enum": [5, 10],
         "required": False, "description": "", "is_media": False, "is_list": False},
    ],
}
ac.schema_for_model = lambda model_identifier, **kwargs: STUB_KLING_SCHEMA

captured.clear()
video_node = ac.AtlasCloudVideo()
frames, fps, urls, video_url, info = video_node.generate(
    api_key="key", model="kwaivgi/kling-v2.5-turbo-pro/image-to-video", prompt="a cat",
    output_type="Video", input_type="Image", duration=0, aspect_ratio="", resolution="",
    seed=7, poll_interval=0.0, timeout=30, max_frames=0, frame_stride=1,
    image_field="", image_transport="upload", upload_cache_minutes=0,
    params_json='{"duration": 10}', input_map='{"image": 0, "last_image": 1}',
    extra_params="", image=image_of(31), image_2=image_of(32),
)
submitted = [call for call in captured["calls"] if call[1] == ac.VIDEO_ENDPOINT][0][2]
check("video node sends schema parameters", submitted.get("duration") == 10, str(submitted))
check("input_map assigns first and last frame",
      submitted.get("image") and submitted.get("last_image")
      and submitted["image"] != submitted["last_image"], str(submitted))
check("video node omits fields the schema does not declare",
      "aspect_ratio" not in submitted and "resolution" not in submitted, str(submitted))

# The brief's own test above leaves aspect_ratio/resolution empty, which
# would pass this check even without the "declared" gating (an empty widget
# is never sent regardless). This variant sets non-empty legacy values
# against the same schema (which only declares "duration") to actually
# exercise the gating from Corrections item 2.
captured.clear()
video_node.generate(
    api_key="key", model="kwaivgi/kling-v2.5-turbo-pro/image-to-video", prompt="a cat",
    output_type="Video", input_type="Image", duration=0, aspect_ratio="16:9",
    resolution="1080p", seed=-1, poll_interval=0.0, timeout=30, max_frames=0, frame_stride=1,
    image_field="", image_transport="upload", upload_cache_minutes=0,
    params_json="{}", input_map="", extra_params="",
)
submitted = [call for call in captured["calls"] if call[1] == ac.VIDEO_ENDPOINT][0][2]
check("legacy aspect_ratio/resolution are dropped when the schema declares neither",
      "aspect_ratio" not in submitted and "resolution" not in submitted, str(submitted))

# --- params_json wins over a legacy widget carrying a non-default value -----
# The video node reads params_json into `parameters` first and only
# `setdefault`s the legacy duration widget on top, so a schema-driven value
# must win even when the legacy widget was left at a real, non-zero value
# (not just its 0/empty "omit" default) rather than by coincidence of
# ordering. Uncovered until now -- the earlier tests either left params_json
# empty or left the legacy widget at its default.
captured.clear()
video_node.generate(
    api_key="key", model="kwaivgi/kling-v2.5-turbo-pro/image-to-video", prompt="a cat",
    output_type="Video", input_type="Image", duration=99, aspect_ratio="", resolution="",
    seed=-1, poll_interval=0.0, timeout=30, max_frames=0, frame_stride=1,
    image_field="", image_transport="upload", upload_cache_minutes=0,
    params_json='{"duration": 10}', input_map="", extra_params="",
)
submitted = [call for call in captured["calls"] if call[1] == ac.VIDEO_ENDPOINT][0][2]
check("params_json wins over a legacy widget carrying a non-default value",
      submitted.get("duration") == 10, str(submitted))

# --- FIX 3: the node's own seed/num_images widgets win over params_json -----
# The opposite precedence from duration/aspect_ratio/resolution just above:
# seed and num_images are the widgets ComfyUI's own IS_CHANGED and
# "control_after_generate" are tied to, so a schema-driven param_seed/
# param_num_images value must never silently outrank them (Finding 3 of the
# final review: a fixed seed=42 on the node used to be overridden by a random
# params_json seed while IS_CHANGED still reported the node's 42, so ComfyUI
# cached the result as reproducible even though a different seed was sent).
# A schema declaring seed/num_images itself (unlike the stubs used above),
# matching the 162/20 live models the final review counted, so build_payload's
# own "unknown key" diagnostic (unrelated to this fix) stays silent and does
# not muddy the empty-output assertions below.
STUB_SEED_SCHEMA = {
    "model": "bytedance/seedream-v5.0-pro/edit",
    "prompt_field": "prompt",
    "image_fields": [{"name": "images", "is_list": True, "max_items": 10}],
    "fields": [
        {"name": "seed", "type": "integer", "default": -1, "enum": None,
         "required": False, "description": "", "is_media": False, "is_list": False},
        {"name": "num_images", "type": "integer", "default": 1, "enum": None,
         "required": False, "description": "", "is_media": False, "is_list": False},
    ],
}
ac.schema_for_model = lambda model_identifier, **kwargs: STUB_SEED_SCHEMA

image_seed_output = io.StringIO()
captured.clear()
with contextlib.redirect_stdout(image_seed_output):
    node.generate(
        api_key="key", model="bytedance/seedream-v5.0-pro/edit", prompt="a cat",
        output_type="Image", input_type="Image",
        width=0, height=0, num_images=3, seed=42,
        poll_interval=0.0, timeout=30, image_field="", image_transport="upload",
        upload_cache_minutes=0, params_json='{"seed": 99, "num_images": 7}', input_map="",
        extra_params="", image=image_of(20),
    )
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("the node's own seed widget wins over a colliding params_json seed",
      submitted.get("seed") == 42, str(submitted))
check("the node's own num_images widget wins over a colliding params_json num_images",
      submitted.get("num_images") == 3, str(submitted))
check("the seed/num_images override prints one diagnostic per overridden field",
      "params_json seed=99" in image_seed_output.getvalue()
      and "node's own seed widget (42)" in image_seed_output.getvalue()
      and "params_json num_images=7" in image_seed_output.getvalue()
      and "node's own num_images widget (3)" in image_seed_output.getvalue(),
      image_seed_output.getvalue())
check("IS_CHANGED is consistent with what was actually sent: the node's own seed",
      ac.AtlasCloudImage.IS_CHANGED(seed=42) == submitted.get("seed") == 42)

# No collision, no diagnostic: params_json is free to carry its own seed when
# the node's own widget is left at its "unset" sentinel (-1).
image_no_collision_output = io.StringIO()
captured.clear()
with contextlib.redirect_stdout(image_no_collision_output):
    node.generate(
        api_key="key", model="bytedance/seedream-v5.0-pro/edit", prompt="a cat",
        output_type="Image", input_type="Image",
        width=0, height=0, num_images=1, seed=-1,
        poll_interval=0.0, timeout=30, image_field="", image_transport="upload",
        upload_cache_minutes=0, params_json='{"seed": 5}', input_map="",
        extra_params="", image=image_of(21),
    )
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("params_json's own seed applies when the node's seed widget is unset",
      submitted.get("seed") == 5, str(submitted))
# generate() always prints unrelated lines (upload, submit, poll status), so
# this checks the override diagnostic specifically is absent, not that
# stdout is empty.
check("no override diagnostic when there is nothing to override",
      "overridden by the node's own" not in image_no_collision_output.getvalue(),
      image_no_collision_output.getvalue())

# --- num_images is routed to the field name the model actually declares -----
# The widget is called num_images, but only 20 of the 119 live image models
# call the field that: 7 take "n", 6 (the Seedream /sequential variants) take
# "max_images", and ~86 -- including bytedance/seedream-v5.0-pro/edit -- have
# no such field at all. Sending "num_images" regardless meant the API ignored
# it and returned one image while the node reported success: a num_images=2
# run on seedream-v5.0-pro/edit produced exactly one image, with nothing but a
# console line to explain it.
STUB_N_SCHEMA = {
    "model": "alibaba/wan-2.7/text-to-image",
    "prompt_field": "prompt",
    "image_fields": [],
    "fields": [
        {"name": "n", "type": "integer", "default": 1, "enum": None,
         "required": False, "description": "", "is_media": False, "is_list": False},
    ],
}
ac.schema_for_model = lambda model_identifier, **kwargs: STUB_N_SCHEMA
alias_output = io.StringIO()
captured.clear()
with contextlib.redirect_stdout(alias_output):
    node.generate(
        api_key="key", model="alibaba/wan-2.7/text-to-image", prompt="a cat",
        output_type="Image", input_type="Image", width=0, height=0,
        num_images=4, seed=-1, poll_interval=0.0, timeout=30, image_field="",
        image_transport="upload", upload_cache_minutes=0, params_json="{}",
        input_map="", extra_params="",
    )
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("num_images is sent under the schema's own count field name",
      submitted.get("n") == 4 and "num_images" not in submitted, str(submitted))
check("the alias routing is named in a diagnostic",
      "num_images=4" in alias_output.getvalue() and "'n'" in alias_output.getvalue(),
      alias_output.getvalue())

# A model with no count field cannot return more than one image, so the
# request stays clean and the node says so in its Info output rather than
# only on the console -- the Info string is what the workflow shows.
STUB_NO_COUNT_SCHEMA = {
    "model": "bytedance/seedream-v5.0-pro/edit",
    "prompt_field": "prompt",
    "image_fields": [{"name": "images", "is_list": True, "max_items": 10}],
    "fields": [
        {"name": "size", "type": "string", "default": "2048*2048", "enum": None,
         "required": False, "description": "", "is_media": False, "is_list": False},
    ],
}
ac.schema_for_model = lambda model_identifier, **kwargs: STUB_NO_COUNT_SCHEMA
no_count_output = io.StringIO()
captured.clear()
with contextlib.redirect_stdout(no_count_output):
    _, _, no_count_info = node.generate(
        api_key="key", model="bytedance/seedream-v5.0-pro/edit", prompt="a cat",
        output_type="Image", input_type="Image", width=0, height=0,
        num_images=2, seed=-1, poll_interval=0.0, timeout=30, image_field="",
        image_transport="upload", upload_cache_minutes=0, params_json="{}",
        input_map="", extra_params="", image=image_of(22),
    )
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("no count field: nothing is sent under a name the model ignores",
      "num_images" not in submitted and "n" not in submitted and "max_images" not in submitted,
      str(submitted))
check("no count field: the console says the request cannot produce several images",
      "num_images=2" in no_count_output.getvalue()
      and "no image count field" in no_count_output.getvalue(), no_count_output.getvalue())
check("no count field: the node's Info output carries the same warning",
      "num_images=2" in no_count_info and "no image count field" in no_count_info,
      no_count_info)

# num_images=1 asks for nothing unusual, so a model without a count field
# must stay silent -- otherwise every ordinary Seedream run would warn.
quiet_output = io.StringIO()
captured.clear()
with contextlib.redirect_stdout(quiet_output):
    _, _, quiet_info = node.generate(
        api_key="key", model="bytedance/seedream-v5.0-pro/edit", prompt="a cat",
        output_type="Image", input_type="Image", width=0, height=0,
        num_images=1, seed=-1, poll_interval=0.0, timeout=30, image_field="",
        image_transport="upload", upload_cache_minutes=0, params_json="{}",
        input_map="", extra_params="", image=image_of(23),
    )
check("num_images=1 on a model without a count field warns about nothing",
      "no image count field" not in quiet_output.getvalue()
      and "no image count field" not in quiet_info, quiet_output.getvalue())

# The node's own widget still outranks a colliding params_json value, now
# under the aliased name too (param_max_images filled in by the browser).
STUB_MAX_IMAGES_SCHEMA = {
    "model": "bytedance/seedream-v4.5/sequential",
    "prompt_field": "prompt",
    "image_fields": [],
    "fields": [
        {"name": "max_images", "type": "integer", "default": 1, "enum": None,
         "required": False, "description": "", "is_media": False, "is_list": False},
    ],
}
ac.schema_for_model = lambda model_identifier, **kwargs: STUB_MAX_IMAGES_SCHEMA
collision_output = io.StringIO()
captured.clear()
with contextlib.redirect_stdout(collision_output):
    node.generate(
        api_key="key", model="bytedance/seedream-v4.5/sequential", prompt="a cat",
        output_type="Image", input_type="Image", width=0, height=0,
        num_images=3, seed=-1, poll_interval=0.0, timeout=30, image_field="",
        image_transport="upload", upload_cache_minutes=0,
        params_json='{"max_images": 7}', input_map="", extra_params="",
    )
submitted = [call for call in captured["calls"] if call[1] == ac.IMAGE_ENDPOINT][0][2]
check("the node's own num_images widget wins over a colliding params_json max_images",
      submitted.get("max_images") == 3, str(submitted))
check("the aliased collision prints a diagnostic naming both values",
      "max_images=7" in collision_output.getvalue()
      and "num_images widget (3)" in collision_output.getvalue(),
      collision_output.getvalue())
# Restore the stub the video checks below were written against, not the real
# lookup: those assert on seed precedence, not on any live schema.
ac.schema_for_model = lambda model_identifier, **kwargs: STUB_SEED_SCHEMA

video_seed_output = io.StringIO()
captured.clear()
with contextlib.redirect_stdout(video_seed_output):
    video_node.generate(
        api_key="key", model="kwaivgi/kling-v2.5-turbo-pro/image-to-video", prompt="a cat",
        output_type="Video", input_type="Image", duration=0, aspect_ratio="", resolution="",
        seed=11, poll_interval=0.0, timeout=30, max_frames=0, frame_stride=1,
        image_field="", image_transport="upload", upload_cache_minutes=0,
        params_json='{"seed": 123}', input_map="", extra_params="",
    )
submitted = [call for call in captured["calls"] if call[1] == ac.VIDEO_ENDPOINT][0][2]
check("the video node's own seed widget also wins over a colliding params_json seed",
      submitted.get("seed") == 11, str(submitted))
check("the video node prints the same override diagnostic",
      "params_json seed=123" in video_seed_output.getvalue()
      and "node's own seed widget (11)" in video_seed_output.getvalue(),
      video_seed_output.getvalue())
check("IS_CHANGED matches what the video node actually sent",
      ac.AtlasCloudVideo.IS_CHANGED(seed=11) == submitted.get("seed") == 11)
ac.schema_for_model = am.schema_for_model

# output_type default and the widgets Task 8's browser wiring depends on
# (see web/atlasCloudModels.js: buildParamWidgets/readParams/writeParams key
# off a widget literally named "params_json", regardless of node type).
video_input_types = ac.AtlasCloudVideo.INPUT_TYPES()
check("video output_type defaults to Video",
      video_input_types["required"]["output_type"][1]["default"] == "Video",
      str(video_input_types["required"]["output_type"]))
check("video node exposes a params_json widget for the schema-driven parameters",
      "params_json" in video_input_types["required"])
check("video image_field defaults to empty so the schema's field names are used",
      video_input_types["required"]["image_field"][1]["default"] == "",
      str(video_input_types["required"]["image_field"]))
check("video node has image slots up to image_10",
      all(f"image_{i}" in video_input_types["optional"] for i in range(2, 11))
      and "image" in video_input_types["optional"])

# --- uncatalogued video model: duration/aspect_ratio/resolution still work --
# The pre-schema node sent these three unconditionally for any model. When
# nothing at all is known about a model (schema_for_model reports the empty
# shape it uses for an unindexed model), that must remain true -- otherwise a
# saved workflow using duration/aspect_ratio/resolution on such a model would
# silently stop sending them the moment this feature lands.
captured.clear()
ac.schema_for_model = lambda model_identifier, **kwargs: {
    "model": model_identifier, "prompt_field": "", "image_fields": [], "fields": [],
}
ac.download_bytes = lambda url, timeout: b""
try:
    video_node.generate(
        api_key="k", output_type="Video", input_type="Any",
        model="totally/unknown-video-model", prompt="p",
        duration=8, aspect_ratio="9:16", resolution="1080p", seed=-1,
        poll_interval=0.01, timeout=30, max_frames=0, frame_stride=1,
        image_field="", image_transport="upload", upload_cache_minutes=0,
        params_json="{}", input_map="", extra_params="",
    )
except Exception:
    pass  # decoding empty bytes fails; only the payload matters here
payload = [call[2] for call in captured["calls"] if call[1] == ac.VIDEO_ENDPOINT][0]
check("an uncatalogued model still sends duration/aspect_ratio/resolution via the fallback",
      payload.get("duration") == 8 and payload.get("aspect_ratio") == "9:16"
      and payload.get("resolution") == "1080p", json.dumps(payload))
ac.download_bytes = fake_download_bytes
ac.schema_for_model = am.schema_for_model

# --- shared diagnostic: two distinct input_map failure modes ----------------
# 1) input_map indices assume one reference per connected slot, but a slot
#    fed by a multi-image batch contributes several references
#    (collect_reference_images emits one per image, not one per slot) -- the
#    browser cannot see tensor shapes to correct for this.
# 2) assign_media's input_map branch only ever writes the indices input_map
#    names; any collected reference nothing points to (an extra
#    reference_urls entry, or more images than the map covers) is silently
#    left out of the request. Combining reference_urls with input_map used to
#    trip the *first* diagnostic too (it counted every collected reference,
#    not just slot-derived ones, against the connected-slot count), blaming a
#    batch that was never involved -- fixed to count only slot-derived
#    references there, and the second diagnostic now covers this case
#    properly instead.
# Both are exercised directly against the shared helper, which isolates their
# own logic from either node's plumbing, using the two-singular-media-field
# schema already defined above (fields "image" / "last_image").

# Case 1: a 2-image batch behind a single connected slot. Both collected
# references are still placed somewhere (assign_media has no way to know
# they're misrouted), so only the batch diagnostic fires, not the unused one.
mismatch_output = io.StringIO()
with contextlib.redirect_stdout(mismatch_output):
    ac.warn_if_input_map_may_misroute(
        [torch.cat([image_of(71), image_of(72)]), None],  # one connected slot, a 2-image batch
        ["url-a", "url-b"],  # two references collected from that one slot
        {"image": 0, "last_image": 1}, TWO_SINGULAR_FIELDS_SCHEMA, "test/mismatch-model", "TestLabel",
    )
check("batch/index mismatch is diagnosed when references outnumber connected slots",
      "may misroute" in mismatch_output.getvalue()
      and "test/mismatch-model" in mismatch_output.getvalue(),
      mismatch_output.getvalue())
check("the batch case does not also report an unused reference",
      "will not reach" not in mismatch_output.getvalue(), mismatch_output.getvalue())

# Case 2: two ordinary single-image slots (no batch) plus a third reference
# (as reference_urls would add) that input_map has no field left for -- warns
# about the unused reference, not about a batch, since there isn't one.
unused_output = io.StringIO()
with contextlib.redirect_stdout(unused_output):
    ac.warn_if_input_map_may_misroute(
        [image_of(71), image_of(72)],  # two connected slots, one image each: no batch
        ["url-a", "url-b", "url-c"],  # url-c: e.g. from reference_urls, unmapped
        {"image": 0, "last_image": 1}, TWO_SINGULAR_FIELDS_SCHEMA, "test/mismatch-model", "TestLabel",
    )
check("an input_map that leaves a collected reference unmapped is diagnosed",
      "will not reach" in unused_output.getvalue()
      and "test/mismatch-model" in unused_output.getvalue(),
      unused_output.getvalue())
check("the unused-reference case does not also report a batch mismatch",
      "may misroute" not in unused_output.getvalue(), unused_output.getvalue())

# Case 3: the ordinary case -- one image per connected slot, input_map covers
# every collected reference. Warns about neither.
clean_output = io.StringIO()
with contextlib.redirect_stdout(clean_output):
    ac.warn_if_input_map_may_misroute(
        [image_of(71), image_of(72)],  # two connected slots, one image each
        ["url-a", "url-b"],
        {"image": 0, "last_image": 1}, TWO_SINGULAR_FIELDS_SCHEMA, "test/mismatch-model", "TestLabel",
    )
check("no diagnostic when connected slots match the reference count and input_map covers everything",
      clean_output.getvalue() == "", repr(clean_output.getvalue()))

no_map_output = io.StringIO()
with contextlib.redirect_stdout(no_map_output):
    ac.warn_if_input_map_may_misroute(
        [torch.cat([image_of(71), image_of(72)]), None], ["url-a", "url-b"], None,
        TWO_SINGULAR_FIELDS_SCHEMA, "test/mismatch-model", "TestLabel",
    )
check("no diagnostic when input_map is unused",
      no_map_output.getvalue() == "", repr(no_map_output.getvalue()))

# --- end-to-end: the video node's generate() actually wires the diagnostic in
captured.clear()
ac.schema_for_model = lambda model_identifier, **kwargs: STUB_KLING_SCHEMA
end_to_end_output = io.StringIO()
with contextlib.redirect_stdout(end_to_end_output):
    try:
        video_node.generate(
            api_key="key", model="kwaivgi/kling-v2.5-turbo-pro/image-to-video", prompt="a cat",
            output_type="Video", input_type="Image", duration=0, aspect_ratio="", resolution="",
            seed=-1, poll_interval=0.0, timeout=30, max_frames=0, frame_stride=1,
            image_field="", image_transport="upload", upload_cache_minutes=0,
            params_json="{}", input_map='{"image": 0, "last_image": 1}',
            extra_params="", image=torch.cat([image_of(73), image_of(74)]),
        )
    except Exception:
        pass  # decoding the fake response fails; only the diagnostic matters here
check("AtlasCloudVideo.generate wires the batch mismatch diagnostic in",
      "may misroute" in end_to_end_output.getvalue(), end_to_end_output.getvalue())
ac.schema_for_model = am.schema_for_model


# --- prompt lists and max_parallel ------------------------------------------
# AtlasCloudImage declares INPUT_IS_LIST, so a prompt-list node ("CR Prompt
# List") hands all of its prompts to one execution instead of making ComfyUI
# run the node once per prompt. These checks cover the three things that must
# hold for that: every prompt gets its own request, requests really do overlap
# up to max_parallel, and the images come back in prompt order regardless of
# which prediction finishes first.
import threading  # noqa: E402
import time  # noqa: E402

ac.download_bytes = fake_download_bytes

check("AtlasCloudImage declares INPUT_IS_LIST", ac.AtlasCloudImage.INPUT_IS_LIST is True)
check("provider is declared first, right above the key it governs",
      list(ac.AtlasCloudImage.INPUT_TYPES()["required"].keys())[:2] == ["provider", "api_key"],
      str(list(ac.AtlasCloudImage.INPUT_TYPES()["required"].keys())[:3]))
check("max_parallel stays the last widget, where it was appended",
      list(ac.AtlasCloudImage.INPUT_TYPES()["required"].keys())[-1] == "max_parallel",
      str(list(ac.AtlasCloudImage.INPUT_TYPES()["required"].keys())[-3:]))
check("AtlasCloudVideo declares provider first as well",
      list(ac.AtlasCloudVideo.INPUT_TYPES()["required"].keys())[:2] == ["provider", "api_key"],
      str(list(ac.AtlasCloudVideo.INPUT_TYPES()["required"].keys())[:3]))

check("first_value collapses a one-entry list", ac.first_value(["a"], "z") == "a")
check("first_value passes a plain value through", ac.first_value(7, 0) == 7)
check("first_value falls back for an empty list", ac.first_value([], 5) == 5)
check("first_value falls back for None", ac.first_value(None, 5) == 5)
check("prompt_values keeps a lone empty prompt", ac.prompt_values([""]) == [""])
check("prompt_values drops blank entries of a real list",
      ac.prompt_values(["a", "  ", "b", ""]) == ["a", "b"])
check("prompt_values accepts a plain string", ac.prompt_values("a") == ["a"])
check("image_batches_of flattens slot lists and drops empty slots",
      len(ac.image_batches_of([[image_of(81)], None, [image_of(82), image_of(83)]])) == 3)

ac.schema_for_model = lambda model_identifier, **kwargs: {
    "model": model_identifier, "prompt_field": "", "image_fields": [], "fields": [],
}

parallel_state = {"active": 0, "peak": 0, "payloads": [], "uploads": 0}
parallel_lock = threading.Lock()


def parallel_request_json(method, url, headers, timeout, **kwargs):
    """Fake API where each prediction takes measurable time.

    The first prompt is deliberately the slowest, so a run that returns its
    images in prompt order proves the ordering is reconstructed rather than
    accidentally matching completion order.
    """
    if url == ac.UPLOAD_ENDPOINT:
        with parallel_lock:
            parallel_state["uploads"] += 1
            index = parallel_state["uploads"]
        return {"url": f"https://cdn.test/ref_{index}.png"}
    if url == ac.IMAGE_ENDPOINT:
        body = kwargs.get("json")
        with parallel_lock:
            parallel_state["payloads"].append(body)
            parallel_state["active"] += 1
            parallel_state["peak"] = max(parallel_state["peak"], parallel_state["active"])
        prompt_text = body.get("prompt", "")
        if prompt_text == "fail me":
            with parallel_lock:
                parallel_state["active"] -= 1
            raise RuntimeError("Atlas Cloud: HTTP 500 for this prompt")
        # prompt "p0" waits longest, "p3" shortest.
        try:
            rank = int(prompt_text[1:])
        except ValueError:
            rank = 0
        time.sleep(max(0.01, 0.12 - 0.02 * rank))
        with parallel_lock:
            parallel_state["active"] -= 1
        return {"data": {"id": f"pred-{prompt_text}"}}
    return {"data": {"status": "completed",
                     "outputs": [f"https://cdn.test/{url.rsplit('/', 1)[-1]}.png"]}}


_previous_request_json = ac.request_json
ac.request_json = parallel_request_json
ac._upload_cache.clear()

PROMPT_BATCH = ["p0", "p1", "p2", "p3"]
images, urls, info = node.generate(
    api_key=["k"], model=["m"], prompt=list(PROMPT_BATCH),
    output_type=["Image"], input_type=["Image"], width=[0], height=[0],
    num_images=[1], seed=[-1], poll_interval=[0.0], timeout=[30],
    image_field=["images"], image_transport=["upload"], upload_cache_minutes=[120],
    extra_params=[""], params_json=["{}"], input_map=[""], max_parallel=[4],
    image=[image_of(91)],
)
sent_prompts = [payload.get("prompt") for payload in parallel_state["payloads"]]
check("every prompt of the list becomes its own request",
      sorted(sent_prompts) == sorted(PROMPT_BATCH), str(sent_prompts))
check("max_parallel=4 really runs four requests at once",
      parallel_state["peak"] == 4, f"peak={parallel_state['peak']}")
check("one image per prompt comes back", tuple(images.shape)[0] == 4, str(images.shape))
check("outputs keep prompt order, not completion order",
      urls.splitlines() == [f"https://cdn.test/pred-{p}.png" for p in PROMPT_BATCH],
      urls.replace("\n", " | "))
check("info reports how many prompts ran", "prompts=4/4" in info, info)
check("references are uploaded once for the whole batch",
      parallel_state["uploads"] == 1, f"{parallel_state['uploads']} uploads")
check("every request carries the shared reference",
      all(payload.get("images") == ["https://cdn.test/ref_1.png"]
          for payload in parallel_state["payloads"]), str(parallel_state["payloads"][0]))

# max_parallel=1 keeps the old strictly sequential behaviour.
parallel_state.update({"active": 0, "peak": 0, "payloads": []})
node.generate(
    api_key=["k"], model=["m"], prompt=list(PROMPT_BATCH),
    output_type=["Image"], input_type=["Image"], width=[0], height=[0],
    num_images=[1], seed=[-1], poll_interval=[0.0], timeout=[30],
    image_field=["images"], image_transport=["upload"], upload_cache_minutes=[120],
    extra_params=[""], params_json=["{}"], input_map=[""], max_parallel=[1],
)
check("max_parallel=1 sends one request at a time",
      parallel_state["peak"] == 1, f"peak={parallel_state['peak']}")

# A failing prompt must not cost the images the other prompts produced.
parallel_state.update({"active": 0, "peak": 0, "payloads": []})
images, urls, info = node.generate(
    api_key=["k"], model=["m"], prompt=["p0", "fail me", "p2"],
    output_type=["Image"], input_type=["Image"], width=[0], height=[0],
    num_images=[1], seed=[-1], poll_interval=[0.0], timeout=[30],
    image_field=["images"], image_transport=["upload"], upload_cache_minutes=[120],
    extra_params=[""], params_json=["{}"], input_map=[""], max_parallel=[3],
)
check("a failed prompt still returns the other images",
      tuple(images.shape)[0] == 2, str(images.shape))
check("info names the failed prompt", "failed prompts:" in info and "[2]" in info, info)
check("info reports the partial count", "prompts=2/3" in info, info)

# Only a batch where nothing succeeded fails the node.
try:
    node.generate(
        api_key=["k"], model=["m"], prompt=["fail me", "fail me"],
        output_type=["Image"], input_type=["Image"], width=[0], height=[0],
        num_images=[1], seed=[-1], poll_interval=[0.0], timeout=[30],
        image_field=["images"], image_transport=["upload"], upload_cache_minutes=[120],
        extra_params=[""], params_json=["{}"], input_map=[""], max_parallel=[2],
    )
    check("a batch with no successful request raises", False)
except RuntimeError as error:
    check("a batch with no successful request raises", "all 2 requests failed" in str(error), str(error))

# A single failing prompt keeps raising the API's own error, unwrapped.
try:
    node.generate(
        api_key=["k"], model=["m"], prompt=["fail me"],
        output_type=["Image"], input_type=["Image"], width=[0], height=[0],
        num_images=[1], seed=[-1], poll_interval=[0.0], timeout=[30],
        image_field=["images"], image_transport=["upload"], upload_cache_minutes=[120],
        extra_params=[""], params_json=["{}"], input_map=[""], max_parallel=[4],
    )
    check("a single failing prompt raises the original error", False)
except RuntimeError as error:
    check("a single failing prompt raises the original error",
          "HTTP 500 for this prompt" in str(error), str(error))

ac.request_json = _previous_request_json
ac._upload_cache.clear()


# --- FIX 1: a workflow saved by the pre-schema node must still load correctly
# ComfyUI serializes a node's widgets_values as a positional array and
# restores it on load by zipping it against the widgets in declaration order.
# output_type/input_type were originally inserted right after api_key
# (positions 1-2), so every later value shifted by two: the model id landed
# in output_type, the prompt in input_type, width in model, and so on. These
# are the *actual* values from a user's real saved workflow (a queued
# Seedream text-to-image run, api_key redacted) -- reproduced here to prove
# the current declaration order (output_type/input_type moved to the very
# end of `required`, see AtlasCloudImage/AtlasCloudVideo.INPUT_TYPES above)
# restores each value into the widget it belonged to before this feature
# added the two new widgets. Confirmed to fail against the pre-fix ordering
# (output_type/input_type right after api_key) before this fix landed.
OLD_SAVED_IMAGE_WIDGETS_VALUES = [
    "apikey-REDACTED", "bytedance/seedream-v5.0-pro/text-to-image", "a prompt", 1328, 1776, 1,
    3208795585, "randomize", 2, 300, "images", "upload", 120, "", "",
]


def declaration_order_with_control_after_generate(required_widget_names):
    """The name sequence ComfyUI actually serializes widgets_values against:
    INPUT_TYPES()["required"] order, with an extra "control_after_generate"
    slot injected right after "seed" -- ComfyUI's frontend adds that
    companion widget automatically for any INT widget it treats as a seed,
    but INPUT_TYPES() itself never lists it."""
    names = []
    for name in required_widget_names:
        names.append(name)
        if name == "seed":
            names.append("control_after_generate")
    return names


PROVIDER_VALUES = set(ac.PROVIDER_IDS)


def migrated_widgets_values(stored, widget_order):
    """Mirror of migrateWidgetValues() in web/atlasCloudModels.js.

    `provider` is declared first, above the api_key it governs, which shifts
    every value of a workflow saved under either older layout (provider last,
    or no provider at all) by one widget. The browser repairs that on load;
    this reproduces the same rule so the Python side can assert what the
    repaired array has to look like. The JS suite tests the real function.
    """
    stored = list(stored)
    if stored and stored[0] in PROVIDER_VALUES:
        return stored  # already saved under the current layout
    static_count = len(widget_order)
    stored_provider = stored[static_count - 1] if len(stored) >= static_count else None
    provider = stored_provider if stored_provider in PROVIDER_VALUES else ac.DEFAULT_PROVIDER
    return [provider] + stored[:static_count - 1]


new_image_widget_order = declaration_order_with_control_after_generate(
    list(ac.AtlasCloudImage.INPUT_TYPES()["required"].keys())
)
restored = dict(zip(new_image_widget_order,
                    migrated_widgets_values(OLD_SAVED_IMAGE_WIDGETS_VALUES,
                                            new_image_widget_order)))
check("a pre-provider workflow lands on the default provider",
      restored.get("provider") == "atlascloud", str(restored))
check("a pre-provider workflow restores the api key into 'api_key', not 'provider'",
      restored.get("api_key") == "apikey-REDACTED", str(restored))

# The generation in between: every widget of the current node existed, but
# `provider` was declared last instead of first -- so the array holds exactly
# one value per static widget, with the provider at the end.
PROVIDER_LAST_WIDGETS_VALUES = [
    "apikey-REDACTED", "bytedance/seedream-v5.0-pro/text-to-image", "a prompt", 1328, 1776, 1,
    3208795585, "randomize", 2, 300, "images", "upload", 120, "", "{}", "", "Image", "Any", 4,
    "wavespeed",
]
provider_last_restored = dict(zip(
    new_image_widget_order,
    migrated_widgets_values(PROVIDER_LAST_WIDGETS_VALUES, new_image_widget_order),
))
check("a workflow saved with provider last keeps its provider",
      provider_last_restored.get("provider") == "wavespeed", str(provider_last_restored))
check("and every other value still lands where it belongs",
      provider_last_restored.get("api_key") == "apikey-REDACTED"
      and provider_last_restored.get("max_parallel") == 4
      and provider_last_restored.get("output_type") == "Image", str(provider_last_restored))

# A workflow saved under the current layout must be left exactly as it is.
current_layout_values = ["wavespeed"] + OLD_SAVED_IMAGE_WIDGETS_VALUES
check("a workflow already saved provider-first is not shifted again",
      migrated_widgets_values(current_layout_values, new_image_widget_order)
      == current_layout_values)
check("a saved pre-schema workflow restores the model id into 'model', not 'output_type'",
      restored.get("model") == "bytedance/seedream-v5.0-pro/text-to-image", str(restored))
check("a saved pre-schema workflow restores the prompt into 'prompt', not 'input_type'",
      restored.get("prompt") == "a prompt", str(restored))
check("a saved pre-schema workflow restores width", restored.get("width") == 1328, str(restored))
check("a saved pre-schema workflow restores height", restored.get("height") == 1776, str(restored))
check("a saved pre-schema workflow restores num_images", restored.get("num_images") == 1, str(restored))
check("a saved pre-schema workflow restores seed", restored.get("seed") == 3208795585, str(restored))
check("a saved pre-schema workflow restores poll_interval", restored.get("poll_interval") == 2, str(restored))
check("a saved pre-schema workflow restores timeout", restored.get("timeout") == 300, str(restored))
check("a saved pre-schema workflow restores image_field",
      restored.get("image_field") == "images", str(restored))
check("a saved pre-schema workflow restores image_transport",
      restored.get("image_transport") == "upload", str(restored))
check("a saved pre-schema workflow restores upload_cache_minutes",
      restored.get("upload_cache_minutes") == 120, str(restored))
check("a saved pre-schema workflow restores extra_params", restored.get("extra_params") == "", str(restored))

# The video node's pre-schema order differs (duration/aspect_ratio/resolution
# instead of width/height/image_field-as-third-widget); a lighter check that
# the same append-at-the-end placement holds there too, spot-checking the
# fields most likely to shift (model, prompt, duration, image_field).
OLD_SAVED_VIDEO_WIDGETS_VALUES = [
    "apikey-REDACTED", "kwaivgi/kling-v2.5-turbo-pro/image-to-video", "a video prompt", 5, "16:9",
    "1080p", 42, "fixed", 5, 900, 0, 1, "image_url", "upload", 120, "",
]
new_video_widget_order = declaration_order_with_control_after_generate(
    list(ac.AtlasCloudVideo.INPUT_TYPES()["required"].keys())
)
video_restored = dict(zip(new_video_widget_order,
                          migrated_widgets_values(OLD_SAVED_VIDEO_WIDGETS_VALUES,
                                                  new_video_widget_order)))
check("a saved pre-schema video workflow restores the model id into 'model'",
      video_restored.get("model") == "kwaivgi/kling-v2.5-turbo-pro/image-to-video", str(video_restored))
check("a saved pre-schema video workflow restores the prompt",
      video_restored.get("prompt") == "a video prompt", str(video_restored))
check("a saved pre-schema video workflow restores duration",
      video_restored.get("duration") == 5, str(video_restored))
check("a saved pre-schema video workflow restores image_field",
      video_restored.get("image_field") == "image_url", str(video_restored))

# --- reference_urls only ever carries URLs -----------------------------------
# A widget-order shift once put the provider id ("wavespeed") into
# reference_urls, from where it was appended to the request as a seventh
# reference image. Seedream answered with "Invalid base64-encoded string:
# number of data characters (9) cannot be 1 more than a multiple of 4", which
# points nowhere near the cause. Anything that is not a URL is now dropped and
# named instead.
accepted, rejected = ac.collect_reference_images(
    "key", [], "https://cdn.test/a.png\nwavespeed\n  \ndata:image/png;base64,AAAA\nnot a url",
    "upload", 30, 0, "T",
)
check("http(s) and data URIs are kept, in order",
      accepted == ["https://cdn.test/a.png", "data:image/png;base64,AAAA"], str(accepted))
check("a bare word never reaches the request as a reference image",
      rejected == ["wavespeed", "not a url"], str(rejected))

empty_accepted, empty_rejected = ac.collect_reference_images(
    "key", [], "", "upload", 30, 0, "T",
)
check("an empty reference_urls stays empty and rejects nothing",
      empty_accepted == [] and empty_rejected == [])

reference_node = ac.AtlasCloudImage()
ac.request_json = fake_request_json
ac.schema_for_model = lambda model_identifier, **kwargs: {
    "model": model_identifier, "prompt_field": "prompt",
    "image_fields": [{"name": "images", "is_list": True, "max_items": 10, "kind": "image"}],
    "fields": [],
}
_, _, reference_info = reference_node.generate(
    api_key="key", output_type="Image", input_type="Any",
    model="bytedance/seedream-v5.0-pro/edit", prompt="p", width=0, height=0, num_images=1,
    seed=-1, poll_interval=0.0, timeout=30, image_field="", image_transport="upload",
    upload_cache_minutes=0, extra_params="", params_json="{}", input_map="", max_parallel=1,
    image=image_of(11), reference_urls="wavespeed",
)
sent_payload = next(call[2] for call in reversed(captured["calls"]) if call[2] is not None)
check("the node never sends a non-URL reference line",
      all(url.startswith(("http", "data:")) for url in sent_payload["images"]),
      str(sent_payload["images"]))
check("and Info names the line it dropped, rather than leaving the API to complain",
      "reference_urls: 1 non-URL line(s) not sent: 'wavespeed'" in reference_info,
      reference_info)

# --- WaveSpeed: endpoints, polling and uploads -------------------------------
import hashlib  # noqa: E402

# WaveSpeed differs from Atlas Cloud in exactly three places: the model is
# named by the request URL instead of the body, the result URL comes back with
# the submit response, and the upload endpoint answers with download_url.

check("Atlas Cloud keeps one endpoint per output kind",
      (ac.submit_endpoint("atlascloud", "any/model", "image"),
       ac.submit_endpoint("atlascloud", "any/model", "video"))
      == (ac.IMAGE_ENDPOINT, ac.VIDEO_ENDPOINT))

ac.submit_path = lambda model_identifier, provider="atlascloud", api_key="": (
    f"/api/v3/{model_identifier}" if model_identifier == "wavespeed-ai/flux-2-pro/text-to-image"
    else ""
)
check("a catalogued WaveSpeed model is submitted to the path its schema declares",
      ac.submit_endpoint("wavespeed", "wavespeed-ai/flux-2-pro/text-to-image", "image")
      == "https://api.wavespeed.ai/api/v3/wavespeed-ai/flux-2-pro/text-to-image")
check("an uncatalogued WaveSpeed model still gets the documented path",
      ac.submit_endpoint("wavespeed", "vendor/brand-new", "image")
      == "https://api.wavespeed.ai/api/v3/vendor/brand-new")
check("the output kind does not change a WaveSpeed endpoint",
      ac.submit_endpoint("wavespeed", "vendor/brand-new", "video")
      == ac.submit_endpoint("wavespeed", "vendor/brand-new", "image"))

check("the result URL from the submit response wins",
      ac.poll_url({"data": {"id": "abc", "urls": {"get": "https://api.wavespeed.ai/custom/abc"}}},
                  "abc", "wavespeed") == "https://api.wavespeed.ai/custom/abc")
check("without one, WaveSpeed polls /predictions/<id>/result",
      ac.poll_url({"data": {"id": "abc"}}, "abc", "wavespeed")
      == "https://api.wavespeed.ai/api/v3/predictions/abc/result")
check("Atlas Cloud polls its own prediction endpoint, with no suffix",
      ac.poll_url({"data": {"id": "abc"}}, "abc", "atlascloud")
      == f"{ac.PREDICTION_ENDPOINT}/abc")

try:
    ac.build_headers("", provider="wavespeed")
    check("a missing key is reported under the provider's own name", False)
except ValueError as error:
    check("a missing key is reported under the provider's own name",
          "WaveSpeed" in str(error) and "Atlas" not in str(error), str(error))

wavespeed_calls = []


def fake_wavespeed_request_json(method, url, headers, timeout, label="Atlas Cloud", **kwargs):
    wavespeed_calls.append((method, url, kwargs.get("json"), label))
    if url == ac.WAVESPEED_UPLOAD_ENDPOINT:
        return {"code": 200, "data": {"type": "image",
                                      "download_url": "https://cdn.wavespeed.ai/up_1.png"}}
    if url.endswith("/result"):
        return {"code": 200, "data": {"status": "completed",
                                      "outputs": ["https://cdn.wavespeed.ai/out.png"]}}
    return {"code": 200, "data": {"id": "ws-pred-1", "status": "created",
                                  "urls": {"get": f"{ac.WAVESPEED_PREDICTION_ENDPOINT}/ws-pred-1/result"}}}


ac.request_json = fake_wavespeed_request_json
ac.schema_for_model = lambda model_identifier, **kwargs: {
    "model": model_identifier,
    "prompt_field": "prompt",
    "image_fields": [{"name": "images", "is_list": True, "max_items": 4, "kind": "image"}],
    "fields": [{"name": "size", "type": "string", "default": None, "enum": None,
                "required": False, "description": "", "is_media": False, "is_list": False}],
}
ac._upload_cache.clear()

wavespeed_url = ac.upload_media(
    "ws-key", b"png-bytes", "ref.png", "image/png", 30, provider="wavespeed"
)
check("an upload goes to the WaveSpeed binary endpoint",
      wavespeed_calls[-1][1] == ac.WAVESPEED_UPLOAD_ENDPOINT, str(wavespeed_calls[-1]))
check("the WaveSpeed upload answer is read from data.download_url",
      wavespeed_url == "https://cdn.wavespeed.ai/up_1.png", wavespeed_url)
check("request errors are reported under the provider's own name",
      wavespeed_calls[-1][3] == "WaveSpeed", str(wavespeed_calls[-1]))

# The same image uploaded to both providers lives on two different hosts, so a
# shared cache entry would hand a WaveSpeed request an Atlas Cloud URL it
# cannot read.
ac._upload_cache[("atlascloud", hashlib.sha256(b"same-image").hexdigest())] = (
    "https://cdn.atlascloud.test/other.png", time.time(),
)
uploads_before = len(wavespeed_calls)
cached_url = ac.upload_image_bytes_cached("ws-key", b"same-image", 30, 120, "T",
                                          provider="wavespeed")
check("the upload cache never serves one provider's URL to the other",
      cached_url == "https://cdn.wavespeed.ai/up_1.png"
      and len(wavespeed_calls) == uploads_before + 1, cached_url)

wavespeed_calls.clear()
wavespeed_node = ac.AtlasCloudImage()
frames, urls, info = wavespeed_node.generate(
    api_key="ws-key", output_type="Image", input_type="Any",
    model="wavespeed-ai/flux-2-pro/text-to-image", prompt="a cat",
    width=1280, height=720, num_images=1, seed=7, poll_interval=0.0, timeout=30,
    image_field="", image_transport="upload", upload_cache_minutes=0,
    extra_params="", params_json="{}", input_map="", max_parallel=1,
    provider="wavespeed", image=image_of(9),
)
submit_call = next(call for call in wavespeed_calls if call[0] == "POST" and call[2] is not None)
check("the request goes to the model's own WaveSpeed URL",
      submit_call[1] == "https://api.wavespeed.ai/api/v3/wavespeed-ai/flux-2-pro/text-to-image",
      submit_call[1])
check("the WaveSpeed request body never repeats the model id",
      "model" not in submit_call[2], str(submit_call[2]))
check("prompt, seed and the derived size reach the request",
      submit_call[2]["prompt"] == "a cat" and submit_call[2]["seed"] == 7
      and submit_call[2]["size"] == "1280*720", str(submit_call[2]))
check("the reference image is uploaded to WaveSpeed and routed by the schema",
      submit_call[2]["images"] == ["https://cdn.wavespeed.ai/up_1.png"], str(submit_call[2]))
check("polling follows the result URL the submit response handed back",
      any(call[0] == "GET" and call[1].endswith("/predictions/ws-pred-1/result")
          for call in wavespeed_calls), str(wavespeed_calls))
check("info names the provider that actually served the request",
      "provider=wavespeed" in info and "model=wavespeed-ai/flux-2-pro/text-to-image" in info, info)
check("the generated image comes back as a batch",
      frames.shape[0] == 1 and urls == "https://cdn.wavespeed.ai/out.png", str(frames.shape))

print()
print(f"{'FAILED: ' + ', '.join(failures) if failures else 'ALL TESTS PASSED'}")
sys.exit(1 if failures else 0)

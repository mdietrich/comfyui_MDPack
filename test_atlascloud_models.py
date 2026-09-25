"""Tests for the Atlas Cloud model catalogue and schema handling.

Run remotely:
    ~/ComfyUI_winows_portable/.venv/bin/python test_atlascloud_models.py
"""

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlascloud_models as am  # noqa: E402

# Captured before any monkeypatching below replaces these with fakes; the live
# end-to-end check at the very end of this file restores them to reach the
# real Atlas Cloud network instead of the stubs the rest of the suite uses.
REAL_FETCH_CATALOGUE = am.fetch_catalogue
REAL_FETCH_SCHEMA = am.fetch_schema
REAL_CACHE_DIRECTORY = am.cache_directory
REAL_REQUESTS_GET = am.requests.get

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"PASS {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name} {detail}")


SAMPLE_CATALOGUE = [
    {"model": "bytedance/seedream-v5.0-pro/text-to-image", "type": "Image",
     "displayName": "Seedream v5.0 Pro", "categories": ["TEXT-TO-IMAGE"],
     "schema": "https://static.test/seedream-t2i.json",
     "price": {"actual": {"base_price": "0.036"}}},
    {"model": "bytedance/seedream-v5.0-pro/edit", "type": "Image",
     "displayName": "Seedream v5.0 Pro Edit", "categories": ["IMAGE-TO-IMAGE"],
     "schema": "https://static.test/seedream-edit.json",
     "price": {"actual": {"base_price": "0.036"}}},
    {"model": "kwaivgi/kling-v2.5-turbo-pro/image-to-video", "type": "Video",
     "displayName": "Kling v2.5 Turbo Pro", "categories": ["IMAGE-TO-VIDEO"],
     "schema": "https://static.test/kling-i2v.json", "price": {}},
    {"model": "xai/grok-4.5", "type": "Text", "displayName": "Grok 4.5",
     "categories": ["LLM"], "schema": "https://static.test/grok.json", "price": {}},
]

# --- catalogue fetch and cache ----------------------------------------------
fetch_calls = []


def fake_fetch(timeout=30.0, provider=am.DEFAULT_PROVIDER, api_key=""):
    fetch_calls.append(timeout)
    return list(SAMPLE_CATALOGUE)


TEMP_CACHE = tempfile.mkdtemp(prefix="mdpack_atlas_test_")
am.cache_directory = lambda: TEMP_CACHE
am.fetch_catalogue = fake_fetch

first = am.load_catalogue()
check("catalogue loaded from API on first call", len(first) == 4 and len(fetch_calls) == 1)

second = am.load_catalogue()
check("catalogue served from cache on second call", len(second) == 4 and len(fetch_calls) == 1)

third = am.load_catalogue(force_refresh=True)
check("force_refresh bypasses the cache", len(third) == 4 and len(fetch_calls) == 2)

fourth = am.load_catalogue(max_age_seconds=0.0)
check("expired cache triggers a refetch", len(fourth) == 4 and len(fetch_calls) == 3)

check("cache file written", os.path.isfile(os.path.join(TEMP_CACHE, "models.json")))

# A cache file can hold syntactically valid JSON that is not a dict (race,
# format change, manual edit, ...); that must not crash load_catalogue.
with open(os.path.join(TEMP_CACHE, "models.json"), "w", encoding="utf-8") as corrupt_cache_file:
    json.dump([], corrupt_cache_file)

calls_before_non_dict_cache = len(fetch_calls)
recovered = am.load_catalogue()
check(
    "non-dict cache content is treated as no cache and triggers a refetch",
    len(recovered) == 4 and len(fetch_calls) == calls_before_non_dict_cache + 1,
)


def failing_fetch(timeout=30.0):
    raise RuntimeError("network down")


am.fetch_catalogue = failing_fetch
stale = am.load_catalogue(max_age_seconds=0.0)
check("stale cache is served when the refetch fails", len(stale) == 4)
am.fetch_catalogue = fake_fetch

# --- category parsing -------------------------------------------------------
check("category IMAGE-TO-IMAGE splits", am.split_category("IMAGE-TO-IMAGE") == ("Image", "Image"))
check("category TEXT-TO-VIDEO splits", am.split_category("TEXT-TO-VIDEO") == ("Text", "Video"))
check("category TEXT-TO-SPEECH maps speech to audio",
      am.split_category("TEXT-TO-SPEECH") == ("Text", "Audio"))
check("category SPEECH-TO-TEXT maps speech to audio",
      am.split_category("SPEECH-TO-TEXT") == ("Audio", "Text"))
check("category IMAGE-TO-3D splits", am.split_category("IMAGE-TO-3D") == ("Image", "3D"))
check("category IMAGE-TOOLS treated as image to image",
      am.split_category("IMAGE-TOOLS") == ("Image", "Image"))
check("category LLM treated as text to text", am.split_category("LLM") == ("Text", "Text"))
check("unknown category yields empty kinds", am.split_category("NONSENSE") == ("", ""))

normalised = am.normalise_entry(SAMPLE_CATALOGUE[1])
check("normalised entry keeps the model id",
      normalised["model"] == "bytedance/seedream-v5.0-pro/edit")
check("normalised entry carries the display name",
      normalised["display_name"] == "Seedream v5.0 Pro Edit")
check("normalised entry carries both kinds",
      normalised["input_kind"] == "Image" and normalised["output_kind"] == "Image")
check("normalised entry carries the schema url",
      normalised["schema_url"] == "https://static.test/seedream-edit.json")
check("normalised entry carries the price", normalised["price"] == "0.036")

# --- filtering --------------------------------------------------------------
image_models = am.list_models(output_kind="Image")
check("filter by output kind", [m["model"] for m in image_models] ==
      ["bytedance/seedream-v5.0-pro/text-to-image", "bytedance/seedream-v5.0-pro/edit"],
      str([m["model"] for m in image_models]))
check("filter result sorted by display name",
      [m["display_name"] for m in image_models] == ["Seedream v5.0 Pro", "Seedream v5.0 Pro Edit"],
      str([m["display_name"] for m in image_models]))

edit_models = am.list_models(output_kind="Image", input_kind="Image")
check("filter by both kinds", [m["model"] for m in edit_models] ==
      ["bytedance/seedream-v5.0-pro/edit"])

video_models = am.list_models(output_kind="Video", input_kind="Image")
check("filter finds the video model", [m["model"] for m in video_models] ==
      ["kwaivgi/kling-v2.5-turbo-pro/image-to-video"])

check("empty kinds return everything", len(am.list_models()) == 4)
check("unmatched filter returns nothing", am.list_models(output_kind="Audio") == [])

# --- classification fallbacks for entries without a categories field -------
# Atlas Cloud leaves `categories` null for some models but still encodes the
# same information elsewhere: a tag that looks like a category (sometimes
# hyphenated, sometimes underscored, sometimes padded with whitespace), or,
# as a last resort, only the coarse `type` field. These shapes are taken
# verbatim from the live catalogue (see task-2-report.md).
reference_tag_entry = am.normalise_entry({
    "model": "vidu/q1/reference-to-video", "type": "Video",
    "displayName": "Vidu Q1 Reference To Video", "categories": None,
    "tags": ["REFERENCE-TO-VIDEO", "NEW"],
    "schema": "https://static.test/vidu-q1.json", "price": {},
})
check("tag-derived hyphenated category classifies via tags",
      (reference_tag_entry["input_kind"], reference_tag_entry["output_kind"]) == ("Image", "Video"))

underscore_tag_entry = am.normalise_entry({
    "model": "ltx-2.3-quality/image-to-video", "type": "Video",
    "displayName": "LTX 2.3 Quality Image To Video", "categories": None,
    "tags": ["IMAGE_TO_VIDEO", "NEW"],
    "schema": "https://static.test/ltx.json", "price": {},
})
check("tag-derived underscored category classifies via tags",
      (underscore_tag_entry["input_kind"], underscore_tag_entry["output_kind"]) == ("Image", "Video"))

llm_tag_entry = am.normalise_entry({
    "model": "google/gemini-2.5-pro", "type": "Text",
    "displayName": "Gemini 2.5 Pro", "categories": None,
    "tags": [" LLM", " NEW", " HOT"],
    "schema": "https://static.test/gemini.json", "price": {},
})
check("whitespace-padded LLM tag classifies via tags",
      (llm_tag_entry["input_kind"], llm_tag_entry["output_kind"]) == ("Text", "Text"))

type_only_entry_raw = {
    "model": "test/mystery-model/video-only", "type": "Video",
    "displayName": "Mystery Video Model", "categories": None, "tags": [],
    "schema": "https://static.test/mystery.json", "price": {},
}
type_only_entry = am.normalise_entry(type_only_entry_raw)
check("entry with neither categories nor usable tags falls back to type",
      type_only_entry["output_kind"] == "Video" and type_only_entry["input_kind"] == "")

# The type-only fallback must make the model reachable through list_models,
# not just correctly classified by normalise_entry in isolation.
FALLBACK_CATALOGUE = SAMPLE_CATALOGUE + [type_only_entry_raw]
am.fetch_catalogue = (
    lambda timeout=30.0, provider=am.DEFAULT_PROVIDER, api_key="": list(FALLBACK_CATALOGUE)
)
am.cache_directory = lambda: tempfile.mkdtemp(prefix="mdpack_atlas_test_fallback_")

fallback_video_models = am.list_models(output_kind="Video")
check("type-only fallback model is reachable by output kind filter",
      "test/mystery-model/video-only" in [m["model"] for m in fallback_video_models])

fallback_all_models = am.list_models()
check("type-only fallback model is reachable with no filter",
      "test/mystery-model/video-only" in [m["model"] for m in fallback_all_models])

# --- schema fetch and disk cache --------------------------------------------
# Unlike the normalisation tests below (which replace fetch_schema wholesale
# with a fake), these exercise fetch_schema itself by patching requests.get,
# the way fetch_catalogue's own tests patch fetch_catalogue -- proving the
# URL-hashed disk cache and the offline-fallback path actually work.
SCHEMA_TEMP_CACHE = tempfile.mkdtemp(prefix="mdpack_atlas_test_schema_")
am.cache_directory = lambda: SCHEMA_TEMP_CACHE

SAMPLE_SCHEMA_DOCUMENT = {"components": {"schemas": {"Input": {"properties": {
    "prompt": {"type": "string"},
}}}}}
CACHED_SCHEMA_URL = "https://static.test/fetch-schema-cached.json"
UNCACHED_SCHEMA_URL = "https://static.test/fetch-schema-uncached.json"


class FakeSchemaResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def fake_requests_get_success(url, timeout=30.0):
    return FakeSchemaResponse(200, SAMPLE_SCHEMA_DOCUMENT)


am.requests.get = fake_requests_get_success
fetched_document = am.fetch_schema(CACHED_SCHEMA_URL)
check("fetch_schema returns the downloaded document",
      fetched_document == SAMPLE_SCHEMA_DOCUMENT, str(fetched_document))

expected_cache_digest = hashlib.sha256(CACHED_SCHEMA_URL.encode("utf-8")).hexdigest()[:16]
expected_cache_path = os.path.join(SCHEMA_TEMP_CACHE, f"schema_{expected_cache_digest}.json")
check("fetch_schema writes the document to the URL-hashed cache path",
      os.path.isfile(expected_cache_path))
with open(expected_cache_path, "r", encoding="utf-8") as written_cache_file:
    check("cached schema file content matches the downloaded document",
          json.load(written_cache_file) == SAMPLE_SCHEMA_DOCUMENT)


def fake_requests_get_network_failure(url, timeout=30.0):
    raise RuntimeError("network unreachable")


am.requests.get = fake_requests_get_network_failure
fallback_document = am.fetch_schema(CACHED_SCHEMA_URL)
check("fetch_schema falls back to the cached document when the request fails",
      fallback_document == SAMPLE_SCHEMA_DOCUMENT, str(fallback_document))

try:
    am.fetch_schema(UNCACHED_SCHEMA_URL)
    check("fetch_schema raises when the request fails and no cache exists", False)
except RuntimeError as error:
    check("fetch_schema raises when the request fails and no cache exists",
          UNCACHED_SCHEMA_URL in str(error), str(error))

# A disk-write failure must not discard an otherwise successful download: it
# should be logged and swallowed, not conflated with a fetch failure. Point
# cache_directory at a path that is itself a regular file, so the write's
# `open(cache_path, "w")` fails with a `NotADirectoryError` (an OSError).
WRITE_FAILURE_CACHE_STUB = os.path.join(SCHEMA_TEMP_CACHE, "not_a_directory")
with open(WRITE_FAILURE_CACHE_STUB, "w", encoding="utf-8") as stub_file:
    stub_file.write("not a directory")
am.cache_directory = lambda: WRITE_FAILURE_CACHE_STUB
am.requests.get = fake_requests_get_success
document_despite_write_failure = am.fetch_schema(CACHED_SCHEMA_URL)
check("fetch_schema returns the document even when writing the cache fails",
      document_despite_write_failure == SAMPLE_SCHEMA_DOCUMENT,
      str(document_despite_write_failure))
am.cache_directory = lambda: SCHEMA_TEMP_CACHE

# --- schema normalisation ---------------------------------------------------
EDIT_SCHEMA = {
    "components": {"schemas": {"Input": {
        "type": "object",
        "required": ["model", "prompt", "images"],
        "x-order-properties": ["model", "prompt", "images", "size", "output_format", "thinking"],
        "properties": {
            "model": {"type": "string", "default": "bytedance/seedream-v5.0-pro/edit"},
            "enable_base64_output": {"type": "boolean", "default": False},
            "prompt": {"type": "string", "description": "The positive prompt."},
            "size": {"type": "string", "default": "2048*2048",
                     "enum": ["2048*2048", "2304*1728", "1728*2304"],
                     "description": "Output image size in 'WIDTH*HEIGHT' pixels."},
            "output_format": {"type": "string", "default": "jpeg", "enum": ["jpeg", "png"]},
            "images": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
            "thinking": {"type": "string", "default": "enabled", "enum": ["enabled", "disabled"]},
        },
    }}},
}

KLING_SCHEMA = {
    "components": {"schemas": {"Input": {
        "type": "object",
        "required": ["model", "prompt", "image"],
        # "camera_control" is listed but never defined in `properties`, to
        # check that a curated-order entry with no matching field is skipped
        # rather than raising a KeyError.
        "x-order-properties": ["model", "image", "last_image", "prompt", "guidance_scale",
                                "duration", "camera_control"],
        "properties": {
            "model": {"type": "string", "default": "kwaivgi/kling-v2.5-turbo-pro/image-to-video"},
            "duration": {"type": "integer", "default": 5, "enum": [5, 10]},
            "guidance_scale": {"type": "number", "default": 0.5},
            "image": {"type": "string"},
            "last_image": {"type": "string"},
            "negative_prompt": {"type": "string"},
            "prompt": {"type": "string"},
        },
    }}},
}

SCHEMAS_BY_URL = {
    "https://static.test/seedream-edit.json": EDIT_SCHEMA,
    "https://static.test/kling-i2v.json": KLING_SCHEMA,
}
schema_fetches = []


def fake_fetch_schema(schema_url, timeout=30.0):
    schema_fetches.append(schema_url)
    return SCHEMAS_BY_URL[schema_url]


am.fetch_schema = fake_fetch_schema
am.clear_schema_memo()

edit = am.schema_for_model("bytedance/seedream-v5.0-pro/edit")
field_names = [f["name"] for f in edit["fields"]]
check("schema drops model, prompt, media and base64 fields",
      field_names == ["size", "output_format", "thinking"], str(field_names))
check("schema keeps the prompt field name", edit["prompt_field"] == "prompt")
check("schema reports the image list field",
      edit["image_fields"] == [{"name": "images", "is_list": True, "max_items": 10, "kind": "image"}],
      str(edit["image_fields"]))
size_field = edit["fields"][0]
check("schema field carries enum", size_field["enum"] == ["2048*2048", "2304*1728", "1728*2304"])
check("schema field carries default", size_field["default"] == "2048*2048")
check("schema field carries type", size_field["type"] == "string")
check("schema field not required", size_field["required"] is False)

check("schema memoised per model", len(schema_fetches) == 1)
am.schema_for_model("bytedance/seedream-v5.0-pro/edit")
check("second lookup served from memo", len(schema_fetches) == 1)

kling = am.schema_for_model("kwaivgi/kling-v2.5-turbo-pro/image-to-video")
check("singular media fields detected in order",
      kling["image_fields"] == [{"name": "image", "is_list": False, "max_items": 1, "kind": "image"},
                                {"name": "last_image", "is_list": False, "max_items": 1, "kind": "image"}],
      str(kling["image_fields"]))
kling_names = [f["name"] for f in kling["fields"]]
check("schema honours x-order-properties and appends unlisted fields last",
      kling_names == ["guidance_scale", "duration", "negative_prompt"], str(kling_names))
check("curated-order entry with no matching property is skipped without error",
      "camera_control" not in kling_names)
duration_field = [f for f in kling["fields"] if f["name"] == "duration"][0]
check("integer field keeps its type and enum",
      duration_field["type"] == "integer" and duration_field["enum"] == [5, 10])

# A field named in MEDIA_FIELD_NAMES only counts as media when its type is
# also media-ish; an "image" field typed as an integer (e.g. a seed-like
# knob that happens to share the name) must surface as a regular field.
NON_MEDIA_TYPED_IMAGE_SCHEMA = {
    "components": {"schemas": {"Input": {
        "type": "object",
        "required": [],
        "properties": {
            "image": {"type": "integer", "description": "Not a media field despite the name."},
        },
    }}},
}
non_media_typed = am.normalise_schema("test/non-media-image", NON_MEDIA_TYPED_IMAGE_SCHEMA)
check("a MEDIA_FIELD_NAMES name with a non-media type is not treated as media",
      non_media_typed["image_fields"] == [], str(non_media_typed["image_fields"]))
check("a MEDIA_FIELD_NAMES name with a non-media type appears in fields instead",
      [f["name"] for f in non_media_typed["fields"]] == ["image"],
      str(non_media_typed["fields"]))

check("unknown model yields an empty schema",
      am.schema_for_model("nope/nope")["fields"] == [])

# --- Fix 4: video/audio media fields carry their own kind, distinct from image
# 25 live video models declare their first (sometimes only) media field as
# `video`/`audio` (e.g. kling-v2.6-pro/avatar, gemini-omni-flash/video-edit);
# both Atlas Cloud nodes only ever have IMAGE inputs, so those fields must
# never be treated the way an `image`/`images` field is.
VIDEO_AUDIO_SCHEMA = {
    "components": {"schemas": {"Input": {
        "type": "object",
        "required": ["model", "prompt"],
        "x-order-properties": ["model", "prompt", "video", "audio", "image"],
        "properties": {
            "model": {"type": "string"},
            "prompt": {"type": "string"},
            "video": {"type": "string"},
            "audio": {"type": "string"},
            "image": {"type": "string"},
        },
    }}},
}
video_audio_normalised = am.normalise_schema("test/video-audio-model", VIDEO_AUDIO_SCHEMA)
video_audio_kinds = {f["name"]: f["kind"] for f in video_audio_normalised["image_fields"]}
check("a video field is tagged with kind 'video'", video_audio_kinds.get("video") == "video",
      str(video_audio_kinds))
check("an audio field is tagged with kind 'audio'", video_audio_kinds.get("audio") == "audio",
      str(video_audio_kinds))
check("an image field is tagged with kind 'image'", video_audio_kinds.get("image") == "image",
      str(video_audio_kinds))
check("video/audio fields still appear in image_fields (shape preserved, kind is additive)",
      {"video", "audio", "image"} == set(video_audio_kinds), str(video_audio_kinds))

check("image_kind_fields drops video/audio fields and keeps the image field",
      [f["name"] for f in am.image_kind_fields(video_audio_normalised)] == ["image"],
      str(am.image_kind_fields(video_audio_normalised)))

VIDEO_ONLY_SCHEMA = {
    "model": "test/video-only-model", "prompt_field": "prompt",
    "image_fields": [{"name": "video", "is_list": False, "max_items": 1, "kind": "video"}],
    "fields": [],
}
check("a schema whose only media field is video-kind has no image-kind fields",
      am.image_kind_fields(VIDEO_ONLY_SCHEMA) == [])

# A schema built before "kind" existed (both hand-written fallback schemas in
# atlascloud_api.py) must still be treated as all-image, not silently dropped.
UNTAGGED_SCHEMA = {
    "model": "test/untagged", "prompt_field": "prompt",
    "image_fields": [{"name": "images", "is_list": True, "max_items": 10}],
    "fields": [],
}
check("a schema with no 'kind' key on its media fields defaults to image (backward compatible)",
      [f["name"] for f in am.image_kind_fields(UNTAGGED_SCHEMA)] == ["images"])

# assign_media must only ever write into the image-kind field, even when a
# video field is declared first (schema order) -- routing an uploaded image
# reference into a `video` field is exactly Finding 4's silent failure mode.
video_first_payload = am.assign_media({}, video_audio_normalised, ["https://a/1.png"])
check("assign_media never writes an image reference into a video/audio field",
      "video" not in video_first_payload and "audio" not in video_first_payload,
      str(video_first_payload))
check("assign_media writes the image reference into the image-kind field instead",
      video_first_payload.get("image") == "https://a/1.png", str(video_first_payload))

# --- Fix 5: a list image field survives being declared next to a video/audio
# field -- 8 live models mix an image list field with a singular video/audio
# field (e.g. wan-2.7/reference-to-video: images (list, max 10) + audio).
MIXED_LIST_AND_VIDEO_SCHEMA = {
    "model": "test/mixed-list-video", "prompt_field": "prompt",
    "image_fields": [
        {"name": "images", "is_list": True, "max_items": 10, "kind": "image"},
        {"name": "audio", "is_list": False, "max_items": 1, "kind": "audio"},
    ],
    "fields": [],
}
check("image_kind_fields on a mixed list/video-audio schema keeps only the list image field",
      am.image_kind_fields(MIXED_LIST_AND_VIDEO_SCHEMA) ==
      [{"name": "images", "is_list": True, "max_items": 10, "kind": "image"}],
      str(am.image_kind_fields(MIXED_LIST_AND_VIDEO_SCHEMA)))
mixed_list_payload = am.assign_media(
    {}, MIXED_LIST_AND_VIDEO_SCHEMA,
    ["https://a/1.png", "https://a/2.png", "https://a/3.png"],
)
check("all three references land in the list image field, none truncated to a video field",
      mixed_list_payload == {"images": ["https://a/1.png", "https://a/2.png", "https://a/3.png"]},
      str(mixed_list_payload))

# --- Fix 6: silent truncation on max_items overflow now prints a diagnostic
SMALL_LIST_SCHEMA = {
    "model": "test/small-list-model", "prompt_field": "prompt",
    "image_fields": [{"name": "images", "is_list": True, "max_items": 2, "kind": "image"}],
    "fields": [],
}
truncation_output = io.StringIO()
with contextlib.redirect_stdout(truncation_output):
    truncated_payload = am.assign_media(
        {}, SMALL_LIST_SCHEMA, ["https://a/1.png", "https://a/2.png", "https://a/3.png"],
        model_identifier="test/small-list-model", label="TestLabel",
    )
check("truncation itself is unchanged: only max_items references are sent",
      truncated_payload == {"images": ["https://a/1.png", "https://a/2.png"]}, str(truncated_payload))
check("truncation prints one diagnostic naming the model, supplied count and sent count",
      "test/small-list-model" in truncation_output.getvalue()
      and "3" in truncation_output.getvalue() and "2" in truncation_output.getvalue()
      and truncation_output.getvalue().count("\n") == 1,
      truncation_output.getvalue())

no_truncation_output = io.StringIO()
with contextlib.redirect_stdout(no_truncation_output):
    am.assign_media({}, SMALL_LIST_SCHEMA, ["https://a/1.png"],
                     model_identifier="test/small-list-model", label="TestLabel")
check("no truncation diagnostic when references fit within max_items",
      no_truncation_output.getvalue() == "", repr(no_truncation_output.getvalue()))

# --- payload building -------------------------------------------------------
edit_schema = am.schema_for_model("bytedance/seedream-v5.0-pro/edit")
kling_schema = am.schema_for_model("kwaivgi/kling-v2.5-turbo-pro/image-to-video")

payload = am.build_payload("bytedance/seedream-v5.0-pro/edit", "a cat",
                           {"size": "2048*2048", "output_format": "png"}, schema=edit_schema)
check("payload carries model and prompt",
      payload["model"] == "bytedance/seedream-v5.0-pro/edit" and payload["prompt"] == "a cat")
check("payload carries schema parameters",
      payload["size"] == "2048*2048" and payload["output_format"] == "png")

sized_output = io.StringIO()
with contextlib.redirect_stdout(sized_output):
    sized = am.build_payload("bytedance/seedream-v5.0-pro/edit", "a cat", {},
                             width=1328, height=1776, schema=edit_schema)
check("width and height become a size string", sized.get("size") == "1328*1776", str(sized))
check("width and height are not sent verbatim",
      "width" not in sized and "height" not in sized)
check("a derived size outside the declared enum is still sent, flagged once",
      sized_output.getvalue().count("\n") == 1
      and "1328*1776" in sized_output.getvalue()
      and "2048*2048" in sized_output.getvalue(),  # one of the declared enum values
      sized_output.getvalue())

explicit_output = io.StringIO()
with contextlib.redirect_stdout(explicit_output):
    explicit = am.build_payload("bytedance/seedream-v5.0-pro/edit", "a cat",
                                {"size": "2304*1728"}, width=1328, height=1776, schema=edit_schema)
check("explicit size still wins over width and height", explicit["size"] == "2304*1728")
# This used to assert the collision printed *nothing* -- that silence is
# Finding 1 from the final whole-branch review: the web extension writes a
# schema default into params_json before the user ever touches it, so this
# exact "width/height + an already-present size" shape is what made a user's
# requested width/height vanish with zero diagnostic the moment a model was
# selected. Precedence (explicit params_json wins) is unchanged and correct;
# only the silence is fixed, by naming the collision instead of eating it.
check("an explicit size already in the payload is flagged instead of silently winning",
      "2304*1728" in explicit_output.getvalue() and "1328" in explicit_output.getvalue()
      and "1776" in explicit_output.getvalue()
      and "bytedance/seedream-v5.0-pro/edit" in explicit_output.getvalue()
      and explicit_output.getvalue().count("\n") == 1,
      explicit_output.getvalue())

dropped = am.build_payload("kwaivgi/kling-v2.5-turbo-pro/image-to-video", "a cat", {},
                           width=1328, height=1776, schema=kling_schema)
check("width and height dropped when the schema has neither size nor width",
      "size" not in dropped and "width" not in dropped)

# The rarer sibling of the size collision above: a schema that takes raw
# width/height keys (rather than a "size" string) whose parameters already
# carry values of their own. No live model currently declares a bare "width"
# field (see the final review's methodology notes), but build_payload must
# not silently drop the node's width/height here either.
WIDTH_FIELD_SCHEMA = {
    "model": "test/width-field-model", "prompt_field": "prompt", "image_fields": [],
    "fields": [
        {"name": "width", "type": "integer", "default": None, "enum": None,
         "required": False, "description": "", "is_media": False, "is_list": False},
        {"name": "height", "type": "integer", "default": None, "enum": None,
         "required": False, "description": "", "is_media": False, "is_list": False},
    ],
}
width_collision_output = io.StringIO()
with contextlib.redirect_stdout(width_collision_output):
    width_collision_payload = am.build_payload(
        "test/width-field-model", "a cat", {"width": 512, "height": 512},
        width=1328, height=1776, schema=WIDTH_FIELD_SCHEMA)
check("explicit width/height from parameters still win over the node's own width/height",
      width_collision_payload.get("width") == 512 and width_collision_payload.get("height") == 512,
      str(width_collision_payload))
check("a width/height collision on a width-style schema is flagged, not silent",
      "512" in width_collision_output.getvalue() and "1328" in width_collision_output.getvalue()
      and "1776" in width_collision_output.getvalue()
      and "test/width-field-model" in width_collision_output.getvalue(),
      width_collision_output.getvalue())

skipped = am.build_payload("bytedance/seedream-v5.0-pro/edit", "a cat",
                           {"size": "", "output_format": None, "thinking": "disabled"},
                           schema=edit_schema)
check("empty parameter values are skipped",
      "size" not in skipped and "output_format" not in skipped and skipped["thinking"] == "disabled")

known_key_output = io.StringIO()
with contextlib.redirect_stdout(known_key_output):
    am.build_payload("bytedance/seedream-v5.0-pro/edit", "a cat",
                     {"size": "2048*2048"}, schema=edit_schema)
check("a known key prints nothing about unknown keys",
      "does not declare" not in known_key_output.getvalue(), known_key_output.getvalue())

unknown_key_output = io.StringIO()
with contextlib.redirect_stdout(unknown_key_output):
    unknown_key_payload = am.build_payload(
        "bytedance/seedream-v5.0-pro/edit", "a cat",
        {"size": "2048*2048", "sharpness": 5}, schema=edit_schema)
check("an unknown key is still written to the payload", unknown_key_payload["sharpness"] == 5)
check("an unknown key prints one line naming the model and the key",
      unknown_key_output.getvalue().count("\n") == 1
      and "bytedance/seedream-v5.0-pro/edit" in unknown_key_output.getvalue()
      and "sharpness" in unknown_key_output.getvalue(),
      unknown_key_output.getvalue())

multiple_unknown_output = io.StringIO()
with contextlib.redirect_stdout(multiple_unknown_output):
    am.build_payload("bytedance/seedream-v5.0-pro/edit", "a cat",
                     {"sharpness": 5, "grain": 1}, schema=edit_schema)
check("multiple unknown keys print a single line, not one per key",
      multiple_unknown_output.getvalue().count("\n") == 1, multiple_unknown_output.getvalue())

# A schema that declares no prompt field at all (prompt_field == ""), the way
# normalise_schema reports it for a model with no text input.
NO_PROMPT_SCHEMA = {"model": "no-prompt/model", "prompt_field": "", "image_fields": [], "fields": []}

no_prompt_output = io.StringIO()
with contextlib.redirect_stdout(no_prompt_output):
    no_prompt_payload = am.build_payload("no-prompt/model", "a cat", {}, schema=NO_PROMPT_SCHEMA)
check("a schema with no prompt field never gets a prompt key",
      "prompt" not in no_prompt_payload, str(no_prompt_payload))
check("dropping a supplied prompt prints one diagnostic line naming the model",
      no_prompt_output.getvalue().count("\n") == 1 and "no-prompt/model" in no_prompt_output.getvalue(),
      no_prompt_output.getvalue())

no_prompt_supplied_output = io.StringIO()
with contextlib.redirect_stdout(no_prompt_supplied_output):
    am.build_payload("no-prompt/model", "", {}, schema=NO_PROMPT_SCHEMA)
check("no diagnostic when no prompt was supplied in the first place",
      no_prompt_supplied_output.getvalue() == "", no_prompt_supplied_output.getvalue())

# --- media assignment -------------------------------------------------------
list_payload = am.assign_media({}, edit_schema, ["https://a/1.png", "https://a/2.png"])
check("list media field receives every url",
      list_payload["images"] == ["https://a/1.png", "https://a/2.png"], str(list_payload))

single_payload = am.assign_media({}, kling_schema, ["https://a/1.png", "https://a/2.png"])
check("singular media fields are filled in order",
      single_payload == {"image": "https://a/1.png", "last_image": "https://a/2.png"},
      str(single_payload))

mapped_payload = am.assign_media({}, kling_schema, ["https://a/1.png", "https://a/2.png"],
                                 input_map={"last_image": 0, "image": 1})
check("input_map overrides the default order",
      mapped_payload == {"last_image": "https://a/1.png", "image": "https://a/2.png"},
      str(mapped_payload))

list_mapped_payload = am.assign_media({}, edit_schema, ["https://a/1.png", "https://a/2.png"],
                                      input_map={"images": 1})
check("input_map on a list field wraps the selected url in a one-element list",
      list_mapped_payload == {"images": ["https://a/2.png"]}, str(list_mapped_payload))

check("no media fields leaves the payload untouched",
      am.assign_media({"model": "m"}, {"image_fields": []}, ["https://a/1.png"]) == {"model": "m"})

# --- image count field: the name a model actually takes for "how many" -------
# Only 20 of the 119 live image models call it "num_images"; 7 call it "n"
# (wan-2.7, gpt-image-1) and 6 "max_images" (the Seedream /sequential
# variants). The node's own num_images widget must therefore be routed to the
# name the selected model declares instead of being sent under a name the API
# ignores -- which is what made a num_images=2 run return a single image.


def count_schema(*names):
    return {"fields": [{"name": name, "type": "integer", "default": None, "enum": None,
                        "required": False, "description": "", "is_media": False,
                        "is_list": False} for name in names]}


check("image_count_field finds a declared num_images",
      am.image_count_field(count_schema("size", "num_images")) == "num_images")
check("image_count_field finds a declared n",
      am.image_count_field(count_schema("size", "n")) == "n")
check("image_count_field finds a declared max_images",
      am.image_count_field(count_schema("max_images", "size")) == "max_images")
check("image_count_field returns None when the model declares no count field",
      am.image_count_field(count_schema("size", "output_format", "thinking")) is None)
check("image_count_field prefers num_images when a schema declares several",
      am.image_count_field(count_schema("n", "num_images")) == "num_images")
check("image_count_field tolerates a schema without fields",
      am.image_count_field({}) is None)

# --- WaveSpeed: catalogue normalisation --------------------------------------
# WaveSpeed describes a model with model_id/name/type plus the request schema
# inline, and its `type` is a single label that only sometimes reads
# INPUT-TO-OUTPUT. The entries below are trimmed copies of live catalogue
# entries, one per shape that needs its own rule.

WAVESPEED_CATALOGUE = [
    {
        "model_id": "wavespeed-ai/flux-2-pro/text-to-image",
        "name": "wavespeed-ai/flux-2-pro/text-to-image",
        "base_price": 0.03,
        "type": "text-to-image",
        "api_schema": {"api_schemas": [{
            "type": "model_run",
            "method": "POST",
            "server": "https://api.wavespeed.ai",
            "api_path": "/api/v3/wavespeed-ai/flux-2-pro/text-to-image",
            "request_schema": {
                "additionalProperties": False,
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The positive prompt."},
                    "size": {"type": "string", "default": "1024*1024",
                             "enum": ["1024*1024", "1280*720"]},
                    "num_images": {"type": "integer", "default": 1},
                    "seed": {"type": "integer", "default": -1},
                    "enable_sync_mode": {"type": "boolean", "default": False, "disabled": True},
                    "enable_base64_output": {"type": "boolean", "default": False,
                                             "disabled": True},
                },
                "required": ["prompt"],
                "x-order-properties": ["prompt", "size", "num_images", "seed"],
            },
        }]},
    },
    {
        "model_id": "wavespeed-ai/minimax-h3/reference-to-video",
        "name": "MiniMax H3 reference-to-video",
        "base_price": 0.25,
        "type": "reference-to-video",
        "api_schema": {"api_schemas": [{
            "type": "model_run",
            "api_path": "/api/v3/wavespeed-ai/minimax-h3/reference-to-video",
            "request_schema": {
                "properties": {
                    "prompt": {"type": "string"},
                    "reference_images": {
                        "type": "array", "items": {"type": "string"}, "maxItems": 9,
                        "x-ui-component": "uploaders",
                        "x-ui-component-props": {"accept": "image/*"},
                    },
                    "reference_videos": {
                        "type": "array", "items": {"type": "string"}, "maxItems": 3,
                        "x-ui-component": "uploaders",
                        "x-ui-component-props": {"accept": "video/*"},
                    },
                    "reference_audios": {
                        "type": "array", "items": {"type": "string"}, "maxItems": 3,
                        "x-ui-component": "uploaders",
                        "x-ui-component-props": {"accept": "audio/*"},
                    },
                    "resolution": {"type": "string", "default": "480p",
                                   "enum": ["480p", "768p"]},
                },
                "required": ["prompt"],
                "x-order-properties": ["prompt", "reference_images", "reference_videos",
                                       "reference_audios", "resolution"],
            },
        }]},
    },
    {
        "model_id": "wavespeed-ai/image-upscaler",
        "name": "Image Upscaler",
        "base_price": 0.01,
        "type": "upscaler",
        "api_schema": {"api_schemas": [{
            "type": "model_run",
            "api_path": "/api/v3/wavespeed-ai/image-upscaler",
            "request_schema": {
                "properties": {
                    "image": {"type": "string", "x-ui-component": "uploader",
                              "x-ui-component-props": {"accept": "image/*"}},
                    "target_resolution": {"type": "string", "default": "2k"},
                },
                "required": ["image"],
            },
        }]},
    },
    {
        "model_id": "wavespeed-ai/video-upscaler",
        "name": "Video Upscaler",
        "base_price": 0.02,
        "type": "upscaler",
        "api_schema": {"api_schemas": [{
            "type": "model_run",
            "api_path": "/api/v3/wavespeed-ai/video-upscaler",
            "request_schema": {
                "properties": {
                    "video": {"type": "string", "x-ui-component": "uploader",
                              "x-ui-component-props": {"accept": "video/*"}},
                },
                "required": ["video"],
            },
        }]},
    },
    {
        "model_id": "wavespeed-ai/z-image-lora-trainer",
        "name": "Z-Image LoRA Trainer",
        "base_price": 2.0,
        "type": "training",
        "api_schema": {"api_schemas": [{
            "type": "model_run",
            "api_path": "/api/v3/wavespeed-ai/z-image-lora-trainer",
            "request_schema": {
                "properties": {
                    "data": {"type": "string", "x-ui-component": "uploader",
                             "x-accept": "application/zip",
                             "description": "URL to zip archive with images."},
                    "steps": {"type": "integer", "default": 1000},
                },
                "required": ["data"],
            },
        }]},
    },
]

wavespeed_fetches = []


def fake_wavespeed_fetch(timeout=30.0, provider=am.DEFAULT_PROVIDER, api_key=""):
    if provider == "wavespeed":
        wavespeed_fetches.append(api_key)
        return list(WAVESPEED_CATALOGUE)
    return list(SAMPLE_CATALOGUE)


am.fetch_catalogue = fake_wavespeed_fetch
am.cache_directory = lambda: tempfile.mkdtemp(prefix="mdpack_wavespeed_test_")
am.clear_schema_memo()

wavespeed_models = am.list_models(provider="wavespeed", api_key="test-key")
by_id = {entry["model"]: entry for entry in wavespeed_models}

check("the WaveSpeed catalogue is read from model_id",
      set(by_id) == {entry["model_id"] for entry in WAVESPEED_CATALOGUE}, str(sorted(by_id)))
check("a text-to-image type maps to Text -> Image",
      (by_id["wavespeed-ai/flux-2-pro/text-to-image"]["input_kind"],
       by_id["wavespeed-ai/flux-2-pro/text-to-image"]["output_kind"]) == ("Text", "Image"))
check("reference-to-video maps a 'reference' input to Image",
      (by_id["wavespeed-ai/minimax-h3/reference-to-video"]["input_kind"],
       by_id["wavespeed-ai/minimax-h3/reference-to-video"]["output_kind"]) == ("Image", "Video"))
check("the 'upscaler' type is split by the model id: image stays Image",
      by_id["wavespeed-ai/image-upscaler"]["output_kind"] == "Image")
check("the 'upscaler' type is split by the model id: video becomes Video",
      by_id["wavespeed-ai/video-upscaler"]["output_kind"] == "Video")
check("a trainer returns a file, not the image its id names",
      by_id["wavespeed-ai/z-image-lora-trainer"]["output_kind"] == "Text",
      by_id["wavespeed-ai/z-image-lora-trainer"]["output_kind"])
check("the upscaler input kind comes from the uploader field it declares",
      (by_id["wavespeed-ai/image-upscaler"]["input_kind"],
       by_id["wavespeed-ai/video-upscaler"]["input_kind"]) == ("Image", "Video"))
check("base_price is carried over as the price label",
      by_id["wavespeed-ai/flux-2-pro/text-to-image"]["price"] == "0.03")
check("the catalogue request carries the API key",
      wavespeed_fetches and wavespeed_fetches[-1] == "test-key", str(wavespeed_fetches))
check("a Video filter keeps exactly the two video models",
      {entry["model"] for entry in am.list_models(output_kind="Video", provider="wavespeed",
                                                  api_key="test-key")}
      == {"wavespeed-ai/minimax-h3/reference-to-video", "wavespeed-ai/video-upscaler"})

# --- WaveSpeed: schema normalisation -----------------------------------------
flux_schema = am.schema_for_model("wavespeed-ai/flux-2-pro/text-to-image",
                                  provider="wavespeed", api_key="test-key")
flux_field_names = [field["name"] for field in flux_schema["fields"]]
check("the inline request schema is read without a second fetch",
      flux_schema["prompt_field"] == "prompt" and "size" in flux_field_names,
      str(flux_field_names))
check("x-order-properties decides the widget order",
      flux_field_names == ["size", "num_images", "seed"], str(flux_field_names))
check("fields WaveSpeed marks as disabled never become widgets",
      "enable_sync_mode" not in flux_field_names and
      "enable_base64_output" not in flux_field_names, str(flux_field_names))
check("the size enum survives normalisation",
      flux_schema["fields"][0]["enum"] == ["1024*1024", "1280*720"])
check("a text-to-image model declares no media field",
      flux_schema["image_fields"] == [], str(flux_schema["image_fields"]))

minimax_schema = am.schema_for_model("wavespeed-ai/minimax-h3/reference-to-video",
                                     provider="wavespeed", api_key="test-key")
media_by_name = {field["name"]: field for field in minimax_schema["image_fields"]}
check("an uploader field with accept=image/* is an image slot",
      media_by_name["reference_images"]["kind"] == "image"
      and media_by_name["reference_images"]["is_list"] is True
      and media_by_name["reference_images"]["max_items"] == 9, str(media_by_name))
check("accept=video/* and accept=audio/* keep their own kinds",
      (media_by_name["reference_videos"]["kind"], media_by_name["reference_audios"]["kind"])
      == ("video", "audio"))
check("only the image-kind field is offered as an image slot",
      [field["name"] for field in am.image_kind_fields(minimax_schema)] == ["reference_images"])
check("a media field never doubles as a parameter widget",
      "reference_images" not in [field["name"] for field in minimax_schema["fields"]])

trainer_schema = am.schema_for_model("wavespeed-ai/z-image-lora-trainer",
                                     provider="wavespeed", api_key="test-key")
check("an uploader that takes a zip is not mistaken for an image slot",
      trainer_schema["image_fields"] == [], str(trainer_schema["image_fields"]))
check("that zip field stays an ordinary widget the user can paste a URL into",
      "data" in [field["name"] for field in trainer_schema["fields"]],
      str([field["name"] for field in trainer_schema["fields"]]))

check("submit_path comes from the model's own api_schema",
      am.submit_path("wavespeed-ai/flux-2-pro/text-to-image", provider="wavespeed",
                     api_key="test-key")
      == "/api/v3/wavespeed-ai/flux-2-pro/text-to-image")
check("submit_path is empty for Atlas Cloud, which names the model in the body",
      am.submit_path("bytedance/seedream-v5.0-pro/text-to-image") == "")
check("an unknown WaveSpeed model yields an empty schema rather than raising",
      am.schema_for_model("nobody/nothing", provider="wavespeed", api_key="test-key")
      == {"model": "nobody/nothing", "prompt_field": "", "image_fields": [], "fields": []})

# --- WaveSpeed: payload building ---------------------------------------------
wavespeed_payload = am.build_payload(
    "wavespeed-ai/flux-2-pro/text-to-image", "a cat", {"num_images": 2},
    width=1280, height=720, schema=flux_schema, provider="wavespeed",
)
check("the model id is not repeated in a WaveSpeed request body",
      "model" not in wavespeed_payload, str(wavespeed_payload))
check("prompt, parameters and the derived size are all sent",
      wavespeed_payload == {"prompt": "a cat", "num_images": 2, "size": "1280*720"},
      str(wavespeed_payload))
atlas_payload = am.build_payload("bytedance/seedream-v5.0-pro/text-to-image", "a cat", {},
                                 schema={"prompt_field": "prompt", "fields": []})
check("Atlas Cloud still carries the model in the body",
      atlas_payload["model"] == "bytedance/seedream-v5.0-pro/text-to-image")

# --- WaveSpeed: provider isolation -------------------------------------------
try:
    # A typo must not silently send a WaveSpeed model id to Atlas Cloud.
    am.provider_config("wavespeeed")
    check("an unknown provider name raises instead of defaulting", False)
except ValueError:
    check("an unknown provider name raises instead of defaulting", True)
check("an empty provider name means Atlas Cloud",
      am.provider_config("")[0] == "atlascloud")
check("the two providers use different catalogue cache files",
      am._cache_path("atlascloud") != am._cache_path("wavespeed"))

# The same model id may exist at both providers with different fields; the
# schema memo must not hand one provider's schema to the other.
am.clear_schema_memo()
same_id = "shared/model-id"
am._schema_memo[("wavespeed", same_id)] = {"model": same_id, "prompt_field": "wavespeed_marker",
                                           "image_fields": [], "fields": []}
check("the schema memo is keyed by provider, not by model id alone",
      am.schema_for_model(same_id, provider="wavespeed")["prompt_field"] == "wavespeed_marker"
      and am.schema_for_model(same_id)["prompt_field"] == "")
am.clear_schema_memo()

am.fetch_catalogue = fake_fetch
am.cache_directory = lambda: TEMP_CACHE

# --- live schema end-to-end check --------------------------------------------
# This is the bug the whole plan exists to fix: a user requesting 1328x1776
# from the real Seedream text-to-image model got 1584x2816 back, because the
# node sent width/height while the schema declares a "WIDTH*HEIGHT" size
# string. Every other check above stubs the network; this one deliberately
# restores the real fetch_catalogue/fetch_schema/requests.get so the fix is
# proven against the schema Atlas Cloud actually serves, not a fake.
am.fetch_catalogue = REAL_FETCH_CATALOGUE
am.fetch_schema = REAL_FETCH_SCHEMA
am.cache_directory = lambda: tempfile.mkdtemp(prefix="mdpack_atlas_test_live_")
am.requests.get = REAL_REQUESTS_GET
am.clear_schema_memo()
try:
    live_schema = am.schema_for_model("bytedance/seedream-v5.0-pro/text-to-image")
    live_payload = am.build_payload("bytedance/seedream-v5.0-pro/text-to-image", "a cat", {},
                                    width=1328, height=1776, schema=live_schema)
    check("live Seedream schema turns width/height into a size string",
          live_payload.get("size") == "1328*1776"
          and "width" not in live_payload and "height" not in live_payload,
          str(live_payload))
except Exception as error:
    check(f"live schema check raised instead of returning a payload ({error})", False)

try:
    check("the live Seedream v5.0 Pro edit schema declares no image count field at all",
          am.image_count_field(am.schema_for_model("bytedance/seedream-v5.0-pro/edit")) is None)
    check("the live Seedream /sequential schema declares max_images",
          am.image_count_field(am.schema_for_model("bytedance/seedream-v4.5/sequential")) == "max_images")
except Exception as error:
    check(f"live image_count_field check raised ({error})", False)

# --- live WaveSpeed catalogue -------------------------------------------------
# Runs only where a key is available (WAVESPEED_API_KEY, or ~/.wavespeed_key on
# the deployment host) and is skipped everywhere else, so the suite stays
# runnable without one. It proves the normalisation above against the ~950
# models WaveSpeed actually serves rather than the five trimmed samples.
def live_wavespeed_key():
    key = os.environ.get("WAVESPEED_API_KEY", "").strip()
    if key:
        return key
    try:
        with open(os.path.expanduser("~/.wavespeed_key"), "r", encoding="utf-8") as key_file:
            return key_file.read().strip()
    except OSError:
        return ""


live_key = live_wavespeed_key()
if not live_key:
    print("SKIP live WaveSpeed checks (no WAVESPEED_API_KEY and no ~/.wavespeed_key)")
else:
    am.clear_schema_memo()
    try:
        live_entries = am.list_models(provider="wavespeed", api_key=live_key)
        check("the live WaveSpeed catalogue lists hundreds of models",
              len(live_entries) > 300, str(len(live_entries)))
        check("every live entry carries a model id and an output kind",
              all(entry["model"] and entry["output_kind"] for entry in live_entries))
        live_by_id = {entry["model"]: entry for entry in live_entries}
        known = "wavespeed-ai/flux-2-pro/text-to-image"
        if known in live_by_id:
            live_schema = am.schema_for_model(known, provider="wavespeed", api_key=live_key)
            live_payload = am.build_payload(known, "a cat", {}, width=1280, height=720,
                                            schema=live_schema, provider="wavespeed")
            check(f"the live {known} schema declares a prompt field",
                  live_schema["prompt_field"] == "prompt")
            check("a live WaveSpeed payload carries no model key",
                  "model" not in live_payload, str(live_payload))
            check("width/height become the size string the live schema declares",
                  live_payload.get("size") == "1280*720", str(live_payload))
            check("the live submit path names the model",
                  am.submit_path(known, provider="wavespeed", api_key=live_key)
                  == f"/api/v3/{known}")
        else:
            print(f"SKIP live schema checks ({known} is no longer catalogued)")
        # Every image slot the nodes can fill must be an image-kind field; a
        # video/audio uploader landing in that list would let the node upload a
        # PNG into a field that only takes a video.
        mixed = am.schema_for_model("wavespeed-ai/minimax-h3/reference-to-video",
                                    provider="wavespeed", api_key=live_key)
        if mixed["image_fields"]:
            check("a live multi-modal model exposes only its image field as a slot",
                  [field["name"] for field in am.image_kind_fields(mixed)] == ["reference_images"],
                  str([(f["name"], f.get("kind")) for f in mixed["image_fields"]]))
    except Exception as error:
        check(f"live WaveSpeed check raised ({error})", False)

print()
if FAILURES:
    print(f"{len(FAILURES)} TEST(S) FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("ALL TESTS PASSED")

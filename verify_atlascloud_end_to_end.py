"""End-to-end verification of the Atlas Cloud model catalogue, schema
fetching and request-building logic against the REAL Atlas Cloud network.

This is deliberately not a stubbed unit test: it fetches the live catalogue
from `https://api.atlascloud.ai/api/v1/models` and the live per-model
OpenAPI schemas from their real `static.atlascloud.ai` URLs (both public, no
API key required), then asserts that `atlascloud_models.build_payload()` and
`atlascloud_models.assign_media()` produce request bodies matching each
model's actual, current field names. It never calls the generation endpoint
itself, so it costs nothing and submits no job.

Run:
    python verify_atlascloud_end_to_end.py

Requires network access to atlascloud.ai. A network failure surfaces as a
FAIL for the affected check (with the underlying error in the detail), not a
silent skip, since a check that could not run proves nothing.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlascloud_models as am  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"PASS {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name} {detail}")


# A scratch cache directory keeps this script from reading or writing a real
# ComfyUI installation's cache and guarantees every run starts from a genuine
# network fetch rather than a stale disk cache from a previous run. Created
# once here (not inside the lambda): cache_directory() is called repeatedly
# per run (via _read_cache/_write_cache/_schema_cache_path), and a lambda
# that calls mkdtemp() itself would mint a fresh, never-cleaned-up directory
# on every single call.
_SCRATCH_CACHE_DIRECTORY = tempfile.mkdtemp(prefix="mdpack_atlas_verify_")
am.cache_directory = lambda: _SCRATCH_CACHE_DIRECTORY


def find_entry(entries, model_id):
    for entry in entries:
        if entry["model"] == model_id:
            return entry
    return None


print("=== Fetching the live Atlas Cloud model catalogue ===")
print(f"GET {am.CATALOGUE_URL}")
try:
    raw_catalogue = am.fetch_catalogue()
except Exception as error:
    check("catalogue fetch succeeded", False, str(error))
    raw_catalogue = []
else:
    check("catalogue fetch succeeded", True)
    print(f"Fetched {len(raw_catalogue)} raw catalogue entries")

entries = [am.normalise_entry(raw) for raw in raw_catalogue]
entries_with_id = [entry for entry in entries if entry["model"]]
print(f"{len(entries_with_id)} of {len(entries)} entries carry a usable model id")

output_kind_counts = {}
unclassified = []
for entry in entries_with_id:
    kind = entry["output_kind"]
    if kind:
        output_kind_counts[kind] = output_kind_counts.get(kind, 0) + 1
    else:
        unclassified.append(entry["model"])

print("Output-kind classification counts:")
for kind in sorted(output_kind_counts):
    print(f"  {kind}: {output_kind_counts[kind]}")
if unclassified:
    preview = unclassified[:10]
    more = "..." if len(unclassified) > 10 else ""
    print(f"  (unclassified): {len(unclassified)} -> {preview}{more}")

check(
    "every catalogued model with a usable id classifies into a known output kind",
    not unclassified,
    f"{len(unclassified)} unclassified model(s): {unclassified[:20]}",
)
check(
    "the catalogue is non-trivially sized (sanity check on the fetch itself)",
    len(entries_with_id) > 50,
    f"only {len(entries_with_id)} entries -- catalogue fetch may be broken or Atlas Cloud changed shape",
)


# --- Test 1: the original bug -----------------------------------------------
# A user asked bytedance/seedream-v5.0-pro/text-to-image for a 1328x1776
# image and received 1584x2816, because the node sent width/height while
# Seedream declares `size` as a "WIDTH*HEIGHT" enum string and ignored the
# unknown width/height keys. This proves the fix against the schema Atlas
# Cloud serves today, not a fixture frozen at fix time.
print("\n=== Test 1: Seedream v5.0 Pro text-to-image size handling (the original bug) ===")
SEEDREAM_T2I = "bytedance/seedream-v5.0-pro/text-to-image"
try:
    entry = find_entry(entries_with_id, SEEDREAM_T2I)
    check(f"{SEEDREAM_T2I} is in the live catalogue", entry is not None, str(entry))
    schema = am.schema_for_model(SEEDREAM_T2I)
    print(f"Live schema fields for {SEEDREAM_T2I}: {[f['name'] for f in schema['fields']]}")
    size_field = am._field_by_name(schema, "size")
    check(
        "schema declares a string 'size' field",
        size_field is not None and size_field["type"] == "string",
        str(size_field),
    )
    payload = am.build_payload(
        SEEDREAM_T2I, "a cat wearing sunglasses", {}, width=1328, height=1776, schema=schema
    )
    print(f"Payload built for width=1328 height=1776: {json.dumps(payload)}")
    check("size == '1328*1776'", payload.get("size") == "1328*1776", str(payload))
    check("no width key sent", "width" not in payload, str(payload))
    check("no height key sent", "height" not in payload, str(payload))
except Exception as error:
    check("Test 1 raised instead of completing", False, str(error))


# --- Test 2: edit model places references into 'images' as a list ----------
print("\n=== Test 2: Seedream v5.0 Pro edit places references into 'images' as a list ===")
SEEDREAM_EDIT = "bytedance/seedream-v5.0-pro/edit"
try:
    entry = find_entry(entries_with_id, SEEDREAM_EDIT)
    check(f"{SEEDREAM_EDIT} is in the live catalogue", entry is not None, str(entry))
    schema = am.schema_for_model(SEEDREAM_EDIT)
    print(f"Live schema image fields for {SEEDREAM_EDIT}: {schema['image_fields']}")
    images_field = next((f for f in schema["image_fields"] if f["name"] == "images"), None)
    check(
        "schema declares an 'images' list field",
        images_field is not None and images_field["is_list"],
        str(schema["image_fields"]),
    )
    reference_urls = ["https://example.com/reference-1.png", "https://example.com/reference-2.png"]
    payload = am.assign_media({"model": SEEDREAM_EDIT}, schema, reference_urls)
    print(f"Payload after assign_media: {json.dumps(payload)}")
    check(
        "'images' holds both references as a list, in order",
        payload.get("images") == reference_urls,
        str(payload),
    )
except Exception as error:
    check("Test 2 raised instead of completing", False, str(error))


# --- Test 3: Kling image-to-video routes 'image'/'last_image', input_map ---
print("\n=== Test 3: Kling v2.5 Turbo Pro routes two references into 'image' and "
      "'last_image', input_map swaps them ===")
KLING_I2V = "kwaivgi/kling-v2.5-turbo-pro/image-to-video"
try:
    entry = find_entry(entries_with_id, KLING_I2V)
    check(f"{KLING_I2V} is in the live catalogue", entry is not None, str(entry))
    schema = am.schema_for_model(KLING_I2V)
    print(f"Live schema image fields for {KLING_I2V}: {schema['image_fields']}")
    field_names = {field["name"] for field in schema["image_fields"]}
    check(
        "schema declares both 'image' and 'last_image'",
        {"image", "last_image"} <= field_names,
        str(schema["image_fields"]),
    )
    reference_urls = ["https://example.com/first-frame.png", "https://example.com/last-frame.png"]

    default_payload = am.assign_media({"model": KLING_I2V}, schema, reference_urls)
    print(f"Default (schema-order) payload: {json.dumps(default_payload)}")
    check(
        "without input_map, both references land in 'image' and 'last_image'",
        {default_payload.get("image"), default_payload.get("last_image")} == set(reference_urls),
        str(default_payload),
    )

    swapped_payload = am.assign_media(
        {"model": KLING_I2V}, schema, reference_urls,
        input_map={"image": 1, "last_image": 0},
    )
    print(f"input_map-swapped payload ({{'image': 1, 'last_image': 0}}): "
          f"{json.dumps(swapped_payload)}")
    check(
        "input_map swaps which reference lands in which field",
        swapped_payload.get("image") == reference_urls[1]
        and swapped_payload.get("last_image") == reference_urls[0],
        str(swapped_payload),
    )
except Exception as error:
    check("Test 3 raised instead of completing", False, str(error))


print()
if FAILURES:
    print(f"{len(FAILURES)} TEST(S) FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("ALL TESTS PASSED")

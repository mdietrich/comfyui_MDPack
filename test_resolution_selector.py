"""Tests for the MDResolutionSelector node.

Run locally (needs the ComfyUI venv for comfy_api):
    /home/mdietrich/ComfyUI_winows_portable/.venv/bin/python test_resolution_selector.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from resolution_selector import ASPECT_RATIOS, AspectRatio, MDResolutionSelector  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"PASS {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name} {detail}")


def size(aspect_ratio, megapixels=1.0, multiple=8):
    result = MDResolutionSelector.execute(aspect_ratio, megapixels, multiple)
    return tuple(result.result)


check("social ratios present",
      AspectRatio.SOCIAL_V in ASPECT_RATIOS and AspectRatio.SOCIAL_H in ASPECT_RATIOS)

check("4:5 maps to (4, 5)", ASPECT_RATIOS[AspectRatio.SOCIAL_V] == (4, 5))
check("5:4 maps to (5, 4)", ASPECT_RATIOS[AspectRatio.SOCIAL_H] == (5, 4))

check("core ratios kept",
      [r.value for r in AspectRatio if r not in (AspectRatio.SOCIAL_V, AspectRatio.SOCIAL_H)] == [
          "1:1 (Square)", "2:3 (Portrait Photo)", "3:2 (Photo)", "3:4 (Portrait Standard)",
          "4:3 (Standard)", "9:16 (Portrait Widescreen)", "16:9 (Widescreen)", "21:9 (Ultrawide)",
      ])

square = size(AspectRatio.SQUARE)
check("1 MP square is 1024x1024", square == (1024, 1024), str(square))

social_v = size(AspectRatio.SOCIAL_V)
check("4:5 is portrait", social_v[0] < social_v[1], str(social_v))
check("4:5 hits the ratio", abs(social_v[0] / social_v[1] - 4 / 5) < 0.01, str(social_v))
check("4:5 hits ~1 MP", abs(social_v[0] * social_v[1] / (1024 * 1024) - 1.0) < 0.02, str(social_v))

social_h = size(AspectRatio.SOCIAL_H)
check("5:4 is the transpose of 4:5", social_h == (social_v[1], social_v[0]), str(social_h))

multiple_64 = size(AspectRatio.SOCIAL_V, multiple=64)
check("multiple is honoured",
      multiple_64[0] % 64 == 0 and multiple_64[1] % 64 == 0, str(multiple_64))

mp_4 = size(AspectRatio.SOCIAL_V, megapixels=4.0)
check("megapixels scale the result",
      abs(mp_4[0] * mp_4[1] / (1024 * 1024) - 4.0) < 0.05, str(mp_4))

schema = MDResolutionSelector.define_schema()
check("node id is namespaced", schema.node_id == "MDResolutionSelector", schema.node_id)
check("outputs are width and height",
      [o.display_name for o in schema.outputs] == ["width", "height"],
      str([o.display_name for o in schema.outputs]))

print()
if FAILURES:
    print(str(len(FAILURES)) + " test(s) failed: " + ", ".join(FAILURES))
    sys.exit(1)
print("all tests passed")

"""Tests for the KeyValueDropdown node.

Run locally:
    python3 test_key_value_dropdown.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from key_value_dropdown import KeyValueDropdown  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"PASS {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name} {detail}")


MAPPING = "hotfitlena = 0\ntatum = 1\nboth = 2"
NODE = KeyValueDropdown()

check("existing outputs keep positions 0 and 1",
      KeyValueDropdown.RETURN_NAMES[:2] == ("value", "key"),
      str(KeyValueDropdown.RETURN_NAMES))

check("mapping is the third output",
      KeyValueDropdown.RETURN_NAMES[2] == "mapping"
      and KeyValueDropdown.RETURN_TYPES[2] == "STRING",
      f"{KeyValueDropdown.RETURN_NAMES} / {KeyValueDropdown.RETURN_TYPES}")

result = NODE.select(MAPPING, "tatum")
check("select returns value, key and the raw mapping",
      result == (1, "tatum", MAPPING), str(result))

check("parse_mapping reads all three pairs",
      KeyValueDropdown.parse_mapping(MAPPING) == {"hotfitlena": 0, "tatum": 1, "both": 2},
      str(KeyValueDropdown.parse_mapping(MAPPING)))

print()
if FAILURES:
    print(f"{len(FAILURES)} TEST(S) FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("ALL TESTS PASSED")

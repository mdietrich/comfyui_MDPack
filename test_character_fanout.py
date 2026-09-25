"""Tests for the CharacterFanout node.

Run locally:
    python3 test_character_fanout.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from character_fanout import CharacterFanout  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"PASS {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name} {detail}")


MAPPING = "hotfitlena = 0\ntatum = 1\nboth = 2"
PROMPTS_A = "prompt a1\nprompt a2"
PROMPTS_B = "prompt b1\nprompt b2"
NODE = CharacterFanout()


def fan_out(mode, primary=PROMPTS_A, secondary=PROMPTS_B):
    return NODE.fan_out(
        mode=mode, mapping=MAPPING,
        prompts_primary=primary, prompts_secondary=secondary,
        trigger_a="triggerA", trigger_b="triggerB",
        lora_a="krea2/a.safetensors", lora_b="krea2/b.safetensors",
    )


check("OUTPUT_IS_LIST marks all four outputs as lists",
      CharacterFanout.OUTPUT_IS_LIST == (True, True, True, True),
      str(CharacterFanout.OUTPUT_IS_LIST))

check("output names are stable",
      CharacterFanout.RETURN_NAMES ==
      ("prompt", "triggerword", "lora_name", "character_key"),
      str(CharacterFanout.RETURN_NAMES))

check("lora_name is wildcard-typed so it can feed a COMBO input",
      not (CharacterFanout.RETURN_TYPES[2] != "COMBO"),
      str(CharacterFanout.RETURN_TYPES))

prompts, triggers, loras, keys = fan_out(0)
check("mode 0 yields one entry per primary prompt with A settings",
      prompts == ["prompt a1", "prompt a2"]
      and triggers == ["triggerA", "triggerA"]
      and loras == ["krea2/a.safetensors", "krea2/a.safetensors"]
      and keys == ["hotfitlena", "hotfitlena"],
      str((prompts, triggers, loras, keys)))

prompts, triggers, loras, keys = fan_out(1)
check("mode 1 uses the primary prompts with B settings",
      prompts == ["prompt a1", "prompt a2"]
      and triggers == ["triggerB", "triggerB"]
      and loras == ["krea2/b.safetensors", "krea2/b.safetensors"]
      and keys == ["tatum", "tatum"],
      str((prompts, triggers, loras, keys)))

prompts, triggers, loras, keys = fan_out(2)
check("mode 2 appends the B block after the A block",
      prompts == ["prompt a1", "prompt a2", "prompt b1", "prompt b2"]
      and triggers == ["triggerA", "triggerA", "triggerB", "triggerB"]
      and loras == ["krea2/a.safetensors", "krea2/a.safetensors",
                    "krea2/b.safetensors", "krea2/b.safetensors"]
      and keys == ["hotfitlena", "hotfitlena", "tatum", "tatum"],
      str((prompts, triggers, loras, keys)))

check("all four output lists always have the same length",
      len({len(part) for part in fan_out(2)}) == 1,
      str([len(part) for part in fan_out(2)]))

prompts, triggers, loras, keys = fan_out(
    2, primary="  prompt a1  \n\n   \nprompt a2\n", secondary="prompt b1\n\n")
check("blank and whitespace-only lines are dropped and entries trimmed",
      prompts == ["prompt a1", "prompt a2", "prompt b1"]
      and keys == ["hotfitlena", "hotfitlena", "tatum"],
      str((prompts, keys)))

try:
    fan_out(2, secondary="   \n\n")
    check("mode 2 with an empty secondary block raises", False, "no error raised")
except ValueError as error:
    check("mode 2 with an empty secondary block raises",
          "prompts_secondary" in str(error), str(error))

try:
    fan_out(0, primary="")
    check("an empty primary block raises", False, "no error raised")
except ValueError as error:
    check("an empty primary block raises",
          "prompts_primary" in str(error), str(error))

try:
    NODE.fan_out(
        mode=1, mapping="only_one = 0",
        prompts_primary=PROMPTS_A, prompts_secondary=PROMPTS_B,
        trigger_a="triggerA", trigger_b="triggerB",
        lora_a="krea2/a.safetensors", lora_b="krea2/b.safetensors",
    )
    check("a mapping without value 1 raises", False, "no error raised")
except ValueError as error:
    check("a mapping without value 1 raises", "1" in str(error), str(error))

prompts, triggers, loras, keys = NODE.fan_out(
    mode=2, mapping="# comment\n  hotfitlena   =   0  \ntatum=1\nboth = 2",
    prompts_primary="p1", prompts_secondary="p2",
    trigger_a="triggerA", trigger_b="triggerB",
    lora_a="krea2/a.safetensors", lora_b="krea2/b.safetensors",
)
check("mapping parsing tolerates comments and loose whitespace",
      keys == ["hotfitlena", "tatum"], str(keys))

try:
    fan_out(7)
    check("an unknown mode raises", False, "no error raised")
except ValueError as error:
    check("an unknown mode raises", "mode" in str(error), str(error))

print()
if FAILURES:
    print(f"{len(FAILURES)} TEST(S) FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("ALL TESTS PASSED")

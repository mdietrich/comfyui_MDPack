"""Pure-logic tests for video_to_h3_prompts (no model, no GPU, no ffmpeg)."""

import os
import shutil
import tempfile
import unittest

from video_to_h3_prompts import (assemble_clip_prompts, chain_state_path,
                                 closing_from_prompt, load_chain_state,
                                 save_chain_state, state_clip,
                                 fill_placeholders, overlap_ratio,
                                 retry_directive,
                                 strip_absence_sentences,
                                 parse_clip_answer, recent_actions,
                                 stub_clip_warning,
                                 plan_clips, strip_meta, transcript_window)


class PlanClipsTest(unittest.TestCase):
    def test_exact_multiple_has_no_remainder_clip(self):
        clips = plan_clips(duration=45.0, chunk_seconds=15.0, fps=24)
        self.assertEqual(len(clips), 3)
        self.assertEqual([c[0] for c in clips], [1, 2, 3])
        self.assertAlmostEqual(clips[0][1], 0.0)
        self.assertAlmostEqual(clips[-1][2], 45.0)

    def test_remainder_becomes_a_shorter_final_clip(self):
        clips = plan_clips(duration=50.0, chunk_seconds=15.0, fps=24)
        self.assertEqual(len(clips), 4)
        self.assertAlmostEqual(clips[-1][2] - clips[-1][1], 5.0, places=2)

    def test_clips_are_contiguous(self):
        clips = plan_clips(duration=61.3, chunk_seconds=12.0, fps=24)
        for previous, current in zip(clips, clips[1:]):
            self.assertAlmostEqual(previous[2], current[1], places=6)

    def test_short_video_yields_one_clip(self):
        self.assertEqual(len(plan_clips(3.0, 15.0, 24)), 1)

    def test_clip_grid_follows_the_frame_rate(self):
        # 24 fps and 30 fps must agree on the clip count for the same seconds.
        self.assertEqual(len(plan_clips(45.0, 15.0, 24)),
                         len(plan_clips(45.0, 15.0, 30)))


class TranscriptWindowTest(unittest.TestCase):
    SEGMENTS = [
        {"start": 0.0, "end": 4.0, "text": " Hello and welcome. "},
        {"start": 4.5, "end": 9.0, "text": "Today we talk about lenses."},
        {"start": 16.0, "end": 20.0, "text": "Second clip line."},
    ]

    def test_collects_every_overlapping_segment(self):
        self.assertEqual(transcript_window(self.SEGMENTS, 0.0, 15.0),
                         "Hello and welcome. Today we talk about lenses.")

    def test_excludes_segments_outside_the_window(self):
        self.assertEqual(transcript_window(self.SEGMENTS, 15.0, 30.0),
                         "Second clip line.")

    def test_partial_overlap_counts(self):
        self.assertIn("lenses", transcript_window(self.SEGMENTS, 8.0, 15.0))

    def test_silent_window_is_empty(self):
        self.assertEqual(transcript_window(self.SEGMENTS, 10.0, 15.0), "")


class StripMetaTest(unittest.TestCase):
    def test_removes_clip_tag(self):
        self.assertEqual(strip_meta("[3] She leans forward."),
                         "She leans forward.")

    def test_removes_clip_heading(self):
        self.assertEqual(strip_meta("Clip 2: She leans forward."),
                         "She leans forward.")

    def test_removes_code_fence_and_think_block(self):
        self.assertEqual(
            strip_meta("<think>hmm</think>```json\nShe leans forward.\n```"),
            "She leans forward.")

    def test_collapses_line_breaks(self):
        self.assertEqual(strip_meta("She leans\nforward."),
                         "She leans forward.")


class ParseClipAnswerTest(unittest.TestCase):
    def test_reads_complete_json_object(self):
        raw = ('She leans forward.", "closing_state": "Hands on the desk.", '
               '"used": ["left hand", "Desk"]}')
        prompt, closing, used = parse_clip_answer(raw)
        self.assertEqual(prompt, "She leans forward.")
        self.assertEqual(closing, "Hands on the desk.")
        self.assertEqual(used, ["left hand", "desk"])

    def test_salvages_truncated_object(self):
        raw = 'She leans forward.", "closing_state": "Hands on the de'
        prompt, closing, used = parse_clip_answer(raw)
        self.assertEqual(prompt, "She leans forward.")
        self.assertEqual(closing, "")
        self.assertEqual(used, [])

    def test_falls_back_to_plain_prose(self):
        prompt, closing, used = parse_clip_answer("She leans forward.", prefill="")
        self.assertEqual(prompt, "She leans forward.")
        self.assertEqual(closing, "")
        self.assertEqual(used, [])

    def test_unescapes_quoted_dialogue(self):
        raw = ('She says \\"good morning\\".", "closing_state": "Silence."}')
        prompt, _, _ = parse_clip_answer(raw)
        self.assertEqual(prompt, 'She says "good morning".')

    def test_missing_used_key_is_an_empty_list(self):
        raw = 'She leans forward.", "closing_state": "Still."}'
        self.assertEqual(parse_clip_answer(raw)[2], [])


class AssembleClipPromptsTest(unittest.TestCase):
    def test_tags_are_one_based_and_line_initial(self):
        text = assemble_clip_prompts(["first", "second"])
        self.assertEqual(text, "[1] first\n\n[2] second")

    def test_matches_the_auto_chain_parser(self):
        import re
        text = assemble_clip_prompts(["first prompt", "second prompt"])
        matches = list(re.finditer(
            r"(?ms)^\s*\[(\d+)\]\s*(.*?)(?=^\s*\[\d+\]\s*|\Z)", text))
        found = {int(m.group(1)): m.group(2).strip() for m in matches}
        self.assertEqual(found, {1: "first prompt", 2: "second prompt"})


class StubClipWarningTest(unittest.TestCase):
    def test_warns_about_a_sub_second_tail(self):
        clips = plan_clips(duration=165.3, chunk_seconds=15.0, fps=24)
        message = stub_clip_warning(clips, 15.0)
        self.assertIn("clip 12", message)
        self.assertIn("15.03", message)

    def test_suggestion_produces_an_even_grid(self):
        clips = plan_clips(duration=165.3, chunk_seconds=15.0, fps=24)
        message = stub_clip_warning(clips, 15.0)
        suggested = float(message.split("chunk_seconds to ")[1].split(" ")[0])
        self.assertEqual(len(plan_clips(165.3, suggested, 24)), len(clips) - 1)

    def test_silent_on_an_even_grid(self):
        self.assertEqual(stub_clip_warning(plan_clips(45.0, 15.0, 24), 15.0), "")

    def test_silent_on_an_acceptable_remainder(self):
        self.assertEqual(stub_clip_warning(plan_clips(50.0, 15.0, 24), 15.0), "")

    def test_silent_on_a_single_clip(self):
        self.assertEqual(stub_clip_warning(plan_clips(3.0, 15.0, 24), 15.0), "")


class ClosingFromPromptTest(unittest.TestCase):
    def test_takes_the_last_sentence(self):
        self.assertEqual(
            closing_from_prompt("She stands up. Her hand rests on the door."),
            "Her hand rests on the door.")

    def test_handles_a_single_sentence(self):
        self.assertEqual(closing_from_prompt("She stands up."), "She stands up.")

    def test_empty_input_is_empty(self):
        self.assertEqual(closing_from_prompt(""), "")


class RecentActionsTest(unittest.TestCase):
    def test_no_directive_before_the_first_clip(self):
        self.assertEqual(recent_actions([], []), "")

    def test_lists_only_the_most_recent_prompts(self):
        directive = recent_actions(["one", "two", "three", "four"], [], count=2)
        self.assertNotIn("- one", directive)
        self.assertIn("- four", directive)

    def test_skips_clips_that_were_not_written(self):
        directive = recent_actions(["written", "", ""], [], count=3)
        self.assertIn("- written", directive)

    def test_names_the_banned_phrases(self):
        self.assertIn("shifts her weight", recent_actions(["anything"], []))

    def test_cumulative_list_reaches_past_the_text_window(self):
        # The element from clip 1 must still be banned while writing clip 11.
        prompts = ["clip %d" % i for i in range(1, 11)]
        directive = recent_actions(prompts, ["left leg", "dresser"], count=2)
        self.assertNotIn("- clip 1\n", directive)
        self.assertIn("left leg", directive)
        self.assertIn("dresser", directive)

    def test_cumulative_list_is_deduplicated_and_sorted(self):
        directive = recent_actions([], ["Dresser", "dresser", "bow"])
        listed = directive.split("around any of them:\n")[1].splitlines()[0]
        self.assertEqual(listed, "bow, dresser")

    def test_used_alone_is_enough_for_a_directive(self):
        self.assertIn("left leg", recent_actions([], ["left leg"]))


class OverlapRatioTest(unittest.TestCase):
    def test_fully_reused_clip_scores_one(self):
        self.assertEqual(
            overlap_ratio(["left foot", "dresser"],
                          ["left foot", "dresser", "bow"]), 1.0)

    def test_fresh_clip_scores_zero(self):
        self.assertEqual(overlap_ratio(["collarbone"], ["left foot"]), 0.0)

    def test_half_reused_clip_scores_half(self):
        self.assertEqual(
            overlap_ratio(["left foot", "collarbone"], ["left foot"]), 0.5)

    def test_comparison_ignores_case_and_padding(self):
        self.assertEqual(overlap_ratio([" Left Foot "], ["left foot"]), 1.0)

    def test_clip_without_reported_elements_is_not_flagged(self):
        self.assertEqual(overlap_ratio([], ["left foot"]), 0.0)

    def test_first_clip_has_nothing_to_overlap_with(self):
        self.assertEqual(overlap_ratio(["left foot"], []), 0.0)


class RetryDirectiveTest(unittest.TestCase):
    def test_names_only_the_repeated_elements(self):
        directive = retry_directive(["left foot", "collarbone"],
                                    ["left foot", "bow"])
        self.assertIn("left foot", directive)
        self.assertNotIn("collarbone", directive)
        self.assertNotIn("bow", directive)

    def test_states_the_attempt_was_rejected(self):
        self.assertIn("REJECTED", retry_directive(["a"], ["a"]))


class StripAbsenceSentencesTest(unittest.TestCase):
    def test_removes_the_does_not_speak_sentence(self):
        text = ("She lifts her arm. She does not speak. The scene settles.")
        self.assertEqual(strip_absence_sentences(text),
                         "She lifts her arm. The scene settles.")

    def test_removes_remains_silent(self):
        self.assertEqual(
            strip_absence_sentences("She nods. The woman remains silent."),
            "She nods.")

    def test_removes_there_is_no_sound(self):
        self.assertEqual(
            strip_absence_sentences("She nods. There is no sound."), "She nods.")

    def test_removes_nothing_moves(self):
        self.assertEqual(
            strip_absence_sentences("She nods. Nothing moves."), "She nods.")

    def test_keeps_a_real_quoted_line(self):
        text = 'She leans in and says "good morning". The scene settles.'
        self.assertEqual(strip_absence_sentences(text), text)

    def test_keeps_ambient_sound_description(self):
        text = "She nods. A fan hums in the corner."
        self.assertEqual(strip_absence_sentences(text), text)

    def test_never_returns_an_empty_prompt(self):
        # A prompt that is nothing but an absence sentence is left alone
        # rather than emptied, so the failure stays visible downstream.
        self.assertEqual(strip_absence_sentences("She does not speak."),
                         "She does not speak.")

    def test_is_case_insensitive(self):
        self.assertEqual(
            strip_absence_sentences("She nods. SHE DOES NOT SPEAK."),
            "She nods.")


class FillPlaceholdersTest(unittest.TestCase):
    def test_substitutes_the_named_placeholder(self):
        self.assertEqual(fill_placeholders("about {clip_words} words",
                                           clip_words=110),
                         "about 110 words")

    def test_substitutes_every_occurrence(self):
        self.assertEqual(fill_placeholders("{a} and {a}", a="x"), "x and x")

    def test_survives_a_stray_brace(self):
        # A user pasting JSON into the instruction widget used to crash the run.
        text = 'return {"prompt": "..."} with {clip_words} words'
        self.assertEqual(fill_placeholders(text, clip_words=90),
                         'return {"prompt": "..."} with 90 words')

    def test_leaves_an_unknown_placeholder_untouched(self):
        self.assertEqual(fill_placeholders("{unknown}", clip_words=1),
                         "{unknown}")


class ChainStatePathTest(unittest.TestCase):
    def test_sanitises_the_name(self):
        path = chain_state_path("/tmp/x", "my chain/../name")
        self.assertEqual(os.path.basename(path), "my_chain_.._name.json")

    def test_empty_name_falls_back(self):
        self.assertEqual(os.path.basename(chain_state_path("/tmp/x", "")),
                         "h3_prompt_chain.json")


class ChainStateRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "chain.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_saved_chain_reads_back_identically(self):
        clips = {1: {"prompt": "She stands.", "closing": "Standing.",
                     "used": ["legs"]}}
        save_chain_state(self.path, "a style block", clips)
        state = load_chain_state(self.path)
        self.assertEqual(state["style"], "a style block")
        self.assertEqual(state_clip(state, 1),
                         {"prompt": "She stands.", "closing": "Standing.",
                          "used": ["legs"]})

    def test_missing_file_is_an_empty_chain(self):
        self.assertEqual(load_chain_state(os.path.join(self.dir, "nope.json")),
                         {"style": "", "clips": {}})

    def test_damaged_file_is_treated_as_absent(self):
        with open(self.path, "w") as handle:
            handle.write("{not json")
        self.assertEqual(load_chain_state(self.path), {"style": "", "clips": {}})

    def test_unknown_clip_is_blank_not_an_error(self):
        save_chain_state(self.path, "s", {1: {"prompt": "a", "closing": "b",
                                              "used": []}})
        self.assertEqual(state_clip(load_chain_state(self.path), 9)["prompt"], "")

    def test_creates_the_directory(self):
        nested = os.path.join(self.dir, "deep", "chain.json")
        save_chain_state(nested, "s", {})
        self.assertTrue(os.path.exists(nested))


if __name__ == "__main__":
    unittest.main()

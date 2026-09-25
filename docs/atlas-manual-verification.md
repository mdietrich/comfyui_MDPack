# Atlas Cloud model dropdown — manual verification checklist

This checklist covers everything the automated test suites (`test_atlascloud_api.py`,
`test_atlascloud_models.py`, `test_atlascloud_routes.py`, `test_atlascloud_web.mjs`,
`verify_atlascloud_end_to_end.py`) cannot verify on their own, because it needs a
running ComfyUI instance with a real browser attached. It was owed once the
`gamingmonster` host went offline mid-project (Tasks 7–10) and by the two live
generations in the original plan (Task 11).

Work through it top to bottom once the host is back. Each item states what to
do and what result to expect; nothing here requires reading the implementation
plan first.

## 0. Deploy

- [ ] Copy the whole `comfyui_MDPack` directory (in particular `web/atlasCloudModels.js`,
  `atlascloud_api.py`, `atlascloud_models.py`, `atlascloud_routes.py`) to
  `~/ComfyUI_winows_portable/ComfyUI/custom_nodes/comfyui_MDPack/` on `gamingmonster`.
  **Expected:** ComfyUI starts without errors in the console/log, and `AtlasCloudImage` /
  `AtlasCloudVideo` appear in the node search under the `AtlasCloud` category.

## 1. Backward compatibility — load an old saved workflow (Fix 1)

The highest-value item on this list, and put here deliberately near the top:
this is the one regression the final review flagged that nobody has actually
opened in a browser. Fix 1 (the widget-order bug that scrambled a saved
workflow's values) was verified programmatically against the user's real
saved widget values, but never against a live LiteGraph load.

- [ ] Find a workflow saved with `AtlasCloudImage` before this fix wave landed
  (any workflow saved before 2026-07-30, or a `.json` whose node has 15
  `widgets_values` for `AtlasCloudImage` / 16 for `AtlasCloudVideo`, i.e. no
  `output_type`/`input_type` serialized at all). If none is available, ask
  the user for one rather than fabricating a fresh one — a freshly saved
  workflow would exercise the current (already-fixed) ordering, not the bug.
  Open it in ComfyUI.
  **Expected:** every widget shows the same value it showed before — in
  particular `model` still holds the model id (not garbled or shifted),
  `width` shows `1328`, `height` shows `1776`, `image_field` shows `images`
  (or whatever the workflow saved) — not a value that belongs to a
  different widget.
  **Expected:** `output_type` and `input_type` sit at the very bottom of the
  widget list and show their class defaults (`Image` / `Any` for
  `AtlasCloudImage`, `Video` / `Any` for `AtlasCloudVideo`), since a pre-fix
  workflow never serialized them.
- [ ] Queue the loaded workflow without changing anything.
  **Expected:** no `Value not in list` / COMBO validation error, and no
  widget visibly holds a value that belongs to a different field (e.g. a
  JSON blob in `image_transport`, or a model id in `output_type`).

## 2. Model dropdown (Task 7)

- [ ] Add an `AtlasCloudImage` node. Look at the `model` widget.
  **Expected:** it renders as a dropdown/combo box, not a free-text field.
- [ ] Change `output_type` from `Image` to `Video`.
  **Expected:** the `model` dropdown's list changes to video models (different entries).
- [ ] Set `output_type=Image`, `input_type=Image`.
  **Expected:** roughly 53 entries in the list, including "Seedream v5.0 Pro Edit".
- [ ] Select a model, save the workflow, then reload the page (or reopen the saved
  workflow file).
  **Expected:** the previously selected model is still shown as selected in the dropdown
  (not reset to the first entry).
- [ ] Look at the dropdown's visible option text.
  **Expected:** entries display as `"<display name> — $<price>"` (e.g.
  `"Seedream v5.0 Pro Edit — $0.036"`), not raw model ids. If it instead shows raw ids,
  that is a cosmetic-only gap (see Task 7 report) — note it but it does not block anything
  else on this list.

## 3. Schema-driven parameter widgets — image node (Task 8)

- [ ] With `output_type=Image`, `input_type=Text`, select model "Seedream v5.0 Pro"
  (text-to-image).
  **Expected:** dynamic `param_size`, `param_output_format`, `param_thinking` widgets
  appear; the built-in `width`/`height` widgets are hidden (superseded by `param_size`).
- [ ] Immediately after selecting the model, before touching any `param_*` widget, look at
  `params_json`.
  **Expected (Fix 2):** `params_json` is `{}` — the widgets display the schema's defaults
  (e.g. `param_size` shows `2048*2048`) but none of that is written into `params_json` yet.
- [ ] Change one of the `param_*` widget values.
  **Expected:** the `params_json` widget's text content updates to include that value.
- [ ] **(Fix 2 / `explicitValue`)** Touch a `param_*` widget (e.g. change
  `param_output_format` from `jpeg` to `png`), confirm it now appears in `params_json`, then
  set it back to its original default value (`jpeg`).
  **Expected:** the value still appears explicitly in `params_json` after being set back to
  the default — it does not disappear or become implicit again. This is the only way to
  exercise the `explicitValue` tracking against a real, live LiteGraph widget; the automated
  JS suite (`test_atlascloud_web.mjs`) only simulates a widget's `callback()`, it never
  drives an actual ComfyUI UI.
- [ ] Save the workflow with a few `param_*` values set (non-default), then reload it.
  **Expected:** the same values are restored into the widgets — not reset to the schema's
  defaults.
- [ ] Switch `model` to a different one (e.g. "Seedream v5.0 Pro Edit").
  **Expected:** the old model's `param_*` widgets disappear, the new model's own appear,
  and `params_json` afterward contains only the new schema's field names.

## 4. Schema-driven parameter widgets — video node (Task 8 handover + Task 10)

- [ ] Add an `AtlasCloudVideo` node, set `output_type=Video`, `input_type=Image`, and
  select model "Kling v2.5 Turbo Pro" (image-to-video).
  **Expected:** `param_duration` and the model's other schema-declared parameter widgets
  appear; the legacy `duration` INT widget is hidden (superseded by `param_duration`).
- [ ] Change `param_duration`'s value, save the workflow, and reload it.
  **Expected:** the value persists in `params_json` and is restored into the widget after
  reload (same behaviour as the image node in section 3).

## 5. Dynamic image inputs (Task 9)

- [ ] On `AtlasCloudImage`, select "Seedream v5.0 Pro Edit".
  **Expected:** 10 image slots (`image` .. `image_10`) become visible/usable, labelled
  `images 1` through `images 10` (the schema's single `images` list field, numbered per
  slot — not the slot names `image`/`image_2`/...).
- [ ] Select "Seedream v5.0 Pro" (text-to-image, no reference images).
  **Expected:** all image slots are hidden. If they instead stay visible, see item 5a below
  — that is a known possible gap, not a new bug.
- [ ] **(5a)** Note explicitly whether this ComfyUI frontend version honours `input.hidden`
  for IMAGE-type node inputs (this determines the expected result of the previous item and
  the next one).
  **Expected either way is informational.** If `input.hidden` is not honoured, unused slots
  stay visible but are still correctly labelled; this is cosmetic only and does not affect
  what gets sent to Atlas Cloud, since routing depends on the schema/`input_map`, not on
  whether a slot is visually hidden.
- [ ] On `AtlasCloudVideo`, select "Kling v2.5 Turbo Pro".
  **Expected:** exactly 2 image slots are shown, labelled `image` and `last_image`. With
  both connected, the `input_map` widget shows `{"image":0,"last_image":1}`.
- [ ] With Kling selected, connect an image only to the slot labelled `last_image` (leave
  the `image` slot disconnected).
  **Expected:** `input_map` reflects only `last_image` mapped; no "may misroute" or
  "dropped" diagnostic appears in the ComfyUI console for this case (nothing is actually
  wrong here — one connected reference, one field to receive it).
- [ ] Connect an image to a slot the currently selected model doesn't use (e.g. connect
  `image_5` while a model with only 2 media fields is selected).
  **Expected:** that slot is never hidden — a connected-but-unused slot always stays
  visible — and is labelled `(unused by this model)`; a `console.warn` appears in the
  browser console.
- [ ] Switch from a model with many image fields (e.g. Seedream Edit, 10 slots) to one with
  none (e.g. Seedream text-to-image) and back again.
  **Expected:** no leftover labels from the narrow model persist — slot labels reset to
  their plain names (`image`, `image_2`, ...) each time the schema changes.

## 6. Live generations — the original bug, in production (Task 11 brief Steps 1–2)

These are the two generations the original plan called for, run against the real Atlas
Cloud API. They cost money (real API calls) — run them deliberately, not as a drive-by
check.

- [ ] **Edit model.** Build a workflow: four `LoadImage` nodes → `AtlasCloudImage` with
  `output_type=Image`, `input_type=Image`, model "Seedream v5.0 Pro Edit",
  `param_size = 1328*1776`. Queue it.
  **Expected:** the node's `Info` output reports `references=4 field=images`, and the
  returned image is 1328×1776 (check the `Info` string's `images=... size=WxH` line, or
  inspect the IMAGE output directly, e.g. via a Preview Image / Save Image node).
- [ ] **Text-to-image model.** Same node, switch `input_type=Text`, model
  "Seedream v5.0 Pro", disconnect all image inputs, `param_size = 1328*1776`. Queue it.
  **Expected:** the returned image is 1328×1776, matching the requested size (this is the
  exact scenario of the original defect — 1328×1776 requested, 1584×2816 received). Also
  check the ComfyUI server log line `AtlasCloudImage: submitting model=...` — the console
  only prints the model id, not the full payload, so to confirm no `images` field was sent
  you may need to temporarily add a print or check via the `Info` output's `references=0
  field=(none)` line instead.

## 7. General diagnostics sanity check

While working through the above, keep an eye on the ComfyUI console for the diagnostics
documented in the README's "Model selection" section — unknown-key warnings, out-of-enum
size warnings, dropped-reference warnings, and possible `input_map` misroute warnings. None
of the scenarios above are expected to trigger one falsely; if any diagnostic fires
unexpectedly, note which item triggered it and what it said, since that would point to a
real gap between what the browser assumes and what the schema actually declares.

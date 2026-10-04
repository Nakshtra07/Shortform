# AutoShorts 9.0 — Caption Regression & Visual QA Report

Harness: `caption_qa` (Rust inspection binary) + `caption_visual_qa.py` (pixel layer)

## Summary

| Status | Count |
|---|---|
| PASS | 96 |
| FAIL | 0 |
| SKIPPED | 3 |
| NOT VERIFIED | 1 |

No overall numeric quality score is produced (per Part 13).

## A: template isolation

| ID | Check | Status | Actual | Expected | Expected source | Severity |
|---|---|---|---|---|---|---|
| ISO-REGISTRY | Template registry T1..T7 present | **PASS** | ["preset_viral_bold", "preset_mrbeast_pop", "preset_minimal_capsule", "preset_cinematic_vlog", "preset_dynamic_editorial | 7 templates: preset_viral_bold, preset_mrbeast_pop, preset_minimal_capsule, preset_cinematic_vlog, preset_dynamic_editor | captions.rs all_caption_templates() | none |
| ISO-T7 | T7 template registered | **PASS** | preset_t7 exists in the 10.0 registry | T7 Reference Style (Matt Bold 700, white + #F9EF07 current-chunk, phrase-chunked rolling, no stroke/shadow/box) | captions.rs all_caption_templates() | none |
| ISO-T8 | T8 template | **SKIPPED** | preset_t8 does not exist in the 10.0 registry | n/a | n/a | none |
| ISO-T8 | T8 template | **SKIPPED** | T8 does not exist in the 10.0 registry | n/a | n/a | none |
| ISO-FONT-preset_viral_bold | preset_viral_bold uses only its own fonts | **PASS** | ["Bebas Neue"] | subset of ["Bebas Neue"] | captions.rs template registry | none |
| ISO-FONT-preset_mrbeast_pop | preset_mrbeast_pop uses only its own fonts | **PASS** | ["Montserrat"] | subset of ["Montserrat", "Poppins", "Inter"] | captions.rs template registry | none |
| ISO-FONT-preset_minimal_capsule | preset_minimal_capsule uses only its own fonts | **PASS** | ["Inter"] | subset of ["Inter"] | captions.rs template registry | none |
| ISO-FONT-preset_cinematic_vlog | preset_cinematic_vlog uses only its own fonts | **PASS** | ["Poppins"] | subset of ["Poppins"] | captions.rs template registry | none |
| ISO-FONT-preset_dynamic_editorial | preset_dynamic_editorial uses only its own fonts | **PASS** | ["Montserrat"] | subset of ["Bebas Neue", "Montserrat", "Inter", "Poppins"] | captions.rs template registry | none |
| ISO-FONT-preset_bhaukal_caption | preset_bhaukal_caption uses only its own fonts | **PASS** | ["Montserrat", "Inter"] | subset of ["Montserrat", "Poppins", "Inter"] | captions.rs template registry | none |
| ISO-FONT-preset_t7 | preset_t7 uses only its own fonts | **PASS** | ["Matt_Trial-Bold"] | subset of ["Matt_Trial-Bold"] | captions.rs template registry | none |
| ISO-MUT-preset_viral_bold-preset_mrbeast_pop | Mutating preset_viral_bold leaves preset_mrbeast_pop unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_viral_bold-preset_minimal_capsule | Mutating preset_viral_bold leaves preset_minimal_capsule unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_viral_bold-preset_cinematic_vlog | Mutating preset_viral_bold leaves preset_cinematic_vlog unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_viral_bold-preset_dynamic_editorial | Mutating preset_viral_bold leaves preset_dynamic_editorial unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_viral_bold-preset_bhaukal_caption | Mutating preset_viral_bold leaves preset_bhaukal_caption unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_viral_bold-preset_t7 | Mutating preset_viral_bold leaves preset_t7 unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-SELF-preset_viral_bold | Mutation of preset_viral_bold actually changes its own ASS | **PASS** | own ASS changed | own ASS changes | captions.rs registry | none |
| ISO-MUT-preset_mrbeast_pop-preset_viral_bold | Mutating preset_mrbeast_pop leaves preset_viral_bold unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_mrbeast_pop-preset_minimal_capsule | Mutating preset_mrbeast_pop leaves preset_minimal_capsule unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_mrbeast_pop-preset_cinematic_vlog | Mutating preset_mrbeast_pop leaves preset_cinematic_vlog unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_mrbeast_pop-preset_dynamic_editorial | Mutating preset_mrbeast_pop leaves preset_dynamic_editorial unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_mrbeast_pop-preset_bhaukal_caption | Mutating preset_mrbeast_pop leaves preset_bhaukal_caption unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_mrbeast_pop-preset_t7 | Mutating preset_mrbeast_pop leaves preset_t7 unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-SELF-preset_mrbeast_pop | Mutation of preset_mrbeast_pop actually changes its own ASS | **PASS** | own ASS changed | own ASS changes | captions.rs registry | none |
| ISO-MUT-preset_minimal_capsule-preset_viral_bold | Mutating preset_minimal_capsule leaves preset_viral_bold unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_minimal_capsule-preset_mrbeast_pop | Mutating preset_minimal_capsule leaves preset_mrbeast_pop unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_minimal_capsule-preset_cinematic_vlog | Mutating preset_minimal_capsule leaves preset_cinematic_vlog unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_minimal_capsule-preset_dynamic_editorial | Mutating preset_minimal_capsule leaves preset_dynamic_editorial unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_minimal_capsule-preset_bhaukal_caption | Mutating preset_minimal_capsule leaves preset_bhaukal_caption unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_minimal_capsule-preset_t7 | Mutating preset_minimal_capsule leaves preset_t7 unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-SELF-preset_minimal_capsule | Mutation of preset_minimal_capsule actually changes its own ASS | **PASS** | own ASS changed | own ASS changes | captions.rs registry | none |
| ISO-MUT-preset_cinematic_vlog-preset_viral_bold | Mutating preset_cinematic_vlog leaves preset_viral_bold unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_cinematic_vlog-preset_mrbeast_pop | Mutating preset_cinematic_vlog leaves preset_mrbeast_pop unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_cinematic_vlog-preset_minimal_capsule | Mutating preset_cinematic_vlog leaves preset_minimal_capsule unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_cinematic_vlog-preset_dynamic_editorial | Mutating preset_cinematic_vlog leaves preset_dynamic_editorial unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_cinematic_vlog-preset_bhaukal_caption | Mutating preset_cinematic_vlog leaves preset_bhaukal_caption unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_cinematic_vlog-preset_t7 | Mutating preset_cinematic_vlog leaves preset_t7 unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-SELF-preset_cinematic_vlog | Mutation of preset_cinematic_vlog actually changes its own ASS | **PASS** | own ASS changed | own ASS changes | captions.rs registry | none |
| ISO-MUT-preset_dynamic_editorial-preset_viral_bold | Mutating preset_dynamic_editorial leaves preset_viral_bold unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_dynamic_editorial-preset_mrbeast_pop | Mutating preset_dynamic_editorial leaves preset_mrbeast_pop unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_dynamic_editorial-preset_minimal_capsule | Mutating preset_dynamic_editorial leaves preset_minimal_capsule unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_dynamic_editorial-preset_cinematic_vlog | Mutating preset_dynamic_editorial leaves preset_cinematic_vlog unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_dynamic_editorial-preset_bhaukal_caption | Mutating preset_dynamic_editorial leaves preset_bhaukal_caption unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_dynamic_editorial-preset_t7 | Mutating preset_dynamic_editorial leaves preset_t7 unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-SELF-preset_dynamic_editorial | Mutation of preset_dynamic_editorial actually changes its own ASS | **PASS** | own ASS changed | own ASS changes | captions.rs registry | none |
| ISO-MUT-preset_bhaukal_caption-preset_viral_bold | Mutating preset_bhaukal_caption leaves preset_viral_bold unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_bhaukal_caption-preset_mrbeast_pop | Mutating preset_bhaukal_caption leaves preset_mrbeast_pop unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_bhaukal_caption-preset_minimal_capsule | Mutating preset_bhaukal_caption leaves preset_minimal_capsule unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_bhaukal_caption-preset_cinematic_vlog | Mutating preset_bhaukal_caption leaves preset_cinematic_vlog unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_bhaukal_caption-preset_dynamic_editorial | Mutating preset_bhaukal_caption leaves preset_dynamic_editorial unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_bhaukal_caption-preset_t7 | Mutating preset_bhaukal_caption leaves preset_t7 unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-SELF-preset_bhaukal_caption | Mutation of preset_bhaukal_caption actually changes its own ASS | **PASS** | own ASS changed | own ASS changes | captions.rs registry | none |
| ISO-MUT-preset_t7-preset_viral_bold | Mutating preset_t7 leaves preset_viral_bold unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_t7-preset_mrbeast_pop | Mutating preset_t7 leaves preset_mrbeast_pop unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_t7-preset_minimal_capsule | Mutating preset_t7 leaves preset_minimal_capsule unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_t7-preset_cinematic_vlog | Mutating preset_t7 leaves preset_cinematic_vlog unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_t7-preset_dynamic_editorial | Mutating preset_t7 leaves preset_dynamic_editorial unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-preset_t7-preset_bhaukal_caption | Mutating preset_t7 leaves preset_bhaukal_caption unchanged | **PASS** | byte-identical ASS | unchanged ASS | captions.rs registry (per-template values, no shared mutable state) | none |
| ISO-MUT-SELF-preset_t7 | Mutation of preset_t7 actually changes its own ASS | **PASS** | own ASS changed | own ASS changes | captions.rs registry | none |

## B: real clip audit

| ID | Check | Status | Actual | Expected | Expected source | Severity |
|---|---|---|---|---|---|---|
| CHAIN-CODE | Template exists in registry (CODE EXISTS) | **PASS** | preset_t7 → T7 Reference Style | registered template | captions.rs all_caption_templates() | none |
| CHAIN-ASS | Generated ASS artifact exists and parses (OUTPUT WAS CONSUMED) | **PASS** | 6966 bytes, 1 styles, 75 events | parseable ASS with styles+events | lib.rs writes clip-<id>.ass before render | none |
| CHAIN-MP4 | Final MP4 exists (FINAL OUTPUT) | **PASS** | C:\Users\naksh\OneDrive\Documents\AutoShorts\Joe-Rogan-Experience--2422---Jensen-Huang---PowerfulJRE--1080p--h264\clips\ | existing mp4 | pacing::render_paced_clip output | none |
| TPL-FONT | Template font family reaches ASS Style | **PASS** | Matt_Trial-Bold | Matt_Trial-Bold | captions.rs all_caption_templates() registry | none |
| TPL-SIZE | Template font size reaches ASS Style | **PASS** | 110 | 110 | captions.rs template registry + apply_subject_face_safeguard | none |
| TPL-WEIGHT | Template font weight reaches ASS Bold | **PASS** | -1 | weight 700 → bold -1 | captions.rs generate_ass (weight→bold mapping) | none |
| TPL-COLOR-PRIMARY | Template primary font color reaches ASS | **PASS** | &H00FFFFFF | &H00FFFFFF | captions.rs hex_to_ass_header(font_color) | none |
| TPL-NO-STROKE | Template has no stroke; ASS outline is 0 | **PASS** | 0 | 0 | captions.rs template registry (stroke_color=None) | none |
| TPL-NO-SHADOW | Template has no shadow; ASS shadow is 0 | **PASS** | 0 | 0 | captions.rs template registry | none |
| TPL-POSITION | Template positionY reaches ASS MarginV (alignment 2) | **PASS** | marginV=647 align=2 | marginV=647 align=2 | captions.rs compute_margin_v(position_y, font_size, max_lines) | none |
| TPL-CASING | Template normal casing; no transform asserted | **PASS** | normal-case template | normal case | captions.rs apply_text_casing | none |
| TPL-SEGMENTATION | Word/line segmentation limits honored in events | **PASS** | all events within 3/2 | <= 3 words/line, <= 2 lines/event | captions.rs wrap_words_into_lines / kinetic groupers | none |
| TPL-ANIMATION | Template active-state animation tags present in ASS | **PASS** | matching transform tags found | tags from template active_state | captions.rs active-state branches | none |
| ASS-EVENTS | Dialogue events exist | **PASS** | 75 dialogue events | >= 1 | captions.rs generators | none |
| ASS-TIMING | All events have valid timing | **PASS** | all events start<end, >=0 | start >= 0 and end > start | ASS timing semantics | none |
| ASS-TAGS | Override tags well-formed (balanced, known) | **PASS** | all tags balanced and in whitelist | balanced {} with known tag names | libass tag set | none |
| ASS-FONTS | Font references resolve to bundled fonts | **PASS** | ["Matt_Trial-Bold"] | font family with required_font_filename mapping | captions.rs required_font_filename + autoshorts/fonts/ | none |
| ASS-LEAKAGE | No unexpected style names | **PASS** | ["Default"] | subset of ["Default", "DualStackSeam"] | captions.rs style emission | none |
| ASS-COVERAGE | All in-window transcript words appear in ASS | **PASS** | 129 in-window words all present | every word with end>start && start<end appears | captions.rs candidate_words filter | none |
| ASS-OVERLAP | No same-style event overlap (sequential cadence) | **PASS** | all sequential | no same style/layer time overlap | captions.rs word-by-word cadence | none |
| CI-EXECUTED | CI plan regenerated from same inputs | **PASS** | plan Some (confidence 1.00, 27 emph words, reason: Hook verified (conf=1.00, 19 words); Payoff verified (score=1.00, 8 w | captionIntelligence=true requires usable plan | caption_intel.rs plan_caption_intel | none |
| CI-REACHED-ASS | CI-derived data reflected in generated ASS | **NOT_VERIFIED** | regenerated ASS identical with and without intel (no emphasized word falls in this clip window or template ignores it) | unknown | n/a | low |
| AF-PACING | smartPacing metadata vs renderLog | **PASS** | flag=false, no pacing in log | flag ⟺ log evidence | lib.rs applied_features | none |
| AF-HOOK-END | hookEndingOptimization metadata vs renderLog | **PASS** | flag=false | no boundary movement | lib.rs applied_features (start_changed//end_changed) | none |
| AF-AUDIO | audioIntelligence metadata vs renderLog | **PASS** | flag=true, log contains [Audio Intelligence] | flag ⟺ log evidence | lib.rs + audio::audio_summary | none |
| AF-CI | captionIntelligence metadata vs real plan | **PASS** | flag=true and plan regenerated Some | flag ⟺ plan exists and was consumed | lib.rs caption_intelligence_applied = usable_and_written && ass_path.is_some() | none |
| AF-PRESET | captionPreset metadata vs ASS template signature | **PASS** | projects.caption_style = preset_t7 | ASS matches preset_t7 | db.rs projects.caption_style + captions.rs template registry | none |
| AF-PRESET-LOG | renderLog caption_template matches metadata | **PASS** | log has 'caption_template: preset_t7' | log template == caption_style | lib.rs log assembly | none |
| QA-COLLISION | T5 word-to-word collision (real metrics) | **SKIPPED** | collision metrics only apply to kinetic T5 layout | n/a | n/a | none |

## E: regression

| ID | Check | Status | Actual | Expected | Expected source | Severity |
|---|---|---|---|---|---|---|
| REG-PROTECTED | Protected pipeline unchanged: autoshorts/src-tauri/src/audio.rs | **PASS** | unchanged (git status clean) | no modification | prompt Part 11 — DO NOT refactor these systems | none |
| REG-PROTECTED | Protected pipeline unchanged: autoshorts/src-tauri/src/boundary.rs | **PASS** | unchanged (git status clean) | no modification | prompt Part 11 — DO NOT refactor these systems | none |
| REG-PROTECTED | Protected pipeline unchanged: autoshorts/src-tauri/src/captions.rs | **PASS** | unchanged (git status clean) | no modification | prompt Part 11 — DO NOT refactor these systems | none |
| REG-PROTECTED | Protected pipeline unchanged: autoshorts/src-tauri/src/caption_intel.rs | **PASS** | unchanged (git status clean) | no modification | prompt Part 11 — DO NOT refactor these systems | none |
| REG-PROTECTED | Protected pipeline unchanged: autoshorts/src-tauri/src/media.rs | **PASS** | unchanged (git status clean) | no modification | prompt Part 11 — DO NOT refactor these systems | none |
| REG-PROTECTED | Protected pipeline unchanged: autoshorts/src-tauri/src/pacing.rs | **PASS** | unchanged (git status clean) | no modification | prompt Part 11 — DO NOT refactor these systems | none |
| REG-PROTECTED | Protected pipeline unchanged: autoshorts/src-tauri/src/transcription.rs | **PASS** | unchanged (git status clean) | no modification | prompt Part 11 — DO NOT refactor these systems | none |
| REG-PROTECTED | Protected pipeline unchanged: autoshorts/src-tauri/src/transcript_normalizer.rs | **PASS** | unchanged (git status clean) | no modification | prompt Part 11 — DO NOT refactor these systems | none |
| REG-PROTECTED | Protected pipeline unchanged: autoshorts/src-tauri/src/youtube.rs | **PASS** | unchanged (git status clean) | no modification | prompt Part 11 — DO NOT refactor these systems | none |
| REG-PROTECTED | Protected pipeline unchanged: autoshorts/src-tauri/src/llm.rs | **PASS** | unchanged (git status clean) | no modification | prompt Part 11 — DO NOT refactor these systems | none |
| REG-PROTECTED | Protected pipeline unchanged: autoshorts/src/main.tsx | **PASS** | unchanged (git status clean) | no modification | prompt Part 11 — DO NOT refactor these systems | none |

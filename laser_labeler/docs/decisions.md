# Builder decisions

Choices made where the spec was silent. Change freely; each is small.

## Phase 1

- **Token rule.** Taxonomy entries (objects, materials, colors, incidence) must match `^[a-z0-9]+$`. Uppercase is refused too, so class names stay case-stable. Laser events additionally allow single `-` between words (`first-incidence`).
- **Color placeholder.** `na` is always accepted as a color and does not need to be in `colors.yaml`. It is configurable (`labeling.color_placeholder`).
- **Grayscale.** `coco_export` forces the placeholder color on grayscale images, whatever the box says.
- **New objects.** `add_entry(kind="object")` adds one object name (superseded: sub-objects no longer exist, see the last section).
- **YAML comments.** Adding a taxonomy entry rewrites the file and drops comments.
- **COCO categories.** Ids start at 1, assigned in sorted class-name order (stable). Whether Roboflow wants an id-0 supergroup category is unverified (Phase 2).
- **COCO export scope.** Only `approved` and `edited` label sets are exported unless `only_exportable=False`. Boxes are clamped to the image; a box fully outside the image raises.
- **Image size.** `ImageLabelSet` gained `image_width` and `image_height` (not in spec section 12) because COCO needs them.
- **Table rule.** Implemented in `core/labeler/table_rule.py` with three modes: `only_if_no_object_hit` (default), `always_label`, `never_label`. "Hit" means the object box lists at least one laser id.
- **Secrets.** Stored in the OS keyring under service `laser_labeler`. `save_settings` refuses any key whose name contains `api_key`, `secret`, `password` or `token`. If no keyring backend exists (some headless Linux), the app raises a clear error instead of falling back to a plaintext file.
- **Dependencies.** Core has only `pyyaml` and `keyring`. Roboflow, OpenCV, provider SDKs, reportlab and PyQt6 are optional extras installed in the phase that needs them.

## Sign-in, Phase 2 and Phase 3

- **Keys.** Nothing is built in; every key is provided by the user. Storage order: OS keyring, then an environment variable (`ROBOFLOW_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`/`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`). The keyring wins. The OpenAI-compatible service is the only one not marked required, because local servers need no key. Entry points: the Sign in tab (`python app.py`), or `python -m cli credentials set|clear|status|verify`. The CLI reads keys with hidden input, never from arguments.
- **Roboflow uses REST, not the `roboflow` package.** The docs contradict themselves on SDK argument names, and REST can be tested offline. Nothing has been run against the live service; `scripts/roboflow_check.py` is the live test (read-only unless `--upload-test` is given).
- **Coordinates.** The AI sees a downscaled image (`provider.max_image_px`). Every coordinate that crosses the AI boundary is in the shown image's pixels; all stored boxes and measurements are in original pixels.
- **Intensity** is the brightest channel, 0..1. `_norm` = (value - background) / (1 - background). Background is the image median.
- **scatter_ratio** = ring / (ring + core) energy, kept inside 0..1. A clean Gaussian spot already reads about 0.45, so compare spots against each other, not against zero. This is a first-draft definition (spec section 17).
- **A box is only measured as a spot** if its peak rises 6 noise levels above background; otherwise confidence is 0 and all numbers are 0.
- **Spot detection** (laser_detect): adaptive threshold, local maxima, growth to the half-maximum core, which is raised in steps when the spot sits on a bright object or beam. Long thin cores (elongation > 6) are beams, found separately with a Hough search. Everything is untuned on real images.
- **Pipeline mode** makes one model call per image: overview with every candidate boxed, plus a crop per candidate. At most `run.max_candidates` (6) candidates per image. The model never draws boxes in this mode.
- **Object boxes**: tightened by the app in pipeline mode; in agent mode the model's own boxes are kept (it has the `tighten_box` tool).
- **Color** always comes from pixels when the pixel match is confident enough (`run.color_confidence`); otherwise the model's value if it is in the taxonomy; otherwise a question. Grayscale images always get the placeholder.
- **Agent mode falls back to pipeline mode** when the provider has no tool support, the model answers without calling a tool, 4 tool calls are invalid, there is no final answer in `run.max_agent_steps`, or the final JSON stays invalid after `run.max_repairs` repairs. The fallback is recorded in the transcript.
- **Questions never block.** Any question (from the model, from taxonomy checks, from low confidence, or "no laser found") sets the status to `needs_input`. A question that points at a missing box keeps its text with no target rather than being dropped.
- **Provider or parse failure** returns a `failed` label set (sidecar still written) instead of raising. Hitting the spend cap does raise (`SpendCapExceeded`) so a whole run stops.
- **Prompt versions** are `draft-<hash>` until a numbered version is saved (Phase 4 and the Prompts tab).
- **Duplicate surface boxes**: two spots on the same material/color region (IoU >= 0.6, same attributes) merge into one object box listing both lasers.
- **Measurements CSV** keeps one row per laser box and replaces an image's rows when it is re-labeled.

## Structured output, Phase 4 (queue and storage)

- **Structured output** is opt-in (`provider.structured_output`, default false). Only the OpenAI-compatible adapter uses it: it sends `response_format=json_schema` (strict) when the flag is on, a schema is given and there are no tools. The schema (`core/labeler/schemas.py::pipeline_schema`) limits taxonomy fields to the allowed values plus "unknown". Gemini and Claude accept the argument and ignore it. If a server rejects the schema, turn the flag off.
- **Queue** is one SQLite file (`<data_dir>/queue.db`, WAL). Statuses: queued, labeling, needs_input, failed, pending, approved, edited, rejected, uploaded. Moves are validated (`ALLOWED` map); `uploaded` is final. One image is labeled at a time.
- **Crash recovery.** An image left in `labeling` goes back to `queued` when a run starts (`recover_interrupted`). Ctrl-C and the spend cap also re-queue the current image.
- **Runs stop themselves.** 5 failures in a row (`--max-failures`) stop the run and tell you to check key, model and server, so a bad setting does not mark 2,000 images failed. A single crash or bad image is recorded on that image and the run continues. A success resets the streak.
- **Confidence** for the queue is the weakest box (object boxes and laser boxes together). Images with no boxes read 0.
- **Bulk approve** (`queue approve --min-confidence X`, X is required, there is no default) only touches `pending` images with no open questions at or above X. Anything needing input is never bulk-approved.
- **Sidecar is the label store; the queue is the status.** Export and upload read each approved image's sidecar and take the status from the queue.
- **Manifest** (`<data_dir>/manifest.csv`): one row per image (image_id, image_name, pdf_path, sidecar_path, prompt_version, provider, model, status, labeled_at), written atomically and replaced on re-label.
- **Prompt versions** are immutable folders `prompts/versions/vNNNN` with a `version.json`. Saving identical content reuses the existing number; numbers are never reused after a delete. `ACTIVE` names the version runs use; with none, runs use the draft parts and report `draft-<hash>`. Version ids are checked against `^v\d{4}$` so no path can escape the folder.
- **Same file name in two folders** gets a short hash suffix on its image id; ids are chosen against both the queue and the current batch (a bug in the first version let the second file be silently dropped, caught by a test).
- **Roboflow download** (`queue add-roboflow`) skips images already queued, does not stop for one failed download, and keeps images whose label state Roboflow did not report (counted as "unknown label state"). The search request shape is unconfirmed live.
- **Upload from the CLI** is a dry run unless `--send`; only approved/edited images are included, and uploaded images are marked in the queue and manifest.

## Phases 5 to 8 (GUI, review, PDF and upload, prompt tools)

- **One shared `AppContext`** (settings path, data folder, queue database, review service, signals `settings_changed`, `queue_changed`, `prompt_changed`, `open_image`). Tabs never import each other; they talk through these signals. Background work (labeling, downloads, provider tests, prompt generation, uploads) always runs in a `QThread`; the GUI thread only shows results.
- **The queue holds status, the sidecar JSON holds boxes.** `ReviewService` is the only place that changes both, and keeps the manifest, measurements CSV, PDF, addenda and corrections in step. Editing a `pending` or `approved` image makes it `edited`; answering the last open question moves `needs_input` to `pending`. A hit-point answer adds a measured laser box snapped to a detected spot within 40 px, else a 15 px square. An answered "no laser found" question is dropped from the sidecar (the answer is kept in the transcript).
- **Approving** with a value that is not in the taxonomy asks first. Nothing is added to the taxonomy without a click ("Add this box's values to the taxonomy" or the checkbox in a question).
- **Bulk approve** only touches ready images without open questions at or above the threshold you set (no default). Its PDFs are built when you open the upload dialog.
- **PDF** is rebuilt from the saved conversation, the current labels and an addenda file, so an edit after approval adds a dated addendum instead of rewriting history. Text is limited to characters the built-in PDF fonts can draw.
- **Upload** (UI and CLI): only approved or edited images, dry run on by default, overwrite off by default, a real send asks to confirm, every send is written to `upload_audit.jsonl`, tags are `pdf:<name>` and `prompt:<version>` plus the image's own tags.
- **Re-run with a note** puts the image back in the queue with the note stored in `rerun_notes.json`; the labeler reads it once and adds it to the model prompt, and records it in the transcript.
- **Corrections** (`data/corrections.jsonl`): every attribute, box, added and removed change you make is kept once per image. Ticked ones are written into a marked block of `07_background.md` and are sent to Generate new prompt. They are never applied silently.
- **Generate new prompt**: the model gets the current parts, the old and new taxonomy (a snapshot stored with each version), ticked corrections and a few approved examples, and must answer with JSON naming whole replacement parts. The reply is checked (valid part names, size limit, placeholders kept, corrections markers kept, `{{taxonomy}}` still present) and up to `max_repairs` repair attempts are made. Nothing is written until you approve; approving saves a new version and makes it active. Spend goes through the same daily cap as labeling.
- **Hybrid escalation** is cloud only (Gemini or Claude), with its own prices and the same daily cap. A failed or unsure first result goes to the second model; a failed escalation never replaces a usable first answer; images with no laser are not escalated.
- **Run estimate** uses the seconds per image measured on your last run (10 s before there is one) and average tokens times the prices you entered; with no prices set it shows time only.
- **Keep awake** during a run is on by default and can be turned off (Windows, macOS and systemd are supported; it never fails the run).
- **GUI tests** run offscreen (`QT_QPA_PLATFORM=offscreen`) against a throwaway copy of the taxonomy and prompt parts, with a scripted model and fake Roboflow calls. Nothing in the test suite touches the network, the keyring or your real settings.
- **Not verified yet**: everything above has been run only against synthetic images and scripted models. Real laser images, real models, and the live Roboflow API (search request shape, annotation upload, class auto-creation, tags) are untested until you run them.

## Label scheme "components" (object + component1/2/3)

- **A setting, not a fork.** `labeling.scheme` is `full` (default) or `components`, chosen on the AI settings tab and read when an image is labeled. Nothing about existing labels changes when you switch.
- **Objects keep their full class names** (`obj_...`). Only laser boxes change: with the components scheme a laser box carries `component` (`component1`, `component2` or `component3`), and that is its Roboflow class. A box without a component keeps `las_<event>`, so both kinds can sit in one project and one export. `LaserBox.class_name` is the one place that decides.
- **Meaning of the parts** (fixed in `core/schema/class_names.py::COMPONENT_MEANING`): component1 = the bright core of the spot, component2 = the halo or glow around it, component3 = the visible beam line leading to it. These were chosen as the example reading of "parts of one laser spot"; if you mean something else, only that text and `core/labeler/components.py` change.
- **The app builds the parts, the model does not.** Pipeline and agent mode work exactly as before (the model says which candidates are lasers and what they hit). After measuring, `apply_components` marks each measured spot component1, adds a component2 box when the measured halo reaches at least 3 px beyond the core's own half-size, and attaches each detected beam line to the one spot it ends nearest to (within 40 px) as component3. A part that is not visible is not made, and nothing is asked about it.
- **Links and numbers.** A part points at its core with `from_id`, has the core's `order` and event as `kind`, and carries the core's confidence so the queue's weakest-box figure is not dragged down. Objects link to the core only. Halo and beam boxes are never re-measured as spots; the core is re-measured when you move it, and its parts are left as they are.
- **Review.** The box editor has a Component dropdown. A drawn laser box is a core under this scheme; a drawn component2 or component3 attaches to the nearest core (and needs one). Deleting a core removes its halo and beam; deleting a part leaves the core. Changes to the component show in the PDF addendum and the corrections list.
- **Beam detection is still the weak spot.** Beam lines come from the Hough search in `laser_detect`, which is untuned (on synthetic scenes with a bright object it can pick up object edges). Expect component3 boxes to need the most correcting in your first test. The unit tests cover the logic with hand-made lines, not real detections.

## Why images need input, spread-out lasers, visible conversation

- **Noise is local.** `image_stats.noise_level` is the robust spread of each pixel's difference from its 3x3 neighbourhood mean, not the spread of the whole image. The old whole-image figure was inflated by any textured or gradient scene, which pushed the "bright enough" bar so high that nothing counted as a laser (every image became "no laser found"). The background level is still the median.
- **Spread spots.** A candidate region is grown from a peak inside a window of max(60 px, 10% of the long side). Regions that touch the window edge are rejected unless at least half saturated (they are a bigger bright thing, not a spot). Nested lobes are merged into one spot (larger region, brightest peak). Maximum area rose from 2% to 5% of the image. A spot must exceed the median of a ring around its box by 6 noise levels, which rejects gradients and the sides of bright objects. Nothing here was tuned on real images; the tests use synthetic Gaussian spots of sigma 3 to 40 px, on gradients, on a bright object, speckled, clipped, and streaks.
- **The app says why it found nothing.** `Detection.explain()` states the brightest contrast against the noise bar, or counts of regions rejected as too small, too large, too stretched or not standing out. It is written into the transcript as a DETECTION event before any model call, so it exists even when no model is called. Each question is logged as a QUESTION event with its kind, and the run ends with a RESULT event.
- **Reasons come from question ids** (`core/review/reasons.py`): `q_no_laser`, `ql_` (confidence below 0.4), `qc_` (color), `qt_` (taxonomy value), `qi_` (incidence), anything else is a model question. The list row, the header and the Label run summary all use it.
- **The conversation is not a hidden tab.** It sits in a splitter under the image in the Review tab, so it is visible for every image.
- **Failed runs show a reason and a hint** (`core/providers/hints.py`). Model names pasted with quotes, backticks or an `ollama run` prefix are cleaned before use.
- **Test setup.** `tests/conftest.py` keeps one QApplication for the whole session. The container used for development had `opencv-python` and `opencv-contrib-python` installed beside the headless build; they load a second Qt (Qt5) into the process. After removing them, the full suite still crashed in the interpreter about once in four or five full runs (never in the GUI test files alone, never in a stress script of the same worker/signal pattern); the cause is not found. Treat a one-off segfault in a full run as a known flake and rerun; if it happens in the app itself, please report it.

## Bigger laser boxes

- Requested: laser boxes should cover roughly a third of the object they hit, not just the bright core. `core/labeler/laser_size.py` pads the box by 25% per side, then grows it to `labeling.laser_box_object_fraction` (default 0.33, capped at 0.5) of the object box that lists the laser, centred on the spot and clamped to the image. It runs in `finalize` after the object boxes exist, only for plain (non-component) boxes. I read "the band" as the object box the laser lands on; if you meant something else, only `grow_laser_boxes` changes.
- In pipeline mode the object box is the app's own estimate (the spot's surroundings tightened to edges), so the laser box inherits any error in it.

## Merged label scheme

- Requested as an alternative setting: one label holding both the object data and the laser component. Chosen with the user: the laser part box carries it, `<obj class>__<componentN>`; separate object boxes are not exported (unless no laser part lands on them, so no object information is lost).
- `labeling.scheme: merged` builds the parts exactly like `components` (core, halo, beam from the pixels). The merge happens when names are produced (`ImageLabelSet.laser_class_name`, used by COCO export, upload plan, CSV, PDF and golden scoring), not in the stored boxes, so Review edits to the object change the exported name and nothing is lost if you switch schemes. `ImageLabelSet.scheme` records the scheme each image was labeled with; old sidecars without it read as `full`.
- A halo or beam finds its object through its core (`from_id`). A part whose core hits no object keeps its plain `componentN` name.
- Golden scoring decodes merged names and also scores the object fields of matched parts, so a Roboflow export in this scheme can be compared like the others.

## Prompt bias toward "pen"

- The shipped prompt used a pen as its worked example, as the value in both output-format JSON templates, and as the first typical object in the background part. Small models copy examples, and the taxonomy has no entry for much of what the photos contain (3D-printed parts, tape, foam, cables), so "pen" became the nearest listed answer. The example is now a cup and the JSON templates use placeholders. New rule 6 and the questions part say to ask (with `allow_taxonomy_add`) instead of picking the closest-sounding entry. Expect more taxonomy questions at first; answering them with "add to taxonomy" is the intended way to grow the list.
- The prompt files are only the starting point: the active prompt is a saved version, so after updating, make a new version in the Prompts tab and activate it.

## No sub-objects

- Requested: "if it's a pen just say pen"; telling a ballpoint from a gel pen is not useful. An object class is now `obj_<object>_<material>_<color>_<incidence>`, `objects.yaml` is a plain list of names, the model is no longer asked for a sub-object (a reply that still has one is accepted and the value ignored), the Review editor has no sub-object field, and `add_entry` has no `parent_object`.
- The internal field is still called `general_object` (it means "the object"); only the sub-object field was removed.
- Compatibility: `decode_class` still reads the old `obj_<object>-<sub>_...` names and drops the sub-object, so ground truth exported from Roboflow earlier can be scored. Old sidecars with a `sub_object` key load (the key is dropped). An old `objects.yaml` mapping of object to sub-objects is still read, keeping the object names only. Saved prompt versions that stored the old taxonomy still diff correctly.
- Not migrated: classes already uploaded to Roboflow keep their old names (for example `obj_pen-ballpoint_...`). New uploads create new classes, so the two sets coexist in a project until you rename or delete the old ones there.

## Pausing on provider limits

- Before: a rate-limit or quota error was retried for about 15 seconds and then the image was marked failed, so a daily quota failed images one after another until five in a row stopped the run.
- Now `label_image_unsaved(raise_on_limit=True)` (used only by the queue's label function) lets `RateLimitError` through. `run_queue` puts the image back in the queue, asks `on_limit(LimitInfo)` and carries on with the same image if it answers True; with no callback, or on False, the run ends with `stopped == "rate_limit"`. The Label run tab's `RunWorker` implements `on_limit` by emitting `rate_limited` and blocking until `resume_after_limit()` or `stop()`.
- Review re-runs, `compare` and `label` still get a failed result for a limit, as before: they are single, attended calls.
- The wait the provider asks for comes from its `Retry-After` header (OpenAI-compatible, Anthropic) or `retryDelay` (Gemini); when it gives none the advice text says so. The app never resumes by itself: it asks, as requested. Auto-resume would be a small addition to `_tick_limit`.
- The window's close handler now ends a running labeling thread (and any limit wait) before the queue database closes.

## Model name and prices per provider

- `provider.profiles` maps each provider to its last `model`, `price_in_per_mtok` and `price_out_per_mtok`. The top-level `provider.model` and prices stay the active ones, so the labeling code, cost tracker and CLI are unchanged. The tab stashes the boxes under the provider being left and loads the one being entered; on load the active top-level values overwrite that provider's profile (a hand or CLI edit wins). Settings from before this change keep their model under their current provider.
- Prices are remembered with the name on purpose: a model name and its price belong together, and carrying one provider's price to another would make the spend cap wrong.
- Not done: separate entries for a local server versus OpenAI versus OpenRouter inside the "OpenAI-compatible" slot. They share one model, server address and key slot (`credentials` has one openai_compat key). Splitting them would mean keyed profiles by server address and one credential per server.

## Components by hit order (supersedes the core / halo / beam parts above)

- `component1` = a direct hit (the beam lands on the surface straight from the laser; two direct hits on the same object are both `component1`). `component2` = any bounce: reflection, refraction, scatter, other, however many bounces. `COMPONENT_NAMES` is now only those two.
- The class follows the event kind (`core/labeler/components.py::component_for_kind`): `first_incidence` is component1, every other kind is component2. Every visible spot gets one of the two, so nothing is left unlabeled. The halo and beam boxes, the beam-ridge checks and `LaserBox.is_part` are gone. Every spot box now grows by the laser box size setting (before, parts were skipped).
- Old sidecars written with the core / halo / beam meaning have no `scheme_version`. On load, `_migrate_parts` (raw dict, before boxes are validated) drops their halo and beam boxes, moves questions about them to the core, and recomputes each remaining component from its kind. New sidecars carry `scheme_version: 2`.
- Prompt: `prompts/parts/02_schema_rules.md` now defines the event types. Make a new prompt version in the Prompts tab and activate it to use the wording.
- Classes already on Roboflow keep their old meaning: component2 and component3 from the earlier scheme meant halo and beam. Use a fresh project or relabel those images.
- `halo_radius_px` stays as a measurement (CSV, PDF); it is not a label.


## Delete a box, and the merged one-label name in Review

- Review: a box can be deleted with the Delete key, the "Delete selected box (Del)" button, or right-click in the box list. Deleting a laser box under "merged" hands its object back to being exported as its own box; deleting the object makes the spot fall back to a plain `componentN`.
- Under "merged" the box list and the image labels show the one exported name (`obj_pen_rubber_red_first__component1`); the object box it is folded into is marked "named inside its laser box's label".
- A laser box drawn in Review is linked to the smallest object box its centre is inside, so it gets the merged name too. A spot that lands on no object box keeps a plain `component1` / `component2`.

## Merged names for images labeled under another scheme

- An image keeps the scheme it was labeled with, so changing the setting to "merged" does not rename old images. Review now has a "Switch to '<scheme>' scheme" button that converts the open image to the current setting without calling the model (components come from each spot's event; under full they are removed). Pressing Apply saves it as an edit; the change shows in the PDF addendum as "label scheme X -> Y" and is not stored as a prompt correction.
- `ImageLabelSet.object_for_laser` falls back to the smallest object box the spot's centre is inside when no object lists the spot, so an unlinked spot still gets the merged name. A spot on no object box keeps a plain `component1` / `component2`.

## Local-model speed controls

- The pipeline sends one whole-image view plus one zoomed crop per candidate spot (up to `run.max_candidates`), so request size, and time on a small GPU, scale with image size, spot count and crop size. All three are now settings in AI settings (`provider.max_image_px`, `run.max_candidates`, new `run.crop_px`).
- `provider.disable_thinking` sends `reasoning_effort: "none"` on the OpenAI-compatible provider only when ticked (Ollama documents this field for its OpenAI endpoint; some other servers may reject it, which is why it is off by default).
- `provider.timeout_s` (default 300) is passed to the OpenAI client. A timeout used to count as a transient connection error and was retried up to five times; it is now a plain `ProviderError` ("Request timed out") that is not retried, with a hint listing the speed settings.
- Not measured: the token numbers in the README are estimates from the model's image patch size, not counts from a real run.

## Taxonomy match: object only by default

- What it was: after a run, every object box's object, material, color and incidence had to be in the YAML lists, and each miss became a "not in taxonomy" question (so the image needed input) and a Review warning on approval.
- Now `labeling.taxonomy_match` is `object` by default: only the object name is matched (`Taxonomy.check_object(only_object=True)`). Material and color are normalised to a legal token (`normalise_token`) and accepted; the pixel color still wins when it is confident. `all` restores the old behaviour. Incidence is still checked against its fixed list (`INCIDENCES`), which is a schema value, not a taxonomy choice.
- Cost: free-text materials and colors can multiply the Roboflow classes (every material/color/object combination is its own class name). The prompt now says to prefer the listed values. The lists are also still what colors are named from in the pixel check.
- Prompt wording changed (`02_schema_rules.md` rule 3): make a new prompt version in the Prompts tab and activate it.

## Taxonomy match: off mode, reuse before adding

- `labeling.taxonomy_match` is now `off` | `object` (default) | `all`. In `off` and `object` the model's object and material are first matched to an existing entry with `match_entry` (exact, singular, last word), so "Ballpoint pens" reuses `pen`.
- `off` asks nothing and adds a word that has no match: `finalize` records it in `ctx.taxonomy_added` and appends it to the in-memory lists (so the next image's prompt shows it); `label_image` writes a transcript event and calls `on_taxonomy_add`; `build_label_fn(taxonomy_dir=...)` turns that into `add_entry` (GUI and CLI pass the project taxonomy folder). Review approval adds any new object or material on the label (`ReviewService._add_new_words`).
- Colors are never auto-added: the color list is the palette the pixel color is named from, and an extra name without a reference color would be unusable there.
- Fixed in the same change: ticking "add to taxonomy" for a material while in `object` mode used to be skipped; an explicit tick is always honoured now.
- Risk: a wrong model answer ("spaceship") is saved onto the lists. The conversation shows every addition so it can be removed from the YAML.

## Object box must hold its laser spot

- Seen in a real Review screenshot: the object box sat in the dark floor area away from the spot, and the laser box was then grown from that oversized object width. Pipeline mode cannot produce such a box (its edge search always starts from the spot), so it most likely came from agent mode, where the model's own coordinates are used as given (and Gemini/Qwen tend to answer in a normalised 0-1000 system, which the prompt does not ask for).
- `finalize.replace_misplaced_objects`: if none of the laser spots an object lists has its centre inside the object box (plus a 25% margin), the app replaces the box with its own edge search around the spot (`tighten_box`), sets confidence to at most 0.3 and adds a note, so the low-confidence question sends it to Needs input. Boxes that hold their spot are untouched.
- This repairs the symptom in every mode; it does not fix agent-mode coordinates. Pipeline mode is the recommended mode for Gemini and local models.

## Letting the model name new objects

- Why runs "stubbornly" never made new objects: the prompt said an object must be on the list and to ask instead of naming one, and with structured output the reply schema allowed only the listed objects (plus "unknown"), so the model physically could not name a new one.
- Now, when `labeling.taxonomy_match` is `off` or `object`: the pipeline schema leaves object and material as free text, and `core/prompts/manager.py` appends a "Naming objects" rule to the end of the system prompt that overrides the earlier lines (it is appended in code, so saved prompt versions get it too). With `all`, the old strict behaviour stays.
- "unknown" and similar (`NOT_A_NAME`) are never added to the lists. In `off` mode an object the model could not name still raises one question ("Type its name"); in `object` mode the normal not-in-taxonomy question covers it.
- Review has a "Clear 'not in taxonomy' questions..." button (`ReviewService.clear_taxonomy_questions`) for images already stuck: it adds each named object and material to the lists, marks those `qt_` questions answered ("added to taxonomy") and returns images with no open question to `pending`. Questions about an unnamed ("unknown") object stay open.

## Upload only updates labels on existing Roboflow images

- The app has never uploaded image files. Every request in `core/roboflow/upload.py` is `POST /dataset/{project}/annotate/{image_id}` (labels), or the tags/metadata routes, addressed by the Roboflow image id of an image already in the project. A test (`test_only_annotate_and_tag_routes_are_used_and_never_an_image_upload`) fails if an image-upload route or image bytes ever appear.
- Images downloaded from Roboflow keep their Roboflow id as the queue id, so their labels update the same image. Tick Overwrite in the upload dialog to replace labels an image already has; without it Roboflow answers 409.
- Images queued from a local file have no Roboflow id (their id is the file name), so they are now left out of the plan with a reason instead of being sent to a route that cannot find them. Matching local files to existing Roboflow images by name is not built (the search response shape is still UNCONFIRMED LIVE, see verified_facts row 11).
- `addToDataset` is still left at Roboflow's default (true); the docs say that moves an image out of its batch/job/review flow. If images change place in Roboflow after an upload, that is the setting to change.

## Truncated replies are reported, not repaired

- A reply whose `stop_reason` is a length stop (`length`, `max_tokens`, `max_output_tokens`) used to fail JSON parsing, and the repair loop then appended the cut-off text plus a correction request, making the request longer still. `repair_loop` now raises `ReplyTruncated` straight away ("The model ran out of tokens: its reply was cut off after N tokens, with the request at M tokens"), the transcript gets a `truncated` event, and the image is marked failed with a hint (`failure_hint`): shrink image size, spots and crop first, because a bigger window costs video memory; a large reply token count means the model is thinking or rambling.

## Auto-fit the request to a small context window

- Ollama's 4096-token window kept rejecting requests ("request (4948 tokens) exceeds the available context size (4096 tokens)"), and a 6 GB card cannot simply be given a bigger window. `openai_compat.map_error` now turns that error (Ollama/llama.cpp and OpenAI wording) into `ContextTooLong(request_tokens, window)`.
- `label_image_unsaved` catches it and retries with a smaller request (`core/labeler/fit.py`: crops and spot count first, then the whole-image size, down to 384 px / 1 spot / 128 px crops), measured from what was really sent so a small image or few candidates do not waste retries. A fresh conversation is started for each attempt. The size that fitted is kept in a `FitMemory` for the rest of the run (`build_label_fn`), so only the first image pays; the escalation (cloud) call never uses it.
- The result's notes say what was shrunk, and the transcript has a `fit` event. Setting `run.auto_fit_context: false` (AI settings checkbox) turns it off and restores the plain failure with its hint.
- Cost: a shrunk request means a smaller whole image and fewer or smaller crops, so labels from a run that had to shrink may be a little less accurate. The notes mark those images.

## Queue delete, passing over stalled images, auto-resume

- `QueueDB.remove(ids)` and `QueueDB.clear(statuses)` delete queue rows only; image files, sidecars and Roboflow are untouched, and `uploaded` rows are kept unless named. The Label run tab has a "Clear queue..." menu (waiting, failed, everything not uploaded; the last one warns how many images hold unreviewed work) and is disabled during a run; the CLI has `queue clear [--status S] [--yes]`.
- A timeout or a "ran out of tokens" failure already marked one image failed and let the run go on, but a streak of five failures stopped it, so three slow images in a row ended a run that could have continued. `runner.failure_kind` now separates "timeout", "image" (cut off) and "setup": only setup failures count toward `max_consecutive_failures`; timeouts have their own looser `max_consecutive_timeouts` (10, stop reason `too_many_timeouts`), since a hung server times out on every image; "ran out of tokens" never counts. The failure panel says in words that the image was marked failed and the run moved on.
- Messages: the truncation message now sums request + reply (the user could not tell that 6161 + 2031 is exactly an 8192 window), and `failure_hint` repeats the arithmetic, blames a long reply when it is over 1,500 tokens, and warns that a 16k window usually makes timeouts worse. The timeout message includes the seconds waited.
- `provider.reply_limit` (OpenAI-compatible only, default 0 = off, sent as `max_tokens`): a model that thinks for 2,000 tokens now fails fast. It is opt-in because some OpenAI models reject `max_tokens`. The setting name avoids the word "token" because the settings loader refuses any key whose name looks like a secret.
- Auto-resume (`run.auto_resume_on_limit`, default on): `core/labeler/limits.py` decides the wait (provider hint plus 3 s, capped at 6 h; otherwise 1, 2, 5, 10, 20, then 30 minutes) and `RunWorker._wait_for_limit` waits that long, then returns True so `run_queue` retries the same image. `LimitInfo.consecutive` counts hits with no image finished in between; after 30 the worker stops waiting and asks as before. Resume now and Stop still work during the wait. CLI: `queue run --wait-on-limit` (off by default so non-interactive runs keep stopping with exit code 3).

## "Unlabeled" means not labeled in the current system

- Before, an image counted as labeled if Roboflow listed any class on it, so images labeled under an older scheme were hidden from "Unlabeled only". `core/schema/scheme_labels.py` now decides per scheme: CURRENT needs at least one laser-side label written the way the scheme writes it (`las_*` for full; `component1/2` for components; merged name or bare `component1/2` for merged). Objects alone are OLDER because they are written the same in every scheme; a `component3`+ (first components scheme) forces OLDER even next to `component1`. States: none, current, older, unknown.
- Known limit: an image legitimately labeled "no laser, objects only" looks OLDER, and one labeled in the first components scheme with only component1/2 looks CURRENT. Images this app uploaded stay in the queue table (status `uploaded`), so the "hide images already in the queue" filter covers the first case.
- The Select images tab uses it (radio "Needs labeling", marks `*` `~` `?`, default after a Roboflow load, follows scheme changes without a reload) and asks before queueing an image that is already CURRENT (skip, label again, cancel). Older-style images are queued normally; replacing their labels on Roboflow needs the existing Overwrite box, and the tab says so. The CLI's `add-roboflow --unlabeled` passes the saved scheme to `list_images(scheme=...)`. This relies on Roboflow's `labels` field listing class names (docs/verified_facts.md row 2, still unconfirmed live).

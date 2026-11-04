# Builder decisions

Choices made where the spec was silent. Change freely; each is small.

## Phase 1

- **Token rule.** Taxonomy entries (objects, sub-objects, materials, colors, incidence) must match `^[a-z0-9]+$`. Uppercase is refused too, so class names stay case-stable. Laser events additionally allow single `-` between words (`first-incidence`).
- **Color placeholder.** `na` is always accepted as a color and does not need to be in `colors.yaml`. It is configurable (`labeling.color_placeholder`).
- **Grayscale.** `coco_export` forces the placeholder color on grayscale images, whatever the box says.
- **New objects.** An object cannot exist without a sub-object, so `add_entry(kind="sub_object", parent_object=...)` creates the parent if needed. `kind="object"` is refused with an explanation.
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

# Laser Labeler

Desktop app that proposes bounding-box labels for laser images, lets you review them, and uploads approved labels to Roboflow. See the build spec for the full design.

## Status

| Phase | State |
| --- | --- |
| 1. Foundations (models, schema, settings, secrets) | done, tested |
| Sign-in (keys for Roboflow and AI providers) | done: Sign in tab and CLI, tested |
| 2. Roboflow connection | code and offline tests done; **not yet run against your real project** |
| 3. Single-image labeler (CLI) | done, tested with a scripted fake model; **not yet tuned on real images or real models** |
| 4. Queue and storage (resumable queue, manifest, prompt versions, `queue` CLI) | done, tested; same caveat: **not run on real images yet** |
| 5. GUI: Select images, Label run, AI settings, status bar | done, tested offscreen (scripted model); **not yet used with a real model and real images** |
| 6. Review queue (edit boxes and attributes, measurements, approve/reject, bulk approve, questions, thumbnails) | done, tested offscreen |
| 7. Conversation PDF (rebuilt with an addendum after edits) and upload with tags and audit log | done, tested; upload is a dry run until you untick it, and **has not run against the live Roboflow project** |
| 8. Prompts and taxonomy (editor, versions, diff, Generate new prompt, corrections as examples) | done, tested with a scripted model |

## Using the app (python app.py)

Tabs, left to right:

1. **Sign in**: Roboflow key and project, plus AI provider keys. Keys go to your OS keyring only.
2. **Select images**: pick a folder on this computer, or load the list from your Roboflow project. Filter (all or unlabeled, split, tag, name), tick images or tick the next N, then **Add ticked images to the queue**. If Roboflow does not say whether an image has labels, you are asked whether to include it.
3. **Label run**: **Start** shows the model, the image count, the estimated time and cost, and asks first. Pause and Stop finish the image in progress. The image and the live conversation with the model are shown. Questions never stop the run.
4. **Review queue**: least confident first. Move, resize, draw or delete boxes, change attributes from the taxonomy lists, edit laser measurements, answer questions (a "where does the beam hit" question is answered by clicking the image). **Approve (A)**, **Reject (R)**, **Next (N)**, **Previous (P)**, **E** toggles box editing, **Ctrl+S** applies edits. **Approve all ready images at or above** is a bulk approve. **Upload approved...** opens a dry run first.
5. **Prompts and taxonomy**: edit prompt parts, save numbered versions, compare versions, edit the taxonomy lists, tick corrections to feed back, and **Generate new prompt** (the model drafts, you read the side-by-side differences and approve or reject).
6. **AI settings**: provider, model, run mode, prices, spend cap, table rule, optional cloud escalation, and **Test provider**.

**Label scheme** (AI settings tab, applies to images labeled from then on):

- *Full* (default): object classes `obj_<object>_<material>_<color>_<incidence>` (one object word, for example `obj_pen_rubber_red_first`; there is no sub-object) and laser classes `las_<event>`.
- *Components*: the same object classes, but each laser spot is split into its parts, which become the classes `component1` (the bright core), `component2` (the halo or glow around it) and `component3` (the visible beam line leading to it). The app finds the parts from the pixels, so a part it cannot see is left out; you can change a box's component, move it, draw one or delete it in the Review queue. Old and new images can sit in the same Roboflow project: each box is exported with its own class.

**Merged scheme** (AI settings, "Label scheme" = Merged): one label carries both the object and the laser part. Each core / halo / beam box is named after the object it lands on, for example `obj_pen_rubber_red_first__component1`, and the object's own box is not sent separately (an object that no laser part lands on is still sent as a normal `obj_...` box). The two halves are joined by `__`. The names are built when you export or upload, so if you change an object's material or color in the Review queue the part names follow it. Images keep the scheme they were labeled with (shown in the sidecar as `scheme`). Expect many more classes, one per object description times three parts, so you need more images per class.

**Laser box size** (AI settings tab, "Laser box size"; `labeling.laser_box_object_fraction`, default 0.33): a plain laser box is padded by a quarter on each side so the glow is inside it, then grown until each side is that fraction of the object box it hits (never more than half of it, never outside the image). Set it to 0 for a tight box. The component boxes (component1/2/3) are not resized. Measurements are taken before the box grows. If you move or resize a laser box in the Review queue it is re-measured from the box you leave, so keep it around the spot.

**Why an image "needs your input"**: the reason is shown in four places, so you never have to guess.

- Review queue list: each row under the filter *Needs your input* has a line `why: no laser found` (or `low confidence`, `color unclear`, `taxonomy value`, `incidence`, `model question`).
- Review queue header: **Needs your input:** followed by the same reasons.
- Review queue **Conversation panel** (always visible under the image, for every image, including ones where no model was called): first the app's own **DETECTION** finding in plain words (for example "brightest place is 11 levels above the background (noise 2.6), but a spot must rise at least 6 noise levels (16 levels) above it"), then the model's messages, then one line per **QUESTION** and a final **RESULT**.
- Label run tab: after a run, "Waiting for you: 9 x no laser found, 3 x low confidence", and a failed image shows its error with a hint.

**Spread-out lasers**: the spot does not have to be a tight dot. The detector grows each bright region from its peak, accepts soft round or oval glows up to 5% of the image, merges nested lobes of one glow into one spot, and judges "bright enough" against local pixel-to-pixel noise (so a textured scene no longer hides the laser). A candidate must stand out from a ring of pixels around it, so gradients and bright objects are not mistaken for a laser. Thresholds are constants at the top of `core/tools/laser_detect.py` (`MIN_NOISES`, `NEST_PEAK_RATIO`).

The status bar shows the Roboflow project, the active prompt version, the provider and model, and today's spend.

## First steps on your machine

```bash
pip install -r requirements.txt                    # (or: pip install -e ".[providers,gui,dev]")
python app.py                                       # sign in: enter your keys
python -m cli credentials status                    # or set keys from the command line
python scripts/roboflow_check.py --workspace W --project P   # read-only check; paste the output back
python -m cli detect some_image.png                 # laser detection and numbers, no AI
python -m cli test-provider                         # does your model return JSON / call tools?
python -m cli label some_image.png --mode pipeline  # label one image (nothing is uploaded)
python -m cli camera-check folder_of_images --camera-from folder
python -m cli compare --images golden/ --truth golden.coco.json --runs runs.yaml --out report.md
```

### Labeling a batch (queue)

```bash
python -m cli queue add images_folder --tag camera:logitech   # local files (subfolders included)
python -m cli queue add-roboflow --unlabeled --limit 200       # or pull unlabeled images from Roboflow
python -m cli queue run --limit 10                             # try 10 first; Ctrl-C is safe, it resumes
python -m cli queue status
python -m cli queue list --status pending --sort confidence_asc   # least sure first
python -m cli queue approve --min-confidence 0.9               # bulk-approve only the sure ones
python -m cli queue export-coco --out approved.coco.json
python -m cli queue upload --workspace W --project P           # DRY RUN; add --send to really upload
```

Everything lands in `data/`: `queue.db`, `sidecars/` (one JSON per image), `measurements.csv`, `manifest.csv`, `spend.json`, `upload_audit.jsonl`. `queue run` stops by itself after 5 failures in a row or when the daily spend cap is hit.

Set the provider and model in `data/user_settings.yaml` (copy the `provider:` block from `config/defaults.yaml`). Model names and prices are yours to fill in; none are built in.

## Run the tests

```bash
pip install -e ".[providers,gui,dev]"
python -m pytest
```

## Layout

- `core/models.py`: dataclasses everything shares
- `core/schema/`: class-name encode/decode, taxonomy load/validate/extend, COCO export
- `core/labeler/table_rule.py`: the table rule (config flag)
- `config/`: defaults, settings load/save, keyring secrets
- `taxonomy/`: editable YAML lists (starter values; replace with yours)
- `docs/decisions.md`, `docs/verified_facts.md`

API keys go in the OS keyring only, never in files.

**Troubleshooting**: if the app or the tests crash at random, check `pip list` for `opencv-python` or `opencv-contrib-python` next to `opencv-python-headless`. Only the headless one should be installed; the others bundle a second copy of Qt that clashes with PyQt6 (`pip uninstall opencv-python opencv-contrib-python`, then `pip install --force-reinstall --no-deps opencv-python-headless`).

**Ollama context size**: Ollama runs models with a 4096-token context unless told otherwise, and this app sends the whole image, up to six crops and the prompt, which can be more (error: "request exceeds the available context size"). Two fixes, pick one:

1. Set the environment variable `OLLAMA_CONTEXT_LENGTH=8192` (Windows: Start menu, "Edit the system environment variables", Environment Variables, New), then quit Ollama from the tray icon and start it again.
2. Or make a copy of the model with a bigger window. Create a file named `Modelfile` containing `FROM qwen3-vl:4b` and `PARAMETER num_ctx 8192`, run `ollama create qwen3-vl-8k -f Modelfile`, and use `qwen3-vl-8k` as the model name in AI settings.

A bigger window uses more video memory. On a 6 GB card also lower Max candidates (3) and the image size (768) in AI settings.


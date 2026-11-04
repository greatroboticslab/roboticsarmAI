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

- *Full* (default): object classes `obj_<object>-<sub>_<material>_<color>_<incidence>` and laser classes `las_<event>`.
- *Components*: the same object classes, but each laser spot is split into its parts, which become the classes `component1` (the bright core), `component2` (the halo or glow around it) and `component3` (the visible beam line leading to it). The app finds the parts from the pixels, so a part it cannot see is left out; you can change a box's component, move it, draw one or delete it in the Review queue. Old and new images can sit in the same Roboflow project: each box is exported with its own class.

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


https://youtu.be/Jo7P1j3Wrhw


**Video link:** https://youtu.be/Jo7P1j3Wrhw ·

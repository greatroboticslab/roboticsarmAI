# Object Labeling Studio

A separate, self-contained GUI tool for turning captured/imported photos into
Roboflow-ready labels with Gemini's help — kept apart from the main app's
`main.py` on purpose, so this can evolve on its own schedule.

## Run it

```
pip install -r requirements.txt          # this tool's own extra dependency (fpdf2)
pip install -r ../requirements.txt       # the shared data-layer dependencies (pymongo, opencv, Pillow, requests...)
python app.py
```

Works from any working directory — `config.py` adds the repo root to
`sys.path` itself.

## What's shared with the main app, and what isn't

This tool imports `vision.storage.mongo_client`, `vision.storage.
roboflow_export`, `vision.storage.attribute_schema`, `vision.storage.
session_manager`, and `vision.storage.package_export` from the main repo —
the shared Mongo/Roboflow data layer both programs need to agree on, so a
capture from the robot arm and an object grouped here look identical to
either program.

Everything under `core/` and `screens/` belongs to this tool alone —
`gemini_client.py` and `laser_dot.py` used to live in the main app's
`vision/services/` and were moved here, since they're specific to this
workflow, not general app infrastructure.

## The five screens

1. **Import** — three tabs: a generic filesystem folder (best-effort
   inference, always shown as an editable preview before committing), a Data
   Package folder from the main app's own export feature, and Roboflow's
   unlabeled backlog (pulls images that have no annotation yet, so Gemini
   can suggest a label before you box them by hand).
2. **Group** — list objects (filterable by Unlabeled / Already labeled /
   All), multi-select several and merge them into one object.
3. **Review** — pick an object, run Gemini, edit the result, then split it
   into one record per material+color combination it identified.
4. **Annotate** — two tabs: Auto-Annotate (laser-dot detection + one-click
   box+label upload to Roboflow, for capture-rig photos) and Label Only
   (pushes Gemini's suggestion into Roboflow metadata for images with no
   laser dot, so you box them by hand with the suggestion already visible).
5. **Archive** — browse every record, open its backing PDF, edit its
   (manually-obtained) Gemini share code, or backfill an old,
   manually-shared conversation by pasting its text in.

## Credentials

Session-only, never written to disk — sign in again each time you run this,
same as the main app's own Roboflow/Gemini panels.

## Storage

- Metadata: a new `gemini_records` Mongo collection (via `mongo_client`).
- PDFs: local disk, under the storage root's `gemini_records/<object_id>/`
  folder (see `vision.storage.storage_location.gemini_records_root()`).
- Roboflow cannot store the PDF itself — there's no file-attachment
  capability in their API or Asset Library (checked before building this).
  Roboflow's own per-image metadata instead gets a small pointer (record
  ids + a one-line summary) back to the real detail living here.

## What this does NOT do

- Does not scrape a Gemini share link's page content — there's no official
  API for that, and the unofficial scrapers that exist need a full headless
  browser and aren't reliable enough to build on. The Archive screen's
  "Paste Conversation" import is the deliberate, ToS-safe alternative.
- Does not auto-draw a box on an image with no laser dot — see the Annotate
  screen's two branches for why that's a real split, not an oversight.

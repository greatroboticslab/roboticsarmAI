# Laser Labeler

Desktop app that proposes bounding-box labels for laser images, lets you review them, and uploads approved labels to Roboflow. See the build spec for the full design.

## Video walkthrough

<!-- Paste the link to your YouTube video below, replacing the placeholder. To show a clickable thumbnail instead,
     use: [![Laser Labeler walkthrough](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID) -->

Watch the walkthrough: **YOUR_YOUTUBE_LINK_HERE**

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
2. **Select images**: pick a folder on this computer, or load the list from your Roboflow project. Filter (all, or *Needs labeling*, which means not labeled in your current label scheme; split, tag, name), tick images or tick the next N, then **Add ticked images to the queue**. If Roboflow does not say whether an image has labels, you are asked whether to include it.
3. **Label run**: **Start** shows the model, the image count, the estimated time and cost, and asks first. **Clear queue...** removes waiting, failed, or all not-yet-uploaded images from the queue (image files and saved results stay on disk; nothing changes on Roboflow). Pause and Stop finish the image in progress. The image and the live conversation with the model are shown. Questions never stop the run.
4. **Review queue**: least confident first. Move, resize, draw or delete boxes, change attributes from the taxonomy lists, edit laser measurements, answer questions (a "where does the beam hit" question is answered by clicking the image). **Approve (A)**, **Reject (R)**, **Next (N)**, **Previous (P)**, **E** toggles box editing, **Ctrl+S** applies edits. **Approve all ready images at or above** is a bulk approve. **Upload approved...** opens a dry run first.
5. **Prompts and taxonomy**: edit prompt parts, save numbered versions, compare versions, edit the taxonomy lists, tick corrections to feed back, and **Generate new prompt** (the model drafts, you read the side-by-side differences and approve or reject).
6. **AI settings**: provider, model, run mode, prices, spend cap, table rule, optional cloud escalation, and **Test provider**.

**Label scheme** (AI settings tab, applies to images labeled from then on):

- *Full* (default): object classes `obj_<object>_<material>_<color>_<incidence>` (one object word, for example `obj_pen_rubber_red_first`; there is no sub-object) and laser classes `las_<event>`.
- *Components*: the same object classes, but each laser spot becomes `component1` (a direct hit: the beam lands on the object; two direct hits on the same object are both `component1`) or `component2` (a bounce: light reflected, refracted or scattered off something, however many bounces). The class comes from the event the model gave the spot (`first_incidence` is component1, everything else is component2), so every visible spot is labeled and there are no halo or beam boxes. In the Review queue, changing a box's event changes its component. The `halo_radius_px` number in the measurements is still computed; it is a measurement, not a label. Old and new images can sit in the same Roboflow project: each box is exported with its own class.

**Merged scheme** (AI settings, "Label scheme" = Merged): one label carries both the object and the laser part. Each laser spot is named after the object it lands on, for example `obj_pen_rubber_red_first__component1`, and the object's own box is not sent separately (an object that no laser spot lands on is still sent as a normal `obj_...` box). The two halves are joined by `__`. The names are built when you export or upload, so if you change an object's material or color in the Review queue the part names follow it. Images keep the scheme they were labeled with (shown in the sidecar as `scheme`). Expect many more classes, one per object description times two, so you need more images per class.

**When the provider's limit is reached** (rate limit or daily quota, for example Gemini's free tier): the app first retries a few times on its own, then pauses the run instead of failing images. The image goes back in the queue and a yellow bar on the Label run tab says why. With **auto-resume** on (AI settings, "When the provider's limit is reached, wait and carry on by itself"; `run.auto_resume_on_limit`, default on) the run waits as long as the provider says (or tries again after 1, 2, 5, 10, 20, then every 30 minutes when it gives no time), carries on by itself when the limit reopens, and shows when it will try again; **Resume now** goes sooner and **Stop the run** ends it. So a queue can run overnight and pick up when a daily quota resets (keep the computer awake; "Keep the computer awake" is on by default). After 30 limit hits in a row with no image finished it stops waiting and asks you. With auto-resume off it just waits for you to press Resume now. Nothing is ever marked failed by a limit. If you close the app while it waits, the run ends and the image stays queued, so the next run picks it up. On the command line (`python -m cli queue run`) it prints the same advice and asks you to press Enter to resume, or q to stop; add `--wait-on-limit` to wait and carry on by itself, and without a keyboard (and without that flag) it stops and exits with code 3. The daily spend cap is separate and still stops the run. A limit on the escalation model does not pause anything: the first model's answer is kept.

**A stalled or oversized image is passed over, not a reason to stop.** A request that times out, or whose reply is cut off ("ran out of tokens"), marks that one image *failed* and the run moves straight on to the next; the failure panel on the Label run tab says so, and **Retry failed images** tries them again later. The run only stops early after 5 failures in a row of the kind that means a wrong setup (key, model name, server), or after 10 timeouts in a row (a hung server). A "ran out of tokens" message now shows the arithmetic, for example `request 6183 + reply 2009 = 8192 tokens`: when that total equals your context window, the reply ran into the end of it, and a reply of 1,500 tokens or more means the model is thinking or rambling rather than the window being too small.

**Only label what needs it** (Select images tab, Roboflow source): an image counts as *labeled in the current system* when Roboflow shows a laser label written the way your current label scheme writes it (Full: `las_<event>`; Components: `component1` or `component2`; Merged: `<object class>__component1/2`, or a bare `component1/2`). Object labels alone do not count (every scheme writes them the same way), and a leftover `component3` marks the first components scheme (core/halo/beam parts), so such images count as older. Choose **Needs labeling** to see images with no labels, with labels of an older kind, or of unknown state (this is the default after loading from Roboflow); marks in the grid: `*` labeled in the current system, `~` has labels but only older ones, `?` unknown. If you tick an image that is already labeled in the current system, the app asks and offers to skip it. Images with older-style labels are labeled again, and uploading replaces their labels only if you tick *Overwrite* in the upload window (the app tells you how many). Queue them, press Start, and with auto-resume on the run works through them whenever the API has quota. On the command line `queue add-roboflow --unlabeled` uses the same rule and your saved scheme. Roboflow reports an image saved as "no objects" the same as an unlabeled one, so those appear in the list too.

**Taxonomy match** (AI settings, Labeling rules): your `taxonomy/*.yaml` lists are what the model is told to use. Three modes:
- *Off*: nothing is asked. If the model's object or material matches a listed word (case, plurals and "ballpoint pen" -> `pen` are handled) that word is used; otherwise the new word is added to your lists automatically (during the run, and on approval for words you typed in Review) and the conversation records "added object 'x' to your taxonomy lists". Good while you are filling the lists up. Check `taxonomy/objects.yaml` afterwards for junk.
- *Object only* (default): a listed object is reused; an object that is not on the list becomes a question ("add it to the taxonomy" or pick another) and the image goes to Needs input. Material and color are reused if listed, otherwise taken as the model says them.
- *Everything*: object, material, color and incidence must all be on the lists, as in the first version.
In every mode free text is made a legal class-name word (lowercase letters and digits, so "Dark Blue" becomes `darkblue`), colors are still measured from the pixels first, and new colors are never added to the lists (the color list is the palette the pixel color is named from). Ticking "add to taxonomy" in Review always adds. Fewer distinct words means fewer Roboflow classes, so prefer the listed words where you can.

**One model per provider**: the AI settings tab remembers the model name, and the two token prices, separately for each provider (OpenAI-compatible, Gemini, Claude). Pick a provider in the list and its last model and prices come back; a provider you have not used yet starts empty, so Gemini's prices are never applied to a local model. Save settings stores them (`provider.profiles` in your settings file; no keys are stored there). Note that "OpenAI-compatible" is one slot: a local Ollama server and OpenAI itself share its single model name, server address and key.

**Laser box size** (AI settings tab, "Laser box size"; `labeling.laser_box_object_fraction`, default 0.33): a plain laser box is padded by a quarter on each side so the glow is inside it, then grown until each side is that fraction of the object box it hits (never more than half of it, never outside the image). Set it to 0 for a tight box. The component boxes (component1/2/3) are not resized. Measurements are taken before the box grows. If you move or resize a laser box in the Review queue it is re-measured from the box you leave, so keep it around the spot.

**Why an image "needs your input"**: the reason is shown in four places, so you never have to guess.

- Review queue list: each row under the filter *Needs your input* has a line `why: no laser found` (or `low confidence`, `color unclear`, `taxonomy value`, `incidence`, `model question`).
- Review queue header: **Needs your input:** followed by the same reasons.
- Review queue **Conversation panel** (always visible under the image, for every image, including ones where no model was called): first the app's own **DETECTION** finding in plain words (for example "brightest place is 11 levels above the background (noise 2.6), but a spot must rise at least 6 noise levels (16 levels) above it"), then the model's messages, then one line per **QUESTION** and a final **RESULT**.
- Label run tab: after a run, "Waiting for you: 9 x no laser found, 3 x low confidence", and a failed image shows its error with a hint.

**Spread-out lasers**: the spot does not have to be a tight dot. The detector grows each bright region from its peak, accepts soft round or oval glows up to 5% of the image, merges nested lobes of one glow into one spot, and judges "bright enough" against local pixel-to-pixel noise (so a textured scene no longer hides the laser). A candidate must stand out from a ring of pixels around it, so gradients and bright objects are not mistaken for a laser. Thresholds are constants at the top of `core/tools/laser_detect.py` (`MIN_NOISES`, `NEST_PEAK_RATIO`).

The status bar shows the Roboflow project, the active prompt version, the provider and model, and today's spend.

**Marking where a laser is.** In Review, "Mark laser spots" lets you click as many places as you like. A click only tells the app where a laser is so it can check there; it is not the laser's box, size or event (the app measures the spot, the model decides what it is, and you can edit the box afterwards). Click a mark again to remove it. Marks are sent with "Re-run with a note..." (no note needed), or turned into laser boxes with "Add these as lasers" when you answer the "where does the beam first hit?" question.

**Box positions.** Photos are turned upright by their EXIF orientation before anything is measured, so boxes line up with what Roboflow shows. Labels made by an older version are moved onto the upright picture when they are opened or uploaded (images already uploaded need Overwrite to fix them). If a vision model answers in a different number system (a 0 to 1000 grid or 0 to 1 fractions) in agent mode, AI settings has "Box numbers from the model are" (default: work it out from the spots the app found).

**Boxes in the wrong place.** In agent mode, a laser box far from every spot the app found is moved onto the nearest spot, set to low confidence and flagged ("box moved onto a spot") so you check it. If a box is still wrong, select it and press "Fit selected box by hand": draw a new box and the label (object, material, event) stays as generated; only the box changes. The app also recognises a model that answered on a 0-1000 grid, in 0-1 fractions, or on a copy the server shrank, and converts it.

**Big lists.** Loading hundreds or thousands of Roboflow images and ticking them all no longer freezes the window (settings are cached, the grid lays out in batches and ticks everything in one update).

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
python -m cli queue add-roboflow --unlabeled --limit 200       # or pull images not yet labeled in your current scheme
python -m cli queue run --limit 10                             # try 10 first; Ctrl-C is safe, it resumes
python -m cli queue run --wait-on-limit                        # overnight: wait out a rate limit / daily quota and carry on
python -m cli queue clear --status failed --yes                # remove images from the queue (files stay on disk)
python -m cli queue status
python -m cli queue list --status pending --sort confidence_asc   # least sure first
python -m cli queue approve --min-confidence 0.9               # bulk-approve only the sure ones
python -m cli queue export-coco --out approved.coco.json
python -m cli queue upload --workspace W --project P           # DRY RUN; add --send to really upload
```

Everything lands in `data/`: `queue.db`, `sidecars/` (one JSON per image), `measurements.csv`, `manifest.csv`, `spend.json`, `upload_audit.jsonl`. `queue run` stops by itself after 5 setup-type failures in a row (a stalled or cut-off image does not count; 10 timeouts in a row do) or when the daily spend cap is hit.

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

A bigger window uses more video memory. On a 6 GB card also lower the speed settings below.

**Making a local model faster** (AI settings, Main model). Each image is sent as one whole-image view plus one zoomed crop per candidate spot, so the request grows with these:
- *Largest image side sent*: the biggest saving. 1024 px costs roughly 1,000 image tokens; 640 px roughly 400.
- *Spots shown to the model per image* (`run.max_candidates`, default 6) and *Zoom crop size* (`run.crop_px`, default 384): fewer and smaller crops mean fewer image tokens and a faster answer.
- *Turn off thinking* (`provider.disable_thinking`): sends `reasoning_effort: "none"`, which Ollama uses to switch off the long thinking step of reasoning models such as qwen3. Leave it off for servers that reject the field.
- *Request timeout* (`provider.timeout_s`, default 300 s): a request that takes longer is given up on and the image is marked failed with a hint, instead of waiting forever; the run moves on to the next image. A timeout is not retried. 0 uses the server library default.
- *Longest reply allowed* (`provider.reply_limit`, default 0 = no limit): stops a reply after this many tokens. The answer the app needs is a few hundred tokens, so about 1200 makes a model that rambles or thinks at length fail fast (as "ran out of tokens") instead of stalling for minutes. It is sent as `max_tokens`; if your server complains about that field, set it back to no limit.
- A bigger context window is not a speed fix: 16k makes every request slower and pushes a 6 GB card onto the CPU. Keep it at the smallest size that fits (about 6144) and cut the long replies instead.
- While a request runs, `ollama ps` should show the model on 100% GPU. A CPU/GPU split means the model plus context is too big for the card; use a smaller model or a smaller context.


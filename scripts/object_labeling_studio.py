#!/usr/bin/env python3
"""
Object Labeling Studio — a small standalone companion tool (separate
from main.py, per request) for finishing Roboflow labeling with as
little manual work as possible:

    1. Auto-detects where the laser dot landed on the object's photo
       (vision.services.laser_dot) and shows it as a draggable box —
       click anywhere on the photo to correct it if the guess is wrong.
    2. Auto-calls Gemini (vision.services.gemini_material_query) for an
       object/material/labels suggestion on that same photo.
    3. Lets you review/edit both before confirming.
    4. On Confirm: saves the Gemini result (+ an optional manually-
       pasted Gemini share code) to the object's attributes in Mongo,
       AND uploads the confirmed box+label as a real annotation to
       Roboflow at the laser-indicated location.

Works on any scope — today's captures, a date range, or your full
history — so the SAME tool covers both "run this on today's new batch"
and "go back and finish labeling old objects," rather than needing two
separate programs.

REQUIREMENTS
------------
Objects must already be uploaded to Roboflow (via the main app's
Roboflow Export panel) before running this tool — annotation upload
needs the Roboflow image ID that upload already recorded. Objects not
yet uploaded are shown but can only be Gemini-labeled/share-coded here,
not annotated (that part is skipped with a note, not silently ignored).

CREDENTIALS
-----------
Session-only, in memory, never written to disk — same convention as
the main app's Roboflow/Gemini panels. Sign in again each run.

RUN
---
    python scripts/object_labeling_studio.py
(run from the repo root, or anywhere — it adds the repo root to
sys.path itself so `import vision...` works either way)
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

# Make `import vision...` work no matter where this script is launched
# from, since it lives in scripts/ alongside (not inside) the vision
# package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from PIL import Image, ImageTk
except ImportError:
    print("This tool needs Pillow: pip install Pillow", file=sys.stderr)
    sys.exit(1)

from vision.storage import mongo_client, roboflow_export, session_manager, attribute_schema
from vision.services import gemini_material_query, laser_dot

CANVAS_SIZE = (640, 480)
PROJECT_ROBOFLOW_KEY = {}  # filled in once signed in to Roboflow: "workspace/project"


class ObjectLabelingStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Object Labeling Studio")
        self.geometry("980x760")

        self.queue = []       # list of object_id, in review order
        self.queue_index = -1
        self.current_obj = None
        self.current_image_path = None
        self.current_pil_image = None   # full-res, for coordinate math
        self.current_photo = None       # Tk PhotoImage, canvas-displayed
        self.box_norm = None            # {"cx_norm","cy_norm","w_norm","h_norm"}
        self.roboflow_image_id = None   # for the currently-shown object's photo

        self._build_credentials_row()
        self._build_scope_row()
        self._build_review_area()
        self._build_nav_row()

    # ---------------------------------------------------------- credentials
    def _build_credentials_row(self):
        frame = tk.LabelFrame(self, text=" Sign In ", padx=8, pady=6)
        frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(frame, text="Roboflow API key:").grid(row=0, column=0, sticky=tk.W)
        self.rf_key_var = tk.StringVar()
        self.rf_key_entry = tk.Entry(frame, textvariable=self.rf_key_var, width=22, show="*")
        self.rf_key_entry.grid(row=0, column=1, padx=4)
        tk.Label(frame, text="Workspace:").grid(row=0, column=2, sticky=tk.W)
        self.rf_ws_var = tk.StringVar()
        self.rf_ws_entry = tk.Entry(frame, textvariable=self.rf_ws_var, width=14)
        self.rf_ws_entry.grid(row=0, column=3, padx=4)
        tk.Label(frame, text="Project:").grid(row=0, column=4, sticky=tk.W)
        self.rf_proj_var = tk.StringVar()
        self.rf_proj_entry = tk.Entry(frame, textvariable=self.rf_proj_var, width=20)
        self.rf_proj_entry.grid(row=0, column=5, padx=4)
        self.rf_signin_btn = tk.Button(frame, text="Sign In", command=self._do_roboflow_sign_in, bg="khaki")
        self.rf_signin_btn.grid(row=0, column=6, padx=(8, 2))
        self.rf_signout_btn = tk.Button(frame, text="Sign Out", command=self._do_roboflow_sign_out,
                                         state=tk.DISABLED)
        self.rf_signout_btn.grid(row=0, column=7)

        tk.Label(frame, text="Gemini API key:").grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        self.gm_key_var = tk.StringVar()
        self.gm_key_entry = tk.Entry(frame, textvariable=self.gm_key_var, width=22, show="*")
        self.gm_key_entry.grid(row=1, column=1, padx=4, pady=(4, 0))
        tk.Label(frame, text="Model:").grid(row=1, column=2, sticky=tk.W, pady=(4, 0))
        self.gm_model_var = tk.StringVar(value="gemini-2.5-flash")
        self.gm_model_entry = tk.Entry(frame, textvariable=self.gm_model_var, width=20)
        self.gm_model_entry.grid(row=1, column=3, padx=4, pady=(4, 0))
        self.gm_signin_btn = tk.Button(frame, text="Sign In", command=self._do_gemini_sign_in, bg="khaki")
        self.gm_signin_btn.grid(row=1, column=6, padx=(8, 2), pady=(4, 0))
        self.gm_signout_btn = tk.Button(frame, text="Sign Out", command=self._do_gemini_sign_out,
                                         state=tk.DISABLED)
        self.gm_signout_btn.grid(row=1, column=7, pady=(4, 0))

        self.cred_status = tk.Label(frame, text="Not signed in to either service yet.", fg="gray")
        self.cred_status.grid(row=2, column=0, columnspan=8, sticky=tk.W, pady=(4, 0))

    def _do_roboflow_sign_in(self):
        api_key, workspace, project = self.rf_key_var.get().strip(), self.rf_ws_var.get().strip(), self.rf_proj_var.get().strip()
        self.cred_status.config(text="Verifying Roboflow credentials...", fg="gray")

        def worker():
            ok, message = roboflow_export.sign_in(api_key, workspace, project)

            def apply():
                if ok:
                    self.rf_key_var.set("")
                    for w in (self.rf_key_entry, self.rf_ws_entry, self.rf_proj_entry, self.rf_signin_btn):
                        w.config(state=tk.DISABLED)
                    self.rf_signout_btn.config(state=tk.NORMAL)
                    self.cred_status.config(text=f"Roboflow: signed in to {workspace}/{project}.", fg="green")
                else:
                    self.cred_status.config(text=f"Roboflow: {message}", fg="red")
            self.after(0, apply)
        threading.Thread(target=worker, daemon=True).start()

    def _do_roboflow_sign_out(self):
        roboflow_export.sign_out()
        for w in (self.rf_key_entry, self.rf_ws_entry, self.rf_proj_entry, self.rf_signin_btn):
            w.config(state=tk.NORMAL)
        self.rf_signout_btn.config(state=tk.DISABLED)
        self.cred_status.config(text="Roboflow: signed out.", fg="blue")

    def _do_gemini_sign_in(self):
        api_key, model = self.gm_key_var.get().strip(), self.gm_model_var.get().strip()
        self.cred_status.config(text="Verifying Gemini credentials...", fg="gray")

        def worker():
            ok, message = gemini_material_query.sign_in(api_key, model)

            def apply():
                if ok:
                    self.gm_key_var.set("")
                    for w in (self.gm_key_entry, self.gm_model_entry, self.gm_signin_btn):
                        w.config(state=tk.DISABLED)
                    self.gm_signout_btn.config(state=tk.NORMAL)
                    self.cred_status.config(text=f"Gemini: signed in (model '{model}').", fg="green")
                else:
                    self.cred_status.config(text=f"Gemini: {message}", fg="red")
            self.after(0, apply)
        threading.Thread(target=worker, daemon=True).start()

    def _do_gemini_sign_out(self):
        gemini_material_query.sign_out()
        for w in (self.gm_key_entry, self.gm_model_entry, self.gm_signin_btn):
            w.config(state=tk.NORMAL)
        self.gm_signout_btn.config(state=tk.DISABLED)
        self.cred_status.config(text="Gemini: signed out.", fg="blue")

    # ---------------------------------------------------------------- scope
    def _build_scope_row(self):
        frame = tk.LabelFrame(self, text=" Batch ", padx=8, pady=6)
        frame.pack(fill=tk.X, padx=8, pady=4)

        self.scope_var = tk.StringVar(value="today")
        tk.Radiobutton(frame, text="Today", variable=self.scope_var, value="today").grid(row=0, column=0)
        tk.Radiobutton(frame, text="Full history (old objects)", variable=self.scope_var,
                       value="all").grid(row=0, column=1)
        tk.Radiobutton(frame, text="Date range:", variable=self.scope_var, value="range").grid(row=0, column=2)
        self.range_start = tk.Entry(frame, width=12)
        self.range_start.grid(row=0, column=3, padx=2)
        tk.Label(frame, text="to").grid(row=0, column=4)
        self.range_end = tk.Entry(frame, width=12)
        self.range_end.grid(row=0, column=5, padx=2)
        self.skip_labeled_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, text="Skip already-labeled", variable=self.skip_labeled_var).grid(
            row=0, column=6, padx=(12, 0))
        tk.Button(frame, text="Load Batch", command=self._load_batch, bg="lightgreen").grid(
            row=0, column=7, padx=(12, 0))
        self.batch_status = tk.Label(frame, text="No batch loaded.", fg="gray")
        self.batch_status.grid(row=1, column=0, columnspan=8, sticky=tk.W, pady=(4, 0))

    def _load_batch(self):
        scope = self.scope_var.get()
        kwargs = {"skip_enriched": self.skip_labeled_var.get()}
        if scope == "all":
            kwargs["all_history"] = True
        elif scope == "range":
            start, end = self.range_start.get().strip(), self.range_end.get().strip()
            if not (start and end):
                messagebox.showerror("Date range", "Enter both a start and end date (YYYY-MM-DD).")
                return
            kwargs["start_date"], kwargs["end_date"] = start, end
        else:
            kwargs["session_id"] = session_manager.today_session_id()

        try:
            object_ids, skipped = gemini_material_query.objects_for_scope(**kwargs)
        except Exception as e:
            messagebox.showerror("Load Batch", f"Could not load objects: {e}")
            return
        self.queue = object_ids
        self.queue_index = -1
        self.batch_status.config(
            text=f"{len(object_ids)} object(s) loaded ({skipped} already-labeled skipped).", fg="blue")
        self._advance(1)

    # ------------------------------------------------------------ review UI
    def _build_review_area(self):
        frame = tk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = tk.Frame(frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(left, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1], bg="black")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        tk.Label(left, text="Click anywhere on the photo to move the laser-point box.",
                 fg="gray", font=("Arial", 8)).pack(anchor=tk.W)
        self.detect_status = tk.Label(left, text="", fg="gray")
        self.detect_status.pack(anchor=tk.W)

        right = tk.Frame(frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        tk.Label(right, text="Object:").grid(row=0, column=0, sticky=tk.W)
        self.object_var = tk.StringVar()
        tk.Entry(right, textvariable=self.object_var, width=40).grid(row=0, column=1, sticky=tk.W)

        tk.Label(right, text="Materials:").grid(row=1, column=0, sticky=tk.NW, pady=(6, 0))
        self.materials_text = tk.Text(right, width=42, height=4)
        self.materials_text.grid(row=1, column=1, sticky=tk.W, pady=(6, 0))

        tk.Label(right, text="Labels:").grid(row=2, column=0, sticky=tk.NW, pady=(6, 0))
        self.labels_text = tk.Text(right, width=42, height=3)
        self.labels_text.grid(row=2, column=1, sticky=tk.W, pady=(6, 0))

        tk.Label(right, text="Notes:").grid(row=3, column=0, sticky=tk.NW, pady=(6, 0))
        self.notes_text = tk.Text(right, width=42, height=2)
        self.notes_text.grid(row=3, column=1, sticky=tk.W, pady=(6, 0))

        tk.Label(right, text="Gemini share code:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.share_code_var = tk.StringVar()
        tk.Entry(right, textvariable=self.share_code_var, width=30).grid(row=4, column=1, sticky=tk.W, pady=(10, 0))
        tk.Label(right, text="(paste the code after the /share/ in the link once you share the "
                              "Gemini chat yourself — optional, added whenever you have it)",
                 fg="gray", font=("Arial", 8), wraplength=320, justify=tk.LEFT).grid(
            row=5, column=1, sticky=tk.W)

        tk.Button(right, text="Re-run Gemini", command=self._run_gemini).grid(
            row=6, column=0, columnspan=2, sticky=tk.W, pady=(12, 0))

        self.gemini_status = tk.Label(right, text="", fg="gray", wraplength=380, justify=tk.LEFT)
        self.gemini_status.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))

    def _on_canvas_click(self, event):
        if self.current_pil_image is None:
            return
        img_w, img_h = self.current_pil_image.size
        disp_w, disp_h = self._displayed_size
        cx_norm = event.x / disp_w
        cy_norm = event.y / disp_h
        cx_norm, cy_norm = min(max(cx_norm, 0), 1), min(max(cy_norm, 0), 1)
        w_norm = self.box_norm["w_norm"] if self.box_norm else 0.06
        h_norm = self.box_norm["h_norm"] if self.box_norm else 0.06
        self.box_norm = {"cx_norm": cx_norm, "cy_norm": cy_norm, "w_norm": w_norm, "h_norm": h_norm,
                          "confidence": "manual"}
        self._redraw_box()
        self.detect_status.config(text="Laser point set manually.", fg="blue")

    def _redraw_box(self):
        self.canvas.delete("box")
        if not self.box_norm:
            return
        disp_w, disp_h = self._displayed_size
        cx, cy = self.box_norm["cx_norm"] * disp_w, self.box_norm["cy_norm"] * disp_h
        bw, bh = self.box_norm["w_norm"] * disp_w, self.box_norm["h_norm"] * disp_h
        color = "red" if self.box_norm.get("confidence") == "low" else "lime"
        self.canvas.create_rectangle(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2,
                                      outline=color, width=2, tags="box")

    # ------------------------------------------------------------- nav
    def _build_nav_row(self):
        frame = tk.Frame(self)
        frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Button(frame, text="< Previous", command=lambda: self._advance(-1)).pack(side=tk.LEFT)
        self.position_label = tk.Label(frame, text="No object loaded.")
        self.position_label.pack(side=tk.LEFT, padx=12)
        tk.Button(frame, text="Skip >", command=lambda: self._advance(1)).pack(side=tk.LEFT, padx=(0, 12))
        tk.Button(frame, text="Confirm & Upload", command=self._confirm_and_upload,
                  bg="lightgreen").pack(side=tk.LEFT)
        self.confirm_status = tk.Label(frame, text="", fg="gray")
        self.confirm_status.pack(side=tk.LEFT, padx=12)

    def _advance(self, step: int):
        if not self.queue:
            return
        new_index = self.queue_index + step
        if new_index < 0 or new_index >= len(self.queue):
            messagebox.showinfo("Object Labeling Studio", "No more objects in this batch.")
            return
        self.queue_index = new_index
        self._load_current_object()

    def _load_current_object(self):
        object_id = self.queue[self.queue_index]
        obj = mongo_client.get_object(object_id)
        self.current_obj = obj
        self.position_label.config(text=f"Object {self.queue_index + 1} of {len(self.queue)}  (id: {object_id})")
        self.gemini_status.config(text="")
        self.confirm_status.config(text="")
        self.object_var.set("")
        self.materials_text.delete("1.0", tk.END)
        self.labels_text.delete("1.0", tk.END)
        self.notes_text.delete("1.0", tk.END)
        self.share_code_var.set((((obj or {}).get("data") or {}).get("Gemini Share Code")) or "")

        image_path = gemini_material_query.pick_representative_image(object_id)
        self.current_image_path = image_path
        self.roboflow_image_id = self._find_roboflow_image_id(object_id, image_path)
        if image_path is None:
            self.canvas.delete("all")
            self.detect_status.config(text="No readable photo found for this object.", fg="red")
            self.current_pil_image = None
            return

        self._show_image(image_path)
        self._detect_laser()
        if gemini_material_query.is_signed_in():
            self._run_gemini()

    def _find_roboflow_image_id(self, object_id, image_path):
        """Looks up the Roboflow image id already recorded for this
        exact photo (see roboflow_export.mark_image_uploaded_to_roboflow
        via the main app's Upload flow) — needed to attach an
        annotation. None if this photo hasn't been uploaded to the
        currently-signed-in project yet."""
        cfg = roboflow_export.current_session()
        if not cfg or image_path is None:
            return None
        key = roboflow_export.project_key(cfg["workspace"], cfg["project_id"])
        for img in mongo_client.get_images_for_object(object_id):
            if img.get("image_path") == image_path:
                upload_record = (img.get("roboflow_uploads") or {}).get(key)
                if upload_record:
                    return upload_record.get("roboflow_image_id") or None
        return None

    def _show_image(self, image_path):
        img = Image.open(image_path)
        self.current_pil_image = img
        display = img.copy()
        display.thumbnail(CANVAS_SIZE)
        self._displayed_size = display.size
        self.current_photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_photo)
        self.box_norm = None

    def _detect_laser(self):
        try:
            result = laser_dot.detect_laser_dot(self.current_image_path)
        except Exception as e:
            self.detect_status.config(text=f"Laser detection failed: {e}", fg="red")
            return
        if result is None:
            self.detect_status.config(
                text="No confident laser point found automatically — click on the photo to set it.",
                fg="orange")
            return
        self.box_norm = result
        self._redraw_box()
        if result["confidence"] == "low":
            self.detect_status.config(
                text="Laser point detected, but LOW confidence — please check/correct it.", fg="orange")
        else:
            self.detect_status.config(text="Laser point detected.", fg="green")

    def _run_gemini(self):
        if not gemini_material_query.is_signed_in():
            self.gemini_status.config(text="Sign in to Gemini first.", fg="red")
            return
        if self.current_obj is None:
            return
        object_id = self.current_obj["_id"]
        self.gemini_status.config(text="Asking Gemini...", fg="gray")
        cfg = gemini_material_query.current_session()

        def worker():
            try:
                result = gemini_material_query.enrich_object(object_id, cfg["api_key"], cfg["model"])
                error = None
            except gemini_material_query.GeminiQueryError as e:
                result, error = None, str(e)

            def apply():
                if error:
                    self.gemini_status.config(text=error, fg="red")
                    return
                self.object_var.set(result["object"])
                self.materials_text.delete("1.0", tk.END)
                self.materials_text.insert(tk.END, gemini_material_query.format_materials(result["materials"]))
                self.labels_text.delete("1.0", tk.END)
                self.labels_text.insert(tk.END, ", ".join(result["labels"]))
                self.notes_text.delete("1.0", tk.END)
                self.notes_text.insert(tk.END, result["notes"])
                self.gemini_status.config(text="Gemini suggestion loaded — review before confirming.", fg="green")
            self.after(0, apply)
        threading.Thread(target=worker, daemon=True).start()

    def _confirm_and_upload(self):
        if self.current_obj is None:
            return
        object_id = self.current_obj["_id"]
        object_name = self.object_var.get().strip()
        share_code = self.share_code_var.get().strip()

        # Share code + whatever's currently in the review fields get
        # saved regardless of Roboflow annotation status below — a
        # missing Roboflow upload shouldn't block saving the label
        # itself to Mongo.
        try:
            obj = mongo_client.get_object(object_id)
            data = dict(obj.get("data") or {})
            freeform_key_name = attribute_schema.freeform_key()
            freeform = dict(data.get(freeform_key_name) or {})
            freeform["Object (AI suggested)"] = object_name
            freeform["Material (AI suggested)"] = self.materials_text.get("1.0", tk.END).strip()
            freeform["AI Labels"] = self.labels_text.get("1.0", tk.END).strip()
            freeform["AI Notes"] = self.notes_text.get("1.0", tk.END).strip()
            if share_code:
                freeform["Gemini Share Code"] = share_code
            data[freeform_key_name] = freeform
            mongo_client.update_object_data(object_id, data)
        except Exception as e:
            self.confirm_status.config(text=f"Could not save to Mongo: {e}", fg="red")
            return

        if not roboflow_export.is_signed_in():
            self.confirm_status.config(text="Saved locally. Sign in to Roboflow to also upload the annotation.",
                                        fg="orange")
            return
        if not self.roboflow_image_id:
            self.confirm_status.config(
                text="Saved locally. This photo isn't uploaded to Roboflow yet — annotation skipped.", fg="orange")
            return
        if not self.box_norm:
            self.confirm_status.config(text="Saved locally. No laser point set — annotation skipped.", fg="orange")
            return
        if not object_name:
            self.confirm_status.config(text="Enter an object name before uploading the annotation.", fg="red")
            return

        cfg = roboflow_export.current_session()
        self.confirm_status.config(text="Uploading annotation to Roboflow...", fg="gray")

        def worker():
            ok, message = roboflow_export.upload_yolo_box_annotation(
                cfg["api_key"], cfg["project_id"], self.roboflow_image_id, object_name,
                self.box_norm["cx_norm"], self.box_norm["cy_norm"],
                self.box_norm["w_norm"], self.box_norm["h_norm"])

            def apply():
                if ok:
                    self.confirm_status.config(text="Saved + annotation uploaded to Roboflow.", fg="green")
                    self._advance(1)
                else:
                    self.confirm_status.config(text=f"Saved locally, but annotation upload failed: {message}",
                                                fg="orange")
            self.after(0, apply)
        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    app = ObjectLabelingStudio()
    app.mainloop()

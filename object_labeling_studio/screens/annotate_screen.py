"""Screen 3 — Annotate. Two tabs:
  - Auto-Annotate: for objects with a real laser dot (capture-rig
    photos) — detect it, let you correct it, Confirm uploads the box +
    label to Roboflow.
  - Label Only: for backlog images (Tab 3 of Import) with no laser dot
    guaranteed — pushes Gemini's suggestion into Roboflow metadata for
    you to box by hand, via core/label_only_flow.py.
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None

from vision.storage import mongo_client, roboflow_export

from object_labeling_studio.core import gemini_client, label_only_flow, laser_dot, object_listing, record_store

CANVAS_SIZE = (620, 460)


def build(parent, app):
    notebook = ttk.Notebook(parent)
    notebook.pack(fill=tk.BOTH, expand=True)
    auto_tab = tk.Frame(notebook)
    label_only_tab = tk.Frame(notebook)
    notebook.add(auto_tab, text="Auto-Annotate (laser dot)")
    notebook.add(label_only_tab, text="Label Only (backlog)")
    _build_auto_annotate_tab(auto_tab)
    _build_label_only_tab(label_only_tab)


# ------------------------------------------------------- Auto-Annotate
def _build_auto_annotate_tab(tab):
    left = tk.Frame(tab)
    left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
    right = tk.Frame(tab)
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

    tk.Label(left, text="Label status:").pack(anchor=tk.W)
    label_filter_var = tk.StringVar(value="labeled")
    for text, val in (("Unlabeled", "unlabeled"), ("Already labeled", "labeled"), ("All", "all")):
        tk.Radiobutton(left, text=text, variable=label_filter_var, value=val).pack(anchor=tk.W)
    tk.Label(left, text="(needs a Gemini label first — Review screen — to have\na name to "
                         "put on the box)", fg="gray", font=("Arial", 8), justify=tk.LEFT).pack(anchor=tk.W)

    listbox = tk.Listbox(left, width=32, height=18)
    listbox.pack(pady=(8, 4))
    object_ids_shown = []

    def do_refresh():
        scope_kwargs = object_listing.build_scope_kwargs("all")
        objects = object_listing.list_objects(scope_kwargs, label_filter_var.get())
        listbox.delete(0, tk.END)
        object_ids_shown.clear()
        for obj in objects:
            listbox.insert(tk.END, (obj.get("data") or {}).get("name", "(unnamed)"))
            object_ids_shown.append(obj["_id"])

    tk.Button(left, text="Refresh List", command=do_refresh).pack(fill=tk.X)

    canvas = tk.Canvas(right, width=CANVAS_SIZE[0], height=CANVAS_SIZE[1], bg="black")
    canvas.pack(anchor=tk.W)
    detect_status = tk.Label(right, text="", fg="gray")
    detect_status.pack(anchor=tk.W)

    object_name_var = tk.StringVar()
    tk.Label(right, text="Object name (goes on the box):").pack(anchor=tk.W, pady=(6, 0))
    tk.Entry(right, textvariable=object_name_var, width=40).pack(anchor=tk.W)

    status = tk.Label(right, text="", fg="gray", wraplength=600, justify=tk.LEFT)
    status.pack(anchor=tk.W, pady=(8, 0))

    state = {"object_id": None, "image_path": None, "pil_image": None, "photo": None,
             "box_norm": None, "displayed_size": None, "roboflow_image_id": None}

    def show_image(path):
        img = Image.open(path)
        state["pil_image"] = img
        display = img.copy()
        display.thumbnail(CANVAS_SIZE)
        state["displayed_size"] = display.size
        state["photo"] = ImageTk.PhotoImage(display)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=state["photo"])
        state["box_norm"] = None

    def redraw_box():
        canvas.delete("box")
        box = state["box_norm"]
        if not box:
            return
        dw, dh = state["displayed_size"]
        cx, cy = box["cx_norm"] * dw, box["cy_norm"] * dh
        bw, bh = box["w_norm"] * dw, box["h_norm"] * dh
        color = "red" if box.get("confidence") == "low" else "lime"
        canvas.create_rectangle(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2,
                                 outline=color, width=2, tags="box")

    def on_canvas_click(event):
        if not state["displayed_size"]:
            return
        dw, dh = state["displayed_size"]
        cx_norm = min(max(event.x / dw, 0), 1)
        cy_norm = min(max(event.y / dh, 0), 1)
        w_norm = state["box_norm"]["w_norm"] if state["box_norm"] else 0.06
        h_norm = state["box_norm"]["h_norm"] if state["box_norm"] else 0.06
        state["box_norm"] = {"cx_norm": cx_norm, "cy_norm": cy_norm, "w_norm": w_norm,
                              "h_norm": h_norm, "confidence": "manual"}
        redraw_box()
        detect_status.config(text="Laser point set manually.", fg="blue")

    canvas.bind("<Button-1>", on_canvas_click)

    def on_select(_event=None):
        selection = listbox.curselection()
        if not selection:
            return
        object_id = object_ids_shown[selection[0]]
        state["object_id"] = object_id
        obj = mongo_client.get_object(object_id)
        object_name_var.set((obj.get("data") or {}).get("name", ""))
        image_path = gemini_client.pick_representative_image(object_id)
        state["image_path"] = image_path
        status.config(text="")
        if not image_path:
            canvas.delete("all")
            detect_status.config(text="No readable photo for this object.", fg="red")
            return
        show_image(image_path)
        cfg = roboflow_export.current_session()
        state["roboflow_image_id"] = _find_roboflow_image_id(object_id, image_path, cfg) if cfg else None
        try:
            result = laser_dot.detect_laser_dot(image_path)
        except Exception as e:
            result = None
            detect_status.config(text=f"Laser detection failed: {e}", fg="red")
        if result:
            state["box_norm"] = result
            redraw_box()
            detect_status.config(
                text="Laser point detected." if result["confidence"] != "low"
                else "Laser point detected — LOW confidence, please check.",
                fg="green" if result["confidence"] != "low" else "orange")
        else:
            detect_status.config(text="No confident laser point found — click the photo to set it.", fg="orange")

    listbox.bind("<<ListboxSelect>>", on_select)

    def do_confirm():
        object_id = state["object_id"]
        if not object_id:
            return
        cfg = roboflow_export.current_session()
        if not cfg:
            messagebox.showerror("Annotate", "Sign in to Roboflow at the top of the window first.")
            return
        if not state["roboflow_image_id"]:
            messagebox.showerror("Annotate", "This photo hasn't been uploaded to Roboflow yet — "
                                              "upload it first (main app's Roboflow Export panel).")
            return
        if not state["box_norm"]:
            messagebox.showerror("Annotate", "No laser point set — click the photo first.")
            return
        object_name = object_name_var.get().strip()
        if not object_name:
            messagebox.showerror("Annotate", "Enter an object name first.")
            return

        # Write the local-record pointer into Roboflow metadata too,
        # same as the Label Only branch, so either path leaves the
        # same trail back to the full PDF/record detail.
        pointer = record_store.roboflow_metadata_pointer(object_id)
        if pointer:
            roboflow_export.attach_metadata(cfg["api_key"], cfg["workspace"], state["roboflow_image_id"], pointer)

        box = state["box_norm"]
        status.config(text="Uploading annotation to Roboflow...", fg="gray")

        def worker():
            ok, message = roboflow_export.upload_yolo_box_annotation(
                cfg["api_key"], cfg["project_id"], state["roboflow_image_id"], object_name,
                box["cx_norm"], box["cy_norm"], box["w_norm"], box["h_norm"])

            def apply():
                status.config(text="Annotation uploaded." if ok else f"Upload failed: {message}",
                              fg="green" if ok else "red")
            right.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    tk.Button(right, text="Confirm & Upload Annotation", command=do_confirm,
              bg="lightgreen").pack(anchor=tk.W, pady=(10, 0))

    do_refresh()


def _find_roboflow_image_id(object_id, image_path, cfg):
    key = roboflow_export.project_key(cfg["workspace"], cfg["project_id"])
    for img in mongo_client.get_images_for_object(object_id):
        if img.get("image_path") == image_path:
            record = (img.get("roboflow_uploads") or {}).get(key)
            if record:
                return record.get("roboflow_image_id") or None
    return None


# --------------------------------------------------------- Label Only
def _build_label_only_tab(tab):
    tk.Label(tab, text="For images with no laser dot (e.g. imported from Roboflow's own "
                        "unlabeled backlog) — runs Gemini, saves a record, and pushes the "
                        "suggestion into that image's Roboflow metadata so it's visible right "
                        "in Roboflow's UI when you draw the box yourself.",
             wraplength=900, justify=tk.LEFT, fg="gray").pack(anchor=tk.W, padx=8, pady=6)

    listbox = tk.Listbox(tab, width=60, height=16)
    listbox.pack(anchor=tk.W, padx=8)
    object_ids_shown = []

    def do_refresh():
        scope_kwargs = object_listing.build_scope_kwargs("all")
        objects = object_listing.list_objects(scope_kwargs, "unlabeled")
        listbox.delete(0, tk.END)
        object_ids_shown.clear()
        for obj in objects:
            source = ((obj.get("data") or {}).get("attributes") or {}).get("Import Source", "")
            listbox.insert(tk.END, f"{(obj.get('data') or {}).get('name', '(unnamed)')}  [{source}]")
            object_ids_shown.append(obj["_id"])

    status = tk.Label(tab, text="", fg="gray", wraplength=900, justify=tk.LEFT)
    status.pack(anchor=tk.W, padx=8, pady=8)

    def do_label_selected():
        selection = listbox.curselection()
        if not selection:
            messagebox.showinfo("Label Only", "Select an object first.")
            return
        object_id = object_ids_shown[selection[0]]
        gm_cfg = gemini_client.current_session()
        rf_cfg = roboflow_export.current_session()
        if not gm_cfg:
            messagebox.showerror("Label Only", "Sign in to Gemini first.")
            return
        if not rf_cfg:
            messagebox.showerror("Label Only", "Sign in to Roboflow first.")
            return
        status.config(text="Running Gemini + pushing metadata...", fg="gray")

        def worker():
            try:
                result = label_only_flow.label_only(
                    object_id, gm_cfg["api_key"], gm_cfg["model"],
                    rf_cfg["api_key"], rf_cfg["workspace"], rf_cfg["project_id"])
                error = None
            except Exception as e:
                result, error = None, str(e)

            def apply():
                if error:
                    status.config(text=error, fg="red")
                    return
                note = "" if result["roboflow_image_id"] else " (not uploaded to Roboflow yet — metadata skipped)"
                status.config(text=f"Labeled '{result['gemini_result']['object']}' — "
                                    f"{len(result['records'])} record(s) saved.{note}",
                              fg="green" if not note else "orange")
                do_refresh()
            tab.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    btn_row = tk.Frame(tab)
    btn_row.pack(anchor=tk.W, padx=8, pady=(0, 8))
    tk.Button(btn_row, text="Refresh List", command=do_refresh, bg="lightblue").pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(btn_row, text="Label Selected", command=do_label_selected, bg="lightgreen").pack(side=tk.LEFT)

    do_refresh()

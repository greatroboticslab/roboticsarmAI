"""Screen 2 — Review. Pick an object (pre-loaded from Group screen's
"Send to Review," or picked fresh here), run Gemini, edit the result,
then split it into per-material-combo records (see
core/record_store.create_records_from_gemini_result)."""

import threading
import tkinter as tk
from tkinter import ttk, messagebox

from vision.storage import mongo_client, session_manager

from object_labeling_studio.core import gemini_client, object_listing, record_store


def build(parent, app):
    left = tk.Frame(parent)
    left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
    right = tk.Frame(parent)
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

    tk.Label(left, text="Label status:").pack(anchor=tk.W)
    label_filter_var = tk.StringVar(value="unlabeled")
    for text, val in (("Unlabeled", "unlabeled"), ("Already labeled", "labeled"), ("All", "all")):
        tk.Radiobutton(left, text=text, variable=label_filter_var, value=val).pack(anchor=tk.W)

    scope_var = tk.StringVar(value="today")
    tk.Label(left, text="Scope:").pack(anchor=tk.W, pady=(8, 0))
    for text, val in (("Today", "today"), ("Full history", "all")):
        tk.Radiobutton(left, text=text, variable=scope_var, value=val).pack(anchor=tk.W)

    listbox = tk.Listbox(left, width=32, height=20, selectmode=tk.SINGLE)
    listbox.pack(pady=(8, 4))
    object_ids_shown = []

    def do_refresh():
        scope_kwargs = object_listing.build_scope_kwargs(
            scope_var.get(), session_id=session_manager.today_session_id())
        objects = object_listing.list_objects(scope_kwargs, label_filter_var.get())
        listbox.delete(0, tk.END)
        object_ids_shown.clear()
        for obj in objects:
            name = (obj.get("data") or {}).get("name", "(unnamed)")
            already = "✓ " if record_store.has_records_for_object(obj["_id"]) else "  "
            listbox.insert(tk.END, f"{already}{name}")
            object_ids_shown.append(obj["_id"])

    tk.Button(left, text="Refresh List", command=do_refresh).pack(fill=tk.X)

    # ---- right side: photo + Gemini result ----
    current_object_id = {"value": None}

    photo_label = tk.Label(right, text="(no object selected)", fg="gray")
    photo_label.pack(anchor=tk.W)

    tk.Label(right, text="Object name:").pack(anchor=tk.W, pady=(8, 0))
    object_name_var = tk.StringVar()
    tk.Entry(right, textvariable=object_name_var, width=50).pack(anchor=tk.W)

    tk.Label(right, text="Materials — one per line, format:  part | material | color | confidence"
             ).pack(anchor=tk.W, pady=(8, 0))
    materials_text = tk.Text(right, width=80, height=8)
    materials_text.pack(anchor=tk.W)

    tk.Label(right, text="Labels (comma-separated):").pack(anchor=tk.W, pady=(8, 0))
    labels_var = tk.StringVar()
    tk.Entry(right, textvariable=labels_var, width=80).pack(anchor=tk.W)

    tk.Label(right, text="Notes:").pack(anchor=tk.W, pady=(8, 0))
    notes_text = tk.Text(right, width=80, height=3)
    notes_text.pack(anchor=tk.W)

    status = tk.Label(right, text="", fg="gray", wraplength=700, justify=tk.LEFT)
    status.pack(anchor=tk.W, pady=(8, 0))

    def on_select(_event=None):
        selection = listbox.curselection()
        if not selection:
            return
        object_id = object_ids_shown[selection[0]]
        current_object_id["value"] = object_id
        obj = mongo_client.get_object(object_id)
        object_name_var.set((obj.get("data") or {}).get("name", ""))
        materials_text.delete("1.0", tk.END)
        labels_var.set("")
        notes_text.delete("1.0", tk.END)
        image_path = gemini_client.pick_representative_image(object_id)
        photo_label.config(text=image_path or "(no readable photo for this object)")

        existing = record_store.records_for_object(object_id)
        if existing:
            lines = [f"{r['part']} | {r['material']} | {r['color']} | {r['confidence']}" for r in existing]
            materials_text.insert(tk.END, "\n".join(lines))
            notes_text.insert(tk.END, existing[0].get("notes", ""))
            status.config(text=f"{len(existing)} existing record(s) for this object shown above — "
                                f"Run Gemini to refresh, or Generate Records to add more as-is.", fg="gray")

    listbox.bind("<<ListboxSelect>>", on_select)

    def do_run_gemini():
        object_id = current_object_id["value"]
        if not object_id:
            messagebox.showinfo("Review", "Select an object first.")
            return
        cfg = gemini_client.current_session()
        if not cfg:
            messagebox.showerror("Review", "Sign in to Gemini at the top of the window first.")
            return
        image_path = gemini_client.pick_representative_image(object_id)
        if not image_path:
            messagebox.showerror("Review", "This object has no readable photo.")
            return
        status.config(text="Asking Gemini...", fg="gray")

        def worker():
            try:
                result = gemini_client.describe_material(
                    cfg["api_key"], cfg["model"], image_path, context={"name": object_name_var.get()})
                error = None
            except gemini_client.GeminiQueryError as e:
                result, error = None, str(e)

            def apply():
                if error:
                    status.config(text=error, fg="red")
                    return
                object_name_var.set(result["object"] or object_name_var.get())
                materials_text.delete("1.0", tk.END)
                lines = [f"{m['part']} | {m['material']} | {m['color']} | {m['confidence']}"
                         for m in result["materials"]]
                materials_text.insert(tk.END, "\n".join(lines))
                labels_var.set(", ".join(result["labels"]))
                notes_text.delete("1.0", tk.END)
                notes_text.insert(tk.END, result["notes"])
                current_object_id["_last_gemini_result"] = result
                status.config(text="Gemini suggestion loaded — edit above, then Generate Records.", fg="green")
            right.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def _parse_materials_text() -> list:
        materials = []
        for line in materials_text.get("1.0", tk.END).splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split("|")]
            while len(parts) < 4:
                parts.append("")
            part, material, color, confidence = parts[:4]
            materials.append({"part": part or "overall", "material": material, "color": color,
                               "confidence": confidence})
        return materials

    def do_generate_records():
        object_id = current_object_id["value"]
        if not object_id:
            messagebox.showinfo("Review", "Select an object first.")
            return
        materials = _parse_materials_text()
        if not materials:
            messagebox.showinfo("Review", "Add at least one material line first (or Run Gemini).")
            return
        last_result = current_object_id.get("_last_gemini_result", {})
        gemini_result = {
            "object": object_name_var.get(),
            "materials": materials,
            "labels": [l.strip() for l in labels_var.get().split(",") if l.strip()],
            "notes": notes_text.get("1.0", tk.END).strip(),
            "prompt_text": last_result.get("prompt_text", ""),
            "response_text": last_result.get("response_text", ""),
        }
        image_path = gemini_client.pick_representative_image(object_id)
        records = record_store.create_records_from_gemini_result(
            object_id, object_name_var.get(), image_path, gemini_result)
        status.config(text=f"Created {len(records)} record(s) with backing PDFs.", fg="green")
        do_refresh()

    btn_row = tk.Frame(right)
    btn_row.pack(anchor=tk.W, pady=(8, 0))
    tk.Button(btn_row, text="Run Gemini", command=do_run_gemini, bg="lightblue").pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(btn_row, text="Generate Records", command=do_generate_records, bg="lightgreen").pack(side=tk.LEFT)

    # Pre-load selection handed off from the Group screen, if any.
    if app.state_obj.selected_object_ids:
        do_refresh()
        for i, object_id in enumerate(object_ids_shown):
            if object_id in app.state_obj.selected_object_ids:
                listbox.selection_set(i)
                on_select()
                break
    else:
        do_refresh()

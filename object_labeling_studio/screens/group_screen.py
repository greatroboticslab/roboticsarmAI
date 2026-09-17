"""Screen 1 — Group. Lists objects (with the shared labeled/unlabeled/
all filter), lets you multi-select several and merge them into one
object by reassigning their images — for filesystem imports that
landed as "ungrouped" or one-object-per-image and actually belong
together."""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

from vision.storage import mongo_client, session_manager

from object_labeling_studio.core import object_listing


def build(parent, app):
    top = tk.Frame(parent)
    top.pack(fill=tk.X, padx=8, pady=8)

    tk.Label(top, text="Show:").grid(row=0, column=0, sticky=tk.W)
    scope_var = tk.StringVar(value="today")
    tk.Radiobutton(top, text="Today", variable=scope_var, value="today").grid(row=0, column=1)
    tk.Radiobutton(top, text="Full history", variable=scope_var, value="all").grid(row=0, column=2)
    tk.Radiobutton(top, text="Date range:", variable=scope_var, value="range").grid(row=0, column=3)
    start_entry = tk.Entry(top, width=12)
    start_entry.grid(row=0, column=4)
    tk.Label(top, text="to").grid(row=0, column=5)
    end_entry = tk.Entry(top, width=12)
    end_entry.grid(row=0, column=6)

    tk.Label(top, text="Label status:").grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
    label_filter_var = tk.StringVar(value="all")
    tk.Radiobutton(top, text="Unlabeled", variable=label_filter_var, value="unlabeled").grid(
        row=1, column=1, pady=(4, 0))
    tk.Radiobutton(top, text="Already labeled", variable=label_filter_var, value="labeled").grid(
        row=1, column=2, pady=(4, 0))
    tk.Radiobutton(top, text="All", variable=label_filter_var, value="all").grid(row=1, column=3, pady=(4, 0))

    tree = ttk.Treeview(parent, columns=("name", "images", "session"), show="headings",
                         height=20, selectmode="extended")
    tree.heading("name", text="Object")
    tree.heading("images", text="# Images")
    tree.heading("session", text="Session")
    tree.column("name", width=300)
    tree.column("images", width=100)
    tree.column("session", width=200)
    tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    status = tk.Label(parent, text="", fg="gray")
    status.pack(anchor=tk.W, padx=8)

    def do_refresh():
        scope_kwargs = object_listing.build_scope_kwargs(
            scope_var.get(), session_id=session_manager.today_session_id(),
            start_date=start_entry.get().strip(), end_date=end_entry.get().strip())
        objects = object_listing.list_objects(scope_kwargs, label_filter_var.get())
        tree.delete(*tree.get_children())
        for obj in objects:
            name = (obj.get("data") or {}).get("name", "(unnamed)")
            num_images = len(mongo_client.get_images_for_object(obj["_id"]))
            tree.insert("", tk.END, iid=obj["_id"], values=(name, num_images, obj.get("session_id", "")))
        status.config(text=f"{len(objects)} object(s).", fg="blue")

    def do_merge():
        selected = tree.selection()
        if len(selected) < 2:
            messagebox.showinfo("Group", "Select two or more objects to merge into one.")
            return
        new_name = simpledialog.askstring("Group Name", "Name for the merged object:")
        if not new_name:
            return
        keep_id = selected[0]
        for object_id in selected[1:]:
            for img in mongo_client.get_images_for_object(object_id):
                mongo_client.reassign_image_object(img["_id"], keep_id)
            mongo_client.delete_object(object_id)
        obj = mongo_client.get_object(keep_id)
        data = dict(obj.get("data") or {})
        data["name"] = new_name
        mongo_client.update_object_data(keep_id, data)
        status.config(text=f"Merged {len(selected)} objects into '{new_name}'.", fg="green")
        do_refresh()

    def do_use_for_review():
        selected = list(tree.selection())
        if not selected:
            messagebox.showinfo("Group", "Select one or more objects first.")
            return
        app.state_obj.selected_object_ids = selected
        messagebox.showinfo("Group", f"{len(selected)} object(s) sent to the Review screen.")

    btn_row = tk.Frame(parent)
    btn_row.pack(fill=tk.X, padx=8, pady=(0, 8))
    tk.Button(btn_row, text="Refresh", command=do_refresh, bg="lightblue").pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(btn_row, text="Merge Selected Into One Object", command=do_merge).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(btn_row, text="Send Selected to Review", command=do_use_for_review, bg="lightgreen").pack(side=tk.LEFT)

    do_refresh()

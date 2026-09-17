"""Screen 0 — Import. Three tabs, one per source; see core/import_*.py
for the actual logic behind each (this module is UI only)."""

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from object_labeling_studio.core import import_filesystem, import_package, import_roboflow
from vision.storage import roboflow_export


def build(parent, app):
    notebook = ttk.Notebook(parent)
    notebook.pack(fill=tk.BOTH, expand=True)

    fs_tab = tk.Frame(notebook)
    pkg_tab = tk.Frame(notebook)
    rf_tab = tk.Frame(notebook)
    notebook.add(fs_tab, text="Filesystem")
    notebook.add(pkg_tab, text="Data Package")
    notebook.add(rf_tab, text="Roboflow Backlog")

    _build_filesystem_tab(fs_tab)
    _build_package_tab(pkg_tab)
    _build_roboflow_tab(rf_tab)


# ------------------------------------------------------------- Tab 1
def _build_filesystem_tab(tab):
    tk.Label(tab, text="Scan a folder of images with no known structure — this GUESSES object "
                        "grouping (by subfolder name), view label (from filename), and capture "
                        "date (EXIF, falling back to file modified time). Everything below is "
                        "editable before Import actually writes anything.",
             wraplength=900, justify=tk.LEFT, fg="gray").pack(anchor=tk.W, padx=8, pady=6)

    path_row = tk.Frame(tab)
    path_row.pack(fill=tk.X, padx=8)
    path_var = tk.StringVar()
    tk.Entry(path_row, textvariable=path_var, width=70).pack(side=tk.LEFT, padx=(0, 6))

    def browse():
        d = filedialog.askdirectory()
        if d:
            path_var.set(d)

    tk.Button(path_row, text="Browse...", command=browse).pack(side=tk.LEFT)

    tree = ttk.Treeview(tab, columns=("group", "view", "date", "note"), show="headings", height=18)
    for col, width in (("group", 160), ("view", 90), ("date", 140), ("note", 380)):
        tree.heading(col, text=col.capitalize())
        tree.column(col, width=width)
    tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    previews = []
    status = tk.Label(tab, text="", fg="gray")
    status.pack(anchor=tk.W, padx=8)

    def do_scan():
        folder = path_var.get().strip()
        if not folder:
            messagebox.showerror("Import", "Choose a folder first.")
            return
        previews.clear()
        tree.delete(*tree.get_children())
        found = import_filesystem.scan_folder(folder)
        previews.extend(found)
        for i, p in enumerate(found):
            date_str = p.captured_at.strftime("%Y-%m-%d %H:%M") if p.captured_at else ""
            tree.insert("", tk.END, iid=str(i), values=(p.object_group, p.view_label, date_str, p.source_note))
        status.config(text=f"{len(found)} image(s) found — double-click a row's Group/View cell "
                            f"to edit before importing.", fg="blue")

    def edit_cell(event):
        item_id = tree.identify_row(event.y)
        col = tree.identify_column(event.x)
        if not item_id or col not in ("#1", "#2"):
            return
        idx = int(item_id)
        field = "object_group" if col == "#1" else "view_label"
        current = getattr(previews[idx], field)

        entry_win = tk.Toplevel(tab)
        entry_win.title("Edit")
        var = tk.StringVar(value=current)
        tk.Entry(entry_win, textvariable=var, width=30).pack(padx=10, pady=10)

        def save():
            setattr(previews[idx], field, var.get().strip())
            tree.set(item_id, "group" if field == "object_group" else "view", var.get().strip())
            entry_win.destroy()

        tk.Button(entry_win, text="Save", command=save).pack(pady=(0, 10))

    tree.bind("<Double-1>", edit_cell)

    def do_commit():
        if not previews:
            return
        if not messagebox.askyesno("Import", f"Import {len(previews)} image(s) into "
                                              f"{len({p.object_group for p in previews})} object(s)?"):
            return
        result = import_filesystem.commit_import(previews)
        status.config(text=f"Imported {result['objects_created']} object(s), "
                            f"{result['images_created']} image(s).", fg="green")
        previews.clear()
        tree.delete(*tree.get_children())

    btn_row = tk.Frame(tab)
    btn_row.pack(fill=tk.X, padx=8, pady=(0, 8))
    tk.Button(btn_row, text="Scan Folder", command=do_scan, bg="lightblue").pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(btn_row, text="Import All Rows Above", command=do_commit, bg="lightgreen").pack(side=tk.LEFT)


# ------------------------------------------------------------- Tab 2
def _build_package_tab(tab):
    tk.Label(tab, text="Import a Data Package folder produced by the main app's own Export "
                        "Range Package / Export Data Package buttons — reads its manifest "
                        "directly, no guessing involved (unlike the Filesystem tab).",
             wraplength=900, justify=tk.LEFT, fg="gray").pack(anchor=tk.W, padx=8, pady=6)

    path_row = tk.Frame(tab)
    path_row.pack(fill=tk.X, padx=8)
    path_var = tk.StringVar()
    tk.Entry(path_row, textvariable=path_var, width=70).pack(side=tk.LEFT, padx=(0, 6))

    def browse():
        d = filedialog.askdirectory()
        if d:
            path_var.set(d)

    tk.Button(path_row, text="Browse...", command=browse).pack(side=tk.LEFT)

    status = tk.Label(tab, text="", fg="gray", wraplength=900, justify=tk.LEFT)
    status.pack(anchor=tk.W, padx=8, pady=8)

    def do_import():
        folder = path_var.get().strip()
        if not folder:
            messagebox.showerror("Import", "Choose a package folder first.")
            return
        try:
            result = import_package.import_from_package(folder)
        except Exception as e:
            status.config(text=f"Import failed: {e}", fg="red")
            return
        msg = f"Imported {result['imported']} object(s), {result['skipped']} skipped."
        if result["warnings"]:
            msg += f" {len(result['warnings'])} warning(s) — see console."
            for w in result["warnings"]:
                print(f"[PACKAGE IMPORT] {w}")
        status.config(text=msg, fg="green" if not result["warnings"] else "orange")

    tk.Button(tab, text="Import Package", command=do_import, bg="lightgreen").pack(anchor=tk.W, padx=8)


# ------------------------------------------------------------- Tab 3
def _build_roboflow_tab(tab):
    tk.Label(tab, text="Pulls images from the currently signed-in Roboflow project that have "
                        "NO annotation yet, so Gemini can suggest a label before you box them "
                        "by hand in Roboflow. Sign in to Roboflow at the top of the window "
                        "first. Each pulled image becomes its own new local object (Roboflow "
                        "has no concept of grouping) — use the Group screen afterward if some "
                        "belong together.",
             wraplength=900, justify=tk.LEFT, fg="gray").pack(anchor=tk.W, padx=8, pady=6)

    tree = ttk.Treeview(tab, columns=("name", "id"), show="headings", height=16, selectmode="extended")
    tree.heading("name", text="Name")
    tree.heading("id", text="Roboflow Image ID")
    tree.column("name", width=400)
    tree.column("id", width=300)
    tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    status = tk.Label(tab, text="", fg="gray", wraplength=900, justify=tk.LEFT)
    status.pack(anchor=tk.W, padx=8)

    results_by_iid = {}

    def do_fetch():
        cfg = roboflow_export.current_session()
        if not cfg:
            messagebox.showerror("Roboflow Backlog", "Sign in to Roboflow at the top of the window first.")
            return
        status.config(text="Fetching unlabeled images...", fg="gray")
        tree.delete(*tree.get_children())
        results_by_iid.clear()

        def worker():
            try:
                results = import_roboflow.list_unlabeled_images(cfg["api_key"], cfg["workspace"],
                                                                  cfg["project_id"])
            except Exception as e:
                tab.after(0, lambda: status.config(text=f"Fetch failed: {e}", fg="red"))
                return

            def apply():
                for r in results:
                    iid = r["id"]
                    results_by_iid[iid] = r
                    tree.insert("", tk.END, iid=iid, values=(r.get("name", ""), iid))
                status.config(text=f"{len(results)} unlabeled image(s) found. Select rows, then Import.",
                              fg="blue")
            tab.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def do_import_selected():
        cfg = roboflow_export.current_session()
        selected = [results_by_iid[iid] for iid in tree.selection()]
        if not selected:
            messagebox.showinfo("Roboflow Backlog", "Select one or more rows first.")
            return
        status.config(text=f"Importing {len(selected)} image(s)...", fg="gray")

        def worker():
            result = import_roboflow.import_selected(cfg["api_key"], cfg["workspace"],
                                                       cfg["project_id"], selected)

            def apply():
                msg = f"Imported {result['imported']} image(s)."
                if result["failed"]:
                    msg += f" {len(result['failed'])} failed — see console."
                    for f in result["failed"]:
                        print(f"[ROBOFLOW BACKLOG IMPORT] {f['id']}: {f['error']}")
                status.config(text=msg, fg="green" if not result["failed"] else "orange")
                for r in selected:
                    tree.delete(r["id"])
            tab.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    btn_row = tk.Frame(tab)
    btn_row.pack(fill=tk.X, padx=8, pady=(0, 8))
    tk.Button(btn_row, text="Fetch Unlabeled Images", command=do_fetch, bg="lightblue").pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(btn_row, text="Import Selected", command=do_import_selected, bg="lightgreen").pack(side=tk.LEFT)

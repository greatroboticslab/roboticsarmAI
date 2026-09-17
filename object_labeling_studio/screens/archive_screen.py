"""Screen 4 — Archive. Browse every gemini_records document, open its
PDF, edit its share code (regenerates the PDF so the code actually
appears in it), and the "Paste Conversation" import for backfilling
old, manually-shared chats (see core/pdf_export.py's docstring for why
this is paste-based rather than an automated fetch)."""

import os
import platform
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

from object_labeling_studio.core import record_store


def build(parent, app):
    top = tk.Frame(parent)
    top.pack(fill=tk.X, padx=8, pady=8)

    tree = ttk.Treeview(parent, columns=("object", "part", "material", "color", "source", "share"),
                         show="headings", height=18)
    for col, width, heading in (("object", 200, "Object"), ("part", 100, "Part"),
                                 ("material", 120, "Material"), ("color", 100, "Color"),
                                 ("source", 80, "Source"), ("share", 160, "Share Code")):
        tree.heading(col, text=heading)
        tree.column(col, width=width)
    tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    records_by_iid = {}

    def do_refresh():
        tree.delete(*tree.get_children())
        records_by_iid.clear()
        for r in record_store.all_records():
            records_by_iid[r["_id"]] = r
            tree.insert("", tk.END, iid=r["_id"],
                        values=(r.get("object", ""), r.get("part", ""), r.get("material", ""),
                                r.get("color", ""), r.get("source", ""), r.get("share_code", "")))

    tk.Button(top, text="Refresh", command=do_refresh, bg="lightblue").pack(side=tk.LEFT)

    def do_open_pdf():
        selection = tree.selection()
        if not selection:
            return
        record = records_by_iid[selection[0]]
        path = record.get("pdf_path")
        if not path or not os.path.exists(path):
            messagebox.showerror("Archive", "PDF file not found on disk.")
            return
        try:
            if platform.system() == "Windows":
                os.startfile(path)  # noqa: S606 — user-owned file, no untrusted input
            elif platform.system() == "Darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception as e:
            messagebox.showerror("Archive", f"Could not open PDF: {e}")

    def do_edit_share_code():
        selection = tree.selection()
        if not selection:
            return
        record_id = selection[0]
        record = records_by_iid[record_id]

        win = tk.Toplevel(parent)
        win.title("Edit Share Code")
        tk.Label(win, text="Gemini share code (paste after the /share/ in the link once you've "
                            "shared this conversation yourself):", wraplength=380, justify=tk.LEFT
                 ).pack(padx=10, pady=(10, 4))
        var = tk.StringVar(value=record.get("share_code", ""))
        tk.Entry(win, textvariable=var, width=40).pack(padx=10)

        def save():
            record_store.set_share_code(record_id, var.get())
            win.destroy()
            do_refresh()

        tk.Button(win, text="Save", command=save, bg="lightgreen").pack(pady=10)

    btn_row = tk.Frame(parent)
    btn_row.pack(fill=tk.X, padx=8, pady=(0, 8))
    tk.Button(btn_row, text="Open PDF", command=do_open_pdf).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(btn_row, text="Edit Share Code", command=do_edit_share_code).pack(side=tk.LEFT)

    # ---- Paste Conversation import ----
    import_frame = tk.LabelFrame(parent, text=" Paste Conversation (backfill an old shared chat) ",
                                  padx=8, pady=8)
    import_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
    tk.Label(import_frame,
             text="No official API can fetch a shared Gemini conversation's content — copy the "
                  "text out of it yourself and paste it below; it'll be formatted into the same "
                  "PDF template as everything else in this archive.",
             wraplength=900, justify=tk.LEFT, fg="gray").pack(anchor=tk.W)

    row1 = tk.Frame(import_frame)
    row1.pack(fill=tk.X, pady=(6, 0))
    tk.Label(row1, text="Object ID:").pack(side=tk.LEFT)
    object_id_var = tk.StringVar()
    tk.Entry(row1, textvariable=object_id_var, width=24).pack(side=tk.LEFT, padx=(2, 10))
    tk.Label(row1, text="Object name:").pack(side=tk.LEFT)
    object_name_var = tk.StringVar()
    tk.Entry(row1, textvariable=object_name_var, width=24).pack(side=tk.LEFT, padx=(2, 10))

    row2 = tk.Frame(import_frame)
    row2.pack(fill=tk.X, pady=(4, 0))
    tk.Label(row2, text="Part:").pack(side=tk.LEFT)
    part_var = tk.StringVar(value="overall")
    tk.Entry(row2, textvariable=part_var, width=14).pack(side=tk.LEFT, padx=(2, 10))
    tk.Label(row2, text="Material:").pack(side=tk.LEFT)
    material_var = tk.StringVar()
    tk.Entry(row2, textvariable=material_var, width=14).pack(side=tk.LEFT, padx=(2, 10))
    tk.Label(row2, text="Color:").pack(side=tk.LEFT)
    color_var = tk.StringVar()
    tk.Entry(row2, textvariable=color_var, width=14).pack(side=tk.LEFT, padx=(2, 10))

    tk.Label(import_frame, text="Pasted conversation text:").pack(anchor=tk.W, pady=(6, 0))
    pasted_text_widget = tk.Text(import_frame, width=100, height=8)
    pasted_text_widget.pack(anchor=tk.W)

    import_status = tk.Label(import_frame, text="", fg="gray")
    import_status.pack(anchor=tk.W, pady=(4, 0))

    def do_paste_import():
        object_id = object_id_var.get().strip()
        pasted_text = pasted_text_widget.get("1.0", tk.END).strip()
        if not (object_id and material_var.get().strip() and pasted_text):
            messagebox.showerror("Paste Import", "Object ID, Material, and the pasted text are all required.")
            return
        record = record_store.create_record_from_pasted_conversation(
            object_id, object_name_var.get().strip(), part_var.get().strip(),
            material_var.get().strip(), color_var.get().strip(), pasted_text)
        import_status.config(text=f"Record '{record['_id']}' created.", fg="green")
        pasted_text_widget.delete("1.0", tk.END)
        do_refresh()

    tk.Button(import_frame, text="Create Record from Pasted Text", command=do_paste_import,
              bg="lightgreen").pack(anchor=tk.W, pady=(6, 0))

    do_refresh()

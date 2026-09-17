#!/usr/bin/env python3
"""
Object Labeling Studio — entry point.

Run with:  python object_labeling_studio/app.py
(works from any working directory — see config.py's sys.path bootstrap)

A separate, self-contained program from the main app's main.py — see
this folder's README.md for why, and for exactly which vision.*
modules it shares (the Mongo/Roboflow data layer) versus owns outright
(everything under core/ and screens/).
"""

import os
import sys
import tkinter as tk
from tkinter import ttk

# Must happen BEFORE any `object_labeling_studio.*`/`vision.*` import
# below — when this file is launched directly (`python
# object_labeling_studio/app.py`), Python only puts THIS file's own
# directory on sys.path, not its parent, so `object_labeling_studio`
# itself isn't importable yet without this. config.py does the same
# fix for every OTHER module in this tool, but app.py is the one entry
# point that runs before config.py gets a chance to.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from object_labeling_studio import config  # noqa: F401,E402 (sys.path bootstrap side effect)
from object_labeling_studio.core import gemini_client  # noqa: E402
from vision.storage import roboflow_export  # noqa: E402


class AppState:
    """Tiny shared mutable state passed between screens — just enough
    to hand off "which objects are we working on" from one screen to
    the next (Group -> Review -> Annotate) without a database round
    trip just to remember a selection."""

    def __init__(self):
        self.selected_object_ids = []


class ObjectLabelingStudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Object Labeling Studio")
        self.geometry("1150x820")
        self.state_obj = AppState()

        self._build_signin_bar()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        # Imported lazily (not at module top) so a missing/broken
        # screen module doesn't prevent the whole app from at least
        # opening with a clear error tab instead of refusing to start.
        self._add_screen_tab("Import", "object_labeling_studio.screens.import_screen")
        self._add_screen_tab("Group", "object_labeling_studio.screens.group_screen")
        self._add_screen_tab("Review", "object_labeling_studio.screens.review_screen")
        self._add_screen_tab("Annotate", "object_labeling_studio.screens.annotate_screen")
        self._add_screen_tab("Archive", "object_labeling_studio.screens.archive_screen")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _add_screen_tab(self, title: str, module_path: str):
        frame = tk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        try:
            module = __import__(module_path, fromlist=["build"])
            module.build(frame, self)
        except Exception as e:
            tk.Label(frame, text=f"Could not load this screen:\n{e}", fg="red",
                     wraplength=700, justify=tk.LEFT).pack(padx=20, pady=20)
            import traceback
            traceback.print_exc()

    def _build_signin_bar(self):
        bar = tk.LabelFrame(self, text=" Sign In (session-only — nothing saved to disk) ", padx=8, pady=6)
        bar.pack(fill=tk.X, padx=8, pady=(8, 0))

        tk.Label(bar, text="Roboflow key:").grid(row=0, column=0, sticky=tk.W)
        self.rf_key_var = tk.StringVar()
        self.rf_key_entry = tk.Entry(bar, textvariable=self.rf_key_var, width=20, show="*")
        self.rf_key_entry.grid(row=0, column=1, padx=4)
        tk.Label(bar, text="Workspace:").grid(row=0, column=2, sticky=tk.W)
        self.rf_ws_var = tk.StringVar()
        self.rf_ws_entry = tk.Entry(bar, textvariable=self.rf_ws_var, width=14)
        self.rf_ws_entry.grid(row=0, column=3, padx=4)
        tk.Label(bar, text="Project:").grid(row=0, column=4, sticky=tk.W)
        self.rf_proj_var = tk.StringVar()
        self.rf_proj_entry = tk.Entry(bar, textvariable=self.rf_proj_var, width=18)
        self.rf_proj_entry.grid(row=0, column=5, padx=4)
        self.rf_btn = tk.Button(bar, text="Sign In", command=self._do_roboflow_signin, bg="khaki")
        self.rf_btn.grid(row=0, column=6, padx=(8, 0))

        tk.Label(bar, text="Gemini key:").grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        self.gm_key_var = tk.StringVar()
        self.gm_key_entry = tk.Entry(bar, textvariable=self.gm_key_var, width=20, show="*")
        self.gm_key_entry.grid(row=1, column=1, padx=4, pady=(4, 0))
        tk.Label(bar, text="Model:").grid(row=1, column=2, sticky=tk.W, pady=(4, 0))
        self.gm_model_var = tk.StringVar(value="gemini-2.5-flash")
        self.gm_model_entry = tk.Entry(bar, textvariable=self.gm_model_var, width=18)
        self.gm_model_entry.grid(row=1, column=3, padx=4, pady=(4, 0))
        self.gm_btn = tk.Button(bar, text="Sign In", command=self._do_gemini_signin, bg="khaki")
        self.gm_btn.grid(row=1, column=6, padx=(8, 0), pady=(4, 0))

        self.signin_status = tk.Label(bar, text="Not signed in to either service yet.", fg="gray")
        self.signin_status.grid(row=2, column=0, columnspan=7, sticky=tk.W, pady=(4, 0))

    def _do_roboflow_signin(self):
        ok, msg = roboflow_export.sign_in(self.rf_key_var.get().strip(), self.rf_ws_var.get().strip(),
                                           self.rf_proj_var.get().strip())
        if ok:
            self.rf_key_var.set("")
            self.rf_key_entry.config(state=tk.DISABLED)
        self.signin_status.config(text=f"Roboflow: {msg}", fg="green" if ok else "red")

    def _do_gemini_signin(self):
        ok, msg = gemini_client.sign_in(self.gm_key_var.get().strip(), self.gm_model_var.get().strip())
        if ok:
            self.gm_key_var.set("")
            self.gm_key_entry.config(state=tk.DISABLED)
        self.signin_status.config(text=f"Gemini: {msg}", fg="green" if ok else "red")

    def _on_close(self):
        roboflow_export.sign_out()
        gemini_client.sign_out()
        self.destroy()


if __name__ == "__main__":
    app = ObjectLabelingStudioApp()
    app.mainloop()

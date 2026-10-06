"""hoid_ui - lightweight Tkinter front-end for hoid_engine.


Layout: left controls (choose/process/stop/preview toggle/progress) |
        right live video preview + interaction summary.

Threading model (keeps inference at full speed):
- Inference runs in a daemon worker thread (hoid_engine.process_video).
- Worker NEVER touches Tk; it only put_nowait()s into a small Queue.
- UI polls via root.after(100ms) and drops behind frames (queue maxsize 4).
- Preview toggle is a threading.Event (NOT a Tk var read from worker).
"""
import queue as queue_mod
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, ttk

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import hoid_engine  # noqa: E402

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


class HoidApp:
    def __init__(self, root):
        self.root = root
        root.title("HOID Runner")
        root.geometry("1100x700")

        self.selected_path = None
        self.msg_q = queue_mod.Queue(maxsize=4)
        self.stop_event = None
        self.worker = None
        self.processing = False
        self.photo = None  # keep PhotoImage ref
        # Native input size (w, h) and aspect-fit display box (w, h).
        self.video_wh = None
        self.disp_wh = (hoid_engine.PREVIEW_MAX_W, hoid_engine.PREVIEW_MAX_H)

        # Thread-safe preview flag (worker checks Event.is_set(); Tk var stays on UI thread)
        self.preview_event = threading.Event()
        self.preview_event.set()
        self.preview_var = tk.BooleanVar(value=True)

        # ---- layout ----
        left = ttk.Frame(root, padding=10, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        right = ttk.Frame(root, padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="1. Choose a video", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        ttk.Button(left, text="Choose video...", command=self.choose_video).pack(fill=tk.X, pady=(4, 2))
        self.path_label = ttk.Label(left, text="(none)", wraplength=260, foreground="gray")
        self.path_label.pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(left, text="2. Process", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.process_btn = ttk.Button(left, text="Process", command=self.start_process, state=tk.DISABLED)
        self.process_btn.pack(fill=tk.X, pady=2)
        self.stop_btn = ttk.Button(left, text="Stop", command=self.stop_process, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)

        self.preview_check = ttk.Checkbutton(
            left, text="Live preview (off = max speed)",
            variable=self.preview_var, command=self._on_preview_toggle)
        self.preview_check.pack(anchor=tk.W, pady=(8, 2))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(left, mode="determinate", maximum=100,
                                            variable=self.progress_var)
        self.progress_bar.pack(fill=tk.X, pady=(8, 2))
        self.progress_label = ttk.Label(left, text="Idle")
        self.progress_label.pack(anchor=tk.W)

        self.status_label = ttk.Label(left, text="Models load once on first Process.",
                                      wraplength=260, foreground="gray")
        self.status_label.pack(anchor=tk.W, pady=(8, 0))

        # Right: preview + summary
        ttk.Label(right, text="Live preview", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.preview_info = ttk.Label(right, text="", foreground="gray")
        self.preview_info.pack(anchor=tk.W)
        # Fixed-size frame locked to the input aspect ratio (no stretch, no
        # per-frame geometry churn). Engine already resizes thumbs to fit it.
        bw, bh = self.disp_wh
        self.preview_frame = tk.Frame(right, bg="black", width=bw, height=bh)
        self.preview_frame.pack(pady=(2, 8))
        self.preview_frame.pack_propagate(False)
        self.preview_label = tk.Label(self.preview_frame, bg="black", fg="white",
                                      text="No preview yet")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right, text="Interaction summary (live = final by end)",
                  font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        text_frame = ttk.Frame(right)
        text_frame.pack(fill=tk.BOTH, expand=True)
        self.summary_text = tk.Text(text_frame, height=12, wrap=tk.NONE, state=tk.DISABLED)
        vsb = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        hsb = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=self.summary_text.xview)
        self.summary_text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        if not _HAS_PIL:
            self._set_status("WARNING: Pillow missing — preview disabled.")
            self.preview_var.set(False)
            self.preview_event.clear()
            self.preview_check.configure(state=tk.DISABLED)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self.poll_queue)

    # ---- controls ----
    def _on_preview_toggle(self):
        if self.preview_var.get():
            self.preview_event.set()
        else:
            self.preview_event.clear()
            self.preview_label.configure(image="", text="Preview off (max speed)")

    # ---- preview sizing (aspect-fit from video input) ----
    def _preview_caps(self):
        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
        except Exception:
            sw, sh = 1600, 900
        cap_w = min(960, max(320, sw - 380))
        cap_h = min(600, max(180, sh - 280))
        return cap_w, cap_h

    def _apply_preview_box(self, vw, vh):
        cap_w, cap_h = self._preview_caps()
        bw, bh = hoid_engine.fit_display_box(vw, vh, cap_w, cap_h)
        self.video_wh = (int(vw), int(vh))
        self.disp_wh = (bw, bh)
        try:
            self.preview_frame.configure(width=bw, height=bh)
        except Exception:
            pass
        self.preview_info.configure(text=f"Input {vw}x{vh} -> preview {bw}x{bh}")
        return bw, bh

    def choose_video(self):
        if self.processing:
            return
        initial = str(hoid_engine.video_input_directory)
        p = filedialog.askopenfilename(
            title="Choose video",
            initialdir=initial,
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if p:
            self.selected_path = Path(p)
            self.path_label.configure(text=str(self.selected_path), foreground="black")
            self.process_btn.configure(state=tk.NORMAL)
            vw, vh, _, _ = hoid_engine.get_video_info(self.selected_path)
            if vw and vh:
                self._apply_preview_box(vw, vh)
                self._set_status(f"Selected: {self.selected_path.name} ({vw}x{vh})")
            else:
                self._set_status(f"Selected: {self.selected_path.name} (size unknown, using default preview)")

    def start_process(self):
        if self.processing or not self.selected_path:
            return
        self.msg_q = queue_mod.Queue(maxsize=4)
        self.stop_event = threading.Event()
        self.processing = True
        self.process_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.configure(text="Starting (models load on first run)...")
        self._set_summary("(processing...)")
        if self.preview_var.get():
            self.preview_label.configure(text="Loading...")
        self._set_status(f"Processing {self.selected_path.name} ...")
        self.worker = threading.Thread(
            target=hoid_engine.process_video,
            kwargs={"video_path": self.selected_path, "q": self.msg_q,
                    "stop_event": self.stop_event, "is_preview_on": self.preview_event,
                    "preview_box": self.disp_wh},
            daemon=True,
        )
        self.worker.start()

    def stop_process(self):
        if self.stop_event is not None:
            self.stop_event.set()
            self._set_status("Stopping (finishing current frame)...")
            self.stop_btn.configure(state=tk.DISABLED)

    def on_close(self):
        if self.processing and self.stop_event is not None:
            self.stop_event.set()
            if self.worker is not None:
                self.worker.join(timeout=5)
        self.root.destroy()

    # ---- queue poll (UI thread only) ----
    def poll_queue(self):
        drained = 0
        while drained < 10:
            try:
                msg = self.msg_q.get_nowait()
            except queue_mod.Empty:
                break
            drained += 1
            kind = msg[0] if msg else None
            if kind == "meta":
                _, vw, vh, _, _ = msg
                if vw and vh and (self.video_wh != (vw, vh)):
                    self._apply_preview_box(vw, vh)
            elif kind == "frame":
                _, idx, total, thumb_rgb, elapsed = msg
                self._update_progress(idx, total, elapsed)
                self._update_preview(thumb_rgb)
            elif kind == "progress":
                _, idx, total, elapsed = msg
                self._update_progress(idx, total, elapsed)
            elif kind == "summary":
                self._set_summary(msg[1])
            elif kind == "done":
                self._on_done(msg[1])
            elif kind == "error":
                self._on_error(msg[1])
        self.root.after(100, self.poll_queue)

    def _update_progress(self, idx, total, elapsed):
        pct = (idx / total * 100.0) if total else 0.0
        self.progress_var.set(min(100.0, max(0.0, pct)))
        self.progress_label.configure(text=f"Frame {idx}/{total}  {pct:.1f}%  {elapsed:.1f}s")

    def _update_preview(self, thumb_rgb):
        if not self.preview_var.get() or thumb_rgb is None or not _HAS_PIL:
            return
        try:
            img = Image.fromarray(thumb_rgb)
            self.photo = ImageTk.PhotoImage(image=img)
            self.preview_label.configure(image=self.photo, text="")
        except Exception:
            pass

    def _set_summary(self, text):
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, text or "")
        self.summary_text.configure(state=tk.DISABLED)

    def _set_status(self, text):
        self.status_label.configure(text=text)

    def _on_done(self, info):
        self.processing = False
        self.process_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.progress_var.set(100)
        stopped = info.get("stopped", False) if isinstance(info, dict) else False
        out = info.get("output_video", "") if isinstance(info, dict) else ""
        summary = info.get("summary_text", "") if isinstance(info, dict) else ""
        self._set_summary(summary)
        tag = "Stopped." if stopped else "Done."
        self._set_status(f"{tag} Output: {out}")
        self.progress_label.configure(text=f"{tag} 100%")

    def _on_error(self, msg):
        self.processing = False
        self.process_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self._set_status(f"Error: {msg}")


def main():
    root = tk.Tk()
    HoidApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

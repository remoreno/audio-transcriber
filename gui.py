"""Tkinter GUI for the audio transcriber."""

import logging
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from config import QUEUE_POLL_MS, SUPPORTED_EXTENSIONS
from file_utils import resolve_output_path, validate_input_file, UnsupportedFileError
from models import ProgressUpdate, Stage, TranscriptionRequest
from settings import UISettings, load_settings, save_settings
from subtitle_writer import write_json, write_srt, write_txt
from transcription_service import ModelLoadError, TranscriptionError, TranscriptionService

logger = logging.getLogger(__name__)

_MODEL_SIZES: tuple[str, ...] = ("tiny", "base", "small", "medium", "large-v3")
_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")
_COMPUTE_TYPES: tuple[str, ...] = ("auto", "int8", "float16", "float32")
_LANGUAGE_CHOICES: dict[str, str | None] = {
    "Spanish (forced)": "es",
    "Auto-detect": None,
}
_SENTINEL = object()
_WRITERS = {".txt": write_txt, ".srt": write_srt, ".json": write_json}


def _worker_run(
    requests: list[TranscriptionRequest],
    msg_queue: queue.Queue,
    cancel_event: threading.Event,
    model_size: str,
    device: str,
    compute_type: str,
    output_formats: set[str],
) -> None:
    """Worker thread entry point.  Communicates with the GUI only via *msg_queue*."""
    total = len(requests)
    try:
        msg_queue.put(ProgressUpdate(
            stage=Stage.LOADING_MODEL,
            message=f"Loading model '{model_size}'...",
            total_files=total,
        ))
        try:
            svc = TranscriptionService(model_size, device=device, compute_type=compute_type)
        except ModelLoadError as exc:
            msg_queue.put(ProgressUpdate(stage=Stage.ERROR, message=f"Model load failed: {exc}", total_files=total))
            return

        if svc.device != device or svc.compute_type != compute_type:
            msg_queue.put(ProgressUpdate(
                stage=Stage.LOADING_MODEL, total_files=total,
                message=f"Fell back to device={svc.device}, compute_type={svc.compute_type}",
            ))

        succeeded = 0
        failed = 0
        cancelled = False

        for file_idx, req in enumerate(requests, start=1):
            if cancel_event.is_set():
                msg_queue.put(ProgressUpdate(
                    stage=Stage.CANCELLED, message="Cancelled by user",
                    file_index=file_idx, total_files=total,
                ))
                cancelled = True
                break

            source = req.source_path
            msg_queue.put(ProgressUpdate(
                stage=Stage.TRANSCRIBING, file_path=source,
                file_index=file_idx, total_files=total,
                percent=0.0, message=f"Transcribing {source.name}...",
            ))

            def _on_progress(update: ProgressUpdate, _i: int = file_idx, _t: int = total) -> None:
                msg_queue.put(ProgressUpdate(
                    stage=update.stage, file_path=update.file_path,
                    file_index=_i, total_files=_t,
                    percent=update.percent, message=update.message,
                ))

            try:
                result = svc.transcribe(req, progress_callback=_on_progress)
            except (TranscriptionError, RuntimeError) as exc:
                failed += 1
                msg_queue.put(ProgressUpdate(
                    stage=Stage.ERROR, file_path=source,
                    file_index=file_idx, total_files=total, message=str(exc),
                ))
                continue
            except Exception as exc:
                failed += 1
                logger.exception("Unexpected error transcribing %s", source)
                msg_queue.put(ProgressUpdate(
                    stage=Stage.ERROR, file_path=source,
                    file_index=file_idx, total_files=total,
                    message=f"Unexpected error: {exc}",
                ))
                continue

            msg_queue.put(ProgressUpdate(
                stage=Stage.WRITING_OUTPUT, file_path=source,
                file_index=file_idx, total_files=total,
                percent=1.0, message=f"Writing output for {source.name}...",
            ))

            write_errors: list[str] = []
            for fmt in output_formats:
                writer_fn = _WRITERS.get(fmt)
                if writer_fn is None:
                    continue
                try:
                    out_path = resolve_output_path(source, fmt, req.output_dir)
                    writer_fn(result, out_path)
                except Exception as exc:
                    logger.exception("Failed to write %s for %s", fmt, source)
                    write_errors.append(f"{fmt}: {exc}")

            if write_errors:
                failed += 1
                for err in write_errors:
                    msg_queue.put(ProgressUpdate(
                        stage=Stage.ERROR, file_path=source,
                        file_index=file_idx, total_files=total,
                        message=f"Write failed ({err})",
                    ))
            else:
                succeeded += 1
                seg_count = len(result.segments)
                no_speech = " (no speech detected)" if seg_count == 0 else ""
                msg_queue.put(ProgressUpdate(
                    stage=Stage.DONE, file_path=source,
                    file_index=file_idx, total_files=total, percent=1.0,
                    message=f"{source.name} — {seg_count} segments, {result.processing_time:.1f}s{no_speech}",
                ))

        if not cancelled:
            msg_queue.put(ProgressUpdate(
                stage=Stage.BATCH_COMPLETE, total_files=total,
                message=f"Batch complete: {succeeded}/{total} succeeded, {failed} failed",
            ))
    except Exception:
        logger.exception("Worker thread crashed unexpectedly")
        msg_queue.put(ProgressUpdate(
            stage=Stage.ERROR,
            message="Internal error — worker thread crashed. See log file for details.",
            total_files=total,
        ))
    finally:
        msg_queue.put(_SENTINEL)


class TranscriberApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Audio Transcriber")
        self.minsize(650, 550)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._selected_files: list[Path] = []
        self._output_dir: Path | None = None
        self._msg_queue: queue.Queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        self._build_ui()
        self._apply_settings(load_settings())

    def _build_ui(self) -> None:
        """Construct all widgets."""
        pad = {"padx": 8, "pady": 4}

        # ── File selection ───────────────────────────────────────
        file_frame = ttk.LabelFrame(self, text="Files")
        file_frame.pack(fill=tk.X, **pad)

        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.X, padx=4, pady=2)
        self._file_listbox = tk.Listbox(list_frame, height=4, selectmode=tk.EXTENDED)
        file_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._file_listbox.yview)
        self._file_listbox.config(yscrollcommand=file_scroll.set)
        self._file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        btn_row = ttk.Frame(file_frame)
        btn_row.pack(fill=tk.X, padx=4, pady=(0, 4))
        self._select_btn = ttk.Button(btn_row, text="Select Files", command=self._select_files)
        self._select_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._clear_files_btn = ttk.Button(btn_row, text="Clear", command=self._clear_files)
        self._clear_files_btn.pack(side=tk.LEFT)

        # ── Output folder ────────────────────────────────────────
        out_frame = ttk.LabelFrame(self, text="Output Folder")
        out_frame.pack(fill=tk.X, **pad)
        self._output_dir_label = ttk.Label(out_frame, text="(save next to source file)")
        self._output_dir_label.pack(side=tk.LEFT, padx=4, pady=4, fill=tk.X, expand=True)
        self._browse_out_btn = ttk.Button(out_frame, text="Browse", command=self._select_output_dir)
        self._browse_out_btn.pack(side=tk.LEFT, padx=2, pady=4)
        self._clear_out_btn = ttk.Button(out_frame, text="Clear", command=self._clear_output_dir)
        self._clear_out_btn.pack(side=tk.LEFT, padx=(0, 4), pady=4)

        # ── Settings ─────────────────────────────────────────────
        settings_frame = ttk.LabelFrame(self, text="Settings")
        settings_frame.pack(fill=tk.X, **pad)

        row0 = ttk.Frame(settings_frame)
        row0.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(row0, text="Model:").pack(side=tk.LEFT)
        self._model_var = tk.StringVar(value="small")
        self._model_combo = ttk.Combobox(row0, textvariable=self._model_var, values=list(_MODEL_SIZES), state="readonly", width=10)
        self._model_combo.pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(row0, text="Device:").pack(side=tk.LEFT)
        self._device_var = tk.StringVar(value="auto")
        self._device_combo = ttk.Combobox(row0, textvariable=self._device_var, values=list(_DEVICES), state="readonly", width=8)
        self._device_combo.pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(row0, text="Compute:").pack(side=tk.LEFT)
        self._compute_var = tk.StringVar(value="auto")
        self._compute_combo = ttk.Combobox(row0, textvariable=self._compute_var, values=list(_COMPUTE_TYPES), state="readonly", width=10)
        self._compute_combo.pack(side=tk.LEFT, padx=(2, 0))

        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(row1, text="Language:").pack(side=tk.LEFT)
        self._lang_var = tk.StringVar(value="Spanish (forced)")
        self._lang_combo = ttk.Combobox(row1, textvariable=self._lang_var, values=list(_LANGUAGE_CHOICES.keys()), state="readonly", width=18)
        self._lang_combo.pack(side=tk.LEFT, padx=(2, 0))

        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, padx=4, pady=(2, 4))
        ttk.Label(row2, text="Output:").pack(side=tk.LEFT)
        self._txt_var = tk.BooleanVar(value=True)
        self._txt_chk = ttk.Checkbutton(row2, text="TXT", variable=self._txt_var)
        self._txt_chk.pack(side=tk.LEFT, padx=(2, 8))
        self._srt_var = tk.BooleanVar(value=True)
        self._srt_chk = ttk.Checkbutton(row2, text="SRT", variable=self._srt_var)
        self._srt_chk.pack(side=tk.LEFT, padx=(0, 8))
        self._json_var = tk.BooleanVar(value=True)
        self._json_chk = ttk.Checkbutton(row2, text="JSON", variable=self._json_var)
        self._json_chk.pack(side=tk.LEFT)

        # ── Controls ─────────────────────────────────────────────
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, **pad)
        self._start_btn = ttk.Button(ctrl_frame, text="Transcribe", command=self._start_transcription)
        self._start_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._cancel_btn = ttk.Button(ctrl_frame, text="Cancel", command=self._cancel_transcription, state=tk.DISABLED)
        self._cancel_btn.pack(side=tk.LEFT)

        # ── Progress ─────────────────────────────────────────────
        prog_frame = ttk.LabelFrame(self, text="Progress")
        prog_frame.pack(fill=tk.X, **pad)
        self._status_label = ttk.Label(prog_frame, text="Ready")
        self._status_label.pack(fill=tk.X, padx=4, pady=(4, 0))
        self._file_label = ttk.Label(prog_frame, text="")
        self._file_label.pack(fill=tk.X, padx=4)
        self._progress_bar = ttk.Progressbar(prog_frame, mode="determinate", maximum=100)
        self._progress_bar.pack(fill=tk.X, padx=4, pady=(0, 4))

        # ── Log ──────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, **pad)
        self._log_text = tk.Text(log_frame, height=8, state=tk.DISABLED, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self._log_text.yview)
        self._log_text.config(yscrollcommand=log_scroll.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=4)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4), pady=4)
        self._log_text.tag_configure("error", foreground="#CC0000")
        self._log_text.tag_configure("success", foreground="#006400")
        self._log_text.tag_configure("info", foreground="#555555")

    # ── File/folder selection ────────────────────────────────────────────

    def _select_files(self) -> None:
        """Open a multi-file dialog and populate the file list."""
        ext_filter = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        paths = filedialog.askopenfilenames(
            title="Select audio/video files",
            filetypes=[("Audio/Video files", ext_filter), ("All files", "*.*")],
        )
        if not paths:
            return
        self._selected_files = [Path(p) for p in paths]
        self._file_listbox.delete(0, tk.END)
        for p in self._selected_files:
            self._file_listbox.insert(tk.END, p.name)

    def _clear_files(self) -> None:
        self._selected_files.clear()
        self._file_listbox.delete(0, tk.END)

    def _select_output_dir(self) -> None:
        d = filedialog.askdirectory(title="Select output folder")
        if not d:
            return
        self._output_dir = Path(d)
        self._output_dir_label.config(text=str(self._output_dir))

    def _clear_output_dir(self) -> None:
        self._output_dir = None
        self._output_dir_label.config(text="(save next to source file)")

    # ── Transcription lifecycle ──────────────────────────────────────────

    def _start_transcription(self) -> None:
        """Validate inputs, build requests, and launch the worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        if not self._selected_files:
            messagebox.showwarning("No files", "Please select at least one file.")
            return

        output_formats: set[str] = set()
        if self._txt_var.get():
            output_formats.add(".txt")
        if self._srt_var.get():
            output_formats.add(".srt")
        if self._json_var.get():
            output_formats.add(".json")
        if not output_formats:
            messagebox.showwarning("No output format", "Select at least one output format.")
            return

        # Clear log and reset progress
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.config(state=tk.DISABLED)
        self._progress_bar["value"] = 0
        self._status_label.config(text="Validating files...")
        self._file_label.config(text="")

        # Validate files on the main thread (fast I/O checks)
        language = _LANGUAGE_CHOICES[self._lang_var.get()]
        requests: list[TranscriptionRequest] = []
        skipped = 0
        for path in self._selected_files:
            try:
                validated = validate_input_file(path)
            except (FileNotFoundError, UnsupportedFileError) as exc:
                self._log(f"Skipped: {path.name} -- {exc}", "error")
                skipped += 1
                continue
            requests.append(TranscriptionRequest(
                source_path=validated, language=language, output_dir=self._output_dir,
            ))

        if not requests:
            self._log("No valid files to process.", "error")
            self._status_label.config(text="Ready")
            return

        if skipped:
            self._log(f"{skipped} file(s) skipped during validation.", "info")

        # Launch worker
        self._cancel_event.clear()
        self._msg_queue = queue.Queue()
        self._set_controls_enabled(False)

        self._worker_thread = threading.Thread(
            target=_worker_run,
            args=(
                requests, self._msg_queue, self._cancel_event,
                self._model_var.get(), self._device_var.get(),
                self._compute_var.get(), output_formats,
            ),
            daemon=True,
        )
        self._worker_thread.start()
        self.after(QUEUE_POLL_MS, self._poll_queue)

    def _cancel_transcription(self) -> None:
        """Signal the worker thread to stop after the current file."""
        self._cancel_event.set()
        self._cancel_btn.config(state=tk.DISABLED)
        self._status_label.config(text="Cancelling...")

    # ── Queue polling ────────────────────────────────────────────────────

    def _poll_queue(self) -> None:
        """Drain the message queue and update the UI accordingly."""
        while True:
            try:
                msg = self._msg_queue.get_nowait()
            except queue.Empty:
                break

            if msg is _SENTINEL:
                self._set_controls_enabled(True)
                self._worker_thread = None
                return

            if not isinstance(msg, ProgressUpdate):
                continue

            stage = msg.stage
            if stage == Stage.LOADING_MODEL:
                self._status_label.config(text=msg.message)
                self._log(msg.message, "info")
            elif stage == Stage.TRANSCRIBING:
                self._status_label.config(text=f"File {msg.file_index}/{msg.total_files}: {msg.message}")
                if msg.file_path:
                    self._file_label.config(text=msg.file_path.name)
                self._progress_bar["value"] = msg.percent * 100
            elif stage == Stage.WRITING_OUTPUT:
                self._status_label.config(text=msg.message)
            elif stage == Stage.DONE:
                self._log(f"[OK] {msg.message}", "success")
                self._progress_bar["value"] = 100
            elif stage == Stage.ERROR:
                self._log(f"[ERROR] {msg.message}", "error")
            elif stage == Stage.BATCH_COMPLETE:
                self._status_label.config(text=msg.message)
                self._file_label.config(text="")
                self._progress_bar["value"] = 0
                self._log(msg.message, "info")
            elif stage == Stage.CANCELLED:
                self._status_label.config(text=msg.message)
                self._log(msg.message, "info")

        self.after(QUEUE_POLL_MS, self._poll_queue)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Toggle all input controls and the cancel button."""
        btn_state = tk.NORMAL if enabled else tk.DISABLED
        combo_state = "readonly" if enabled else "disabled"

        for w in (self._select_btn, self._clear_files_btn,
                  self._browse_out_btn, self._clear_out_btn, self._start_btn):
            w.config(state=btn_state)

        for c in (self._model_combo, self._device_combo,
                  self._compute_combo, self._lang_combo):
            c.config(state=combo_state)

        for chk in (self._txt_chk, self._srt_chk, self._json_chk):
            chk.config(state=btn_state)

        self._file_listbox.config(state=btn_state)
        self._cancel_btn.config(state=tk.DISABLED if enabled else tk.NORMAL)

    def _log(self, message: str, tag: str = "") -> None:
        """Append a line to the log widget."""
        self._log_text.config(state=tk.NORMAL)
        if tag:
            self._log_text.insert(tk.END, message + "\n", tag)
        else:
            self._log_text.insert(tk.END, message + "\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _gather_settings(self) -> UISettings:
        """Snapshot the current UI state into a settings object."""
        out_dir = str(self._output_dir) if self._output_dir is not None else None
        return UISettings(
            model=self._model_var.get(),
            device=self._device_var.get(),
            compute_type=self._compute_var.get(),
            language=self._lang_var.get(),
            fmt_txt=self._txt_var.get(),
            fmt_srt=self._srt_var.get(),
            fmt_json=self._json_var.get(),
            output_dir=out_dir,
            window_x=self.winfo_x(),
            window_y=self.winfo_y(),
            window_width=self.winfo_width(),
            window_height=self.winfo_height(),
        )

    def _apply_settings(self, s: UISettings) -> None:
        """Restore widget state from a settings object."""
        self._model_var.set(s.model)
        self._device_var.set(s.device)
        self._compute_var.set(s.compute_type)
        self._lang_var.set(s.language)
        self._txt_var.set(s.fmt_txt)
        self._srt_var.set(s.fmt_srt)
        self._json_var.set(s.fmt_json)
        if s.output_dir is not None:
            p = Path(s.output_dir)
            if p.is_dir():
                self._output_dir = p
                self._output_dir_label.config(text=str(p))
        self._restore_window_geometry(s)

    def _restore_window_geometry(self, s: UISettings) -> None:
        """Restore saved window position/size, or center on screen if no saved geometry."""
        self.update_idletasks()
        if (s.window_x is not None and s.window_y is not None
                and s.window_width is not None and s.window_height is not None):
            self.geometry(f"{s.window_width}x{s.window_height}+{s.window_x}+{s.window_y}")
        else:
            w = self.winfo_width()
            h = self.winfo_height()
            title_bar_h = self.winfo_rooty() - self.winfo_y()
            outer_h = h + title_bar_h
            work = self._get_work_area()
            if work is not None:
                work_x, work_y, work_w, work_h = work
            else:
                work_x, work_y = 0, 0
                work_w = self.winfo_screenwidth()
                work_h = self.winfo_screenheight()
            x = work_x + (work_w - w) // 2
            y = work_y + (work_h - outer_h) // 2
            self.geometry(f"+{x}+{y}")

    @staticmethod
    def _get_work_area() -> tuple[int, int, int, int] | None:
        """Return (x, y, width, height) of the desktop work area (excludes taskbar)."""
        try:
            import ctypes
            from ctypes import wintypes

            rect = wintypes.RECT()
            ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
            return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
        except Exception:
            return None

    def _on_close(self) -> None:
        """Handle window close — save settings, cancel any running worker, then destroy."""
        save_settings(self._gather_settings())
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._cancel_event.set()
        self.destroy()

"""Entry point for the audio transcriber application."""

import sys
import tkinter as tk
from tkinter import messagebox

from file_utils import check_ffmpeg
from logging_utils import setup_logging


def main() -> None:
    """Initialise logging, verify prerequisites, and launch the GUI."""
    setup_logging()

    if not check_ffmpeg():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "ffmpeg not found",
            "ffmpeg is required but was not found on your system PATH.\n\n"
            "Please install ffmpeg and ensure it is accessible from the command line.",
        )
        root.destroy()
        sys.exit(1)

    from gui import TranscriberApp

    app = TranscriberApp()
    app.mainloop()


if __name__ == "__main__":
    main()

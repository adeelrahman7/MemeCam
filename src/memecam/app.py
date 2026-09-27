"""Entry point: parse args, load config, start the worker and window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from memecam.config import ConfigError, load_settings
from memecam.core.frame_worker import FrameWorker
from memecam.paths import default_config_path, resource_root
from memecam.ui.main_window import MainWindow


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="memecam", description="Gesture-triggered meme reactions."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="path to reactions.json (default: config/reactions.json)",
    )
    parser.add_argument("--debug", action="store_true", help="start with the debug overlay on")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("MemeCam")

    root = resource_root()
    config_path = args.config or default_config_path()
    try:
        settings = load_settings(config_path, root)
    except ConfigError as exc:
        print(f"Config error:\n{exc}", file=sys.stderr)
        QMessageBox.critical(None, "MemeCam: config error", str(exc))
        return 2

    worker = FrameWorker(settings)
    window = MainWindow(worker, settings)
    if args.debug:
        window.enable_debug()
    window.show()
    worker.start()
    try:
        return app.exec()
    finally:
        worker.stop()


if __name__ == "__main__":
    raise SystemExit(main())

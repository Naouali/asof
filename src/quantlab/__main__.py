"""Allow `python -m quantlab`, which the scheduler uses to launch jobs."""

from __future__ import annotations

from quantlab.cli import app

if __name__ == "__main__":
    app()

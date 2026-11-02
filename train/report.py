"""T047: build both tables and results/report.md from saved results only (no weights loaded)."""
import argparse

from _cli import apply_roots

ap = argparse.ArgumentParser()
ap.add_argument("--results-root", default=None)
ap.add_argument("--data-prep", default=None)
a = ap.parse_args()
apply_roots(a)
from src.reporting.report import write_report  # noqa: E402

print(write_report())

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def apply_roots(args) -> None:
    """--results-root / --data-prep override output locations (used by smoke tests)."""
    if getattr(args, "results_root", None):
        os.environ["CRR_RESULTS_ROOT"] = str(Path(args.results_root).resolve())
    if getattr(args, "data_prep", None):
        os.environ["CRR_DATA_PREP"] = str(Path(args.data_prep).resolve())


def add_common(ap) -> None:
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--data-prep", default=None)

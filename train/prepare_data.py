"""T010: fix paths, CLAHE, offline augmentation, derived labels, leakage check."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.prepare import prepare_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="use only the first N images per split (smoke tests)")
    ap.add_argument("--no-clean", action="store_true")
    a = ap.parse_args()
    m = prepare_all(limit=a.limit, clean=not a.no_clean)
    print("prepared:", m["counts"])
    print("leakage check:", m["leakage_check"])

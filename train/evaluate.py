"""T024/T037: predict (if needed), choose tau on val only, compute test metrics, write results."""
import argparse
import json

from _cli import add_common, apply_roots

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["detection", "segmentation"])
ap.add_argument("--model", required=True)
ap.add_argument("--split", default="test", help="reporting split; only 'test' is allowed")
ap.add_argument("--repredict", action="store_true", help="regenerate predictions from saved weights")
add_common(ap)
a = ap.parse_args()
apply_roots(a)
if a.split != "test":
    raise SystemExit("Only the test split is reported; tau/checkpoints are selected on val inside the pipeline.")
from src.evaluation.run_eval import evaluate_detection_model, evaluate_segmentation_model  # noqa: E402

m = (evaluate_detection_model if a.task == "detection" else evaluate_segmentation_model)(a.model, a.seed, a.repredict)
print(json.dumps({k: v for k, v in m.items() if k != "per_image"}, indent=2, default=str))

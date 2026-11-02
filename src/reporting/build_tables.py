"""Comparison tables built only from saved metrics files (T025, T038, T044)."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from src import common

DET_HEADER = ["模型", "τ", "AP@[.5:.95]", "AP@.5", "Precision [95% CI]", "Recall [95% CI]", "F1 [95% CI]"]
SEG_HEADER = ["模型", "Dice", "IoU", "Precision", "Recall", "HD95(px)", "Boundary F1"]
DAGGER = "†"
FOOTNOTE = (f"{DAGGER} instance-segmentation model, scored on both tasks; trained/tested on the segmentation dataset, "
            "so not directly comparable with box-only rows.")


def display_name(model: str, meta: dict | None = None) -> str:
    return common.MODEL_REGISTRY.get(model, {}).get("display") or (meta or {}).get("display") or model


def _ci(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['value']:.3f} [{lo:.3f}, {hi:.3f}]"


def _pm(d: dict, nd: int) -> str:
    if d is None or d.get("mean") is None or (isinstance(d["mean"], float) and math.isnan(d["mean"])):
        return "n/a"
    if d.get("std") is None:
        return f"{d['mean']:.{nd}f} (1 seed)"
    return f"{d['mean']:.{nd}f} ± {d['std']:.{nd}f}"


def _ordered(root: Path, preferred: list[str]) -> list[str]:
    present = [p.name for p in root.iterdir() if p.is_dir()] if root.exists() else []
    return [m for m in preferred if m in present] + sorted(m for m in present if m not in preferred)


def detection_rows() -> list[dict]:
    root = common.results_root() / "detection"
    rows = []
    for m in _ordered(root, common.DET_MODELS + common.CROSS_TASK_MODELS):
        f = root / m / "metrics.json"
        if f.exists():
            rows.append(common.read_json(f))
    return rows


def segmentation_rows() -> list[dict]:
    root = common.results_root() / "segmentation"
    rows = []
    for m in _ordered(root, common.SEG_MODELS):
        f = root / m / "summary.json"
        if f.exists():
            rows.append(common.read_json(f))
    return rows


def build_detection_table() -> tuple[list[list[str]], list[dict]]:
    rows = detection_rows()
    table = []
    for r in rows:
        name = display_name(r["model"], r) + (DAGGER if r.get("cross_task") else "")
        table.append([name, f"{r['tau']:.3f}", f"{r['ap_50_95']:.3f}", f"{r['ap_50']:.3f}",
                      _ci(r["precision"]), _ci(r["recall"]), _ci(r["f1"])])
    return table, rows


def build_segmentation_table() -> tuple[list[list[str]], list[dict]]:
    rows = segmentation_rows()
    table = []
    for r in rows:
        name = display_name(r["model"], r) + (DAGGER if r.get("cross_task") else "")
        table.append([name, _pm(r["dice"], 4), _pm(r["iou"], 4), _pm(r["precision"], 4), _pm(r["recall"], 4),
                      _pm(r["hd95_px"], 2), _pm(r["boundary_f1"], 4)])
    return table, rows


def to_markdown(header: list[str], rows: list[list[str]], footnotes: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n\n" + "\n".join(f"> {n}" for n in footnotes) + "\n"


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_tables() -> dict:
    root = common.results_root()
    det, det_rows = build_detection_table()
    seg, seg_rows = build_segmentation_table()
    cross = any(r.get("cross_task") for r in det_rows + seg_rows)
    dnotes = ["Test split only; τ chosen on the validation split (max F1 at IoU 0.5); CI = 95% percentile bootstrap over test images."]
    snotes = [f"Test split only; mean ± std over seeds ({seg_rows[0]['n_seeds'] if seg_rows else 'n/a'}); "
              f"Boundary F1 tolerance = {seg_rows[0].get('bf1_tolerance_px') if seg_rows else 'n/a'} px."]
    if cross:
        dnotes.append(FOOTNOTE)
        snotes.append(FOOTNOTE)
    (root / "detection").mkdir(parents=True, exist_ok=True)
    (root / "segmentation").mkdir(parents=True, exist_ok=True)
    (root / "detection" / "detection_table.md").write_text(to_markdown(DET_HEADER, det, dnotes), encoding="utf-8")
    write_csv(root / "detection" / "detection_table.csv", DET_HEADER, det)
    (root / "segmentation" / "segmentation_table.md").write_text(to_markdown(SEG_HEADER, seg, snotes), encoding="utf-8")
    write_csv(root / "segmentation" / "segmentation_table.csv", SEG_HEADER, seg)
    return {"detection": det, "segmentation": seg, "det_rows": det_rows, "seg_rows": seg_rows,
            "det_notes": dnotes, "seg_notes": snotes}

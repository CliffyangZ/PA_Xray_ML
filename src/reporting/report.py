"""results/report.md from saved metrics only; includes the leakage audit (T047, T049)."""
from __future__ import annotations

import itertools

from src import common
from src.reporting import build_tables as bt


def leakage_audit(det_rows: list[dict]) -> list[str]:
    problems = []
    for r in det_rows:
        if r.get("split") != "test":
            problems.append(f"{r['model']}: metrics split is {r.get('split')!r}, expected 'test'")
        if r.get("tau_selected_on") != "val":
            problems.append(f"{r['model']}: tau_selected_on is {r.get('tau_selected_on')!r}, expected 'val'")
    seg_root = common.results_root() / "segmentation"
    for f in sorted(seg_root.glob("*/seed_*/metrics.json")):
        m = common.read_json(f)
        if m.get("split") != "test":
            problems.append(f"{f}: split {m.get('split')!r}")
        if m.get("tau") is not None and m.get("tau_selected_on") != "val":
            problems.append(f"{f}: tau not selected on val")
    man = common.data_prep_root() / "manifest.json"
    if man.exists():
        if not all(v.get("ok") for v in common.read_json(man)["leakage_check"].values()):
            problems.append("data_prep manifest reports split leakage")
    else:
        problems.append("data_prep/manifest.json missing; split-leakage check not recorded")
    return problems


def overlap_notes(det_rows: list[dict], seg_rows: list[dict]) -> list[str]:
    notes = []
    box_only = [r for r in det_rows if not r.get("cross_task")]
    for a, b in itertools.combinations(box_only, 2):
        (l1, h1), (l2, h2) = a["f1"]["ci95"], b["f1"]["ci95"]
        if max(l1, l2) <= min(h1, h2):
            notes.append(f"Detection: F1 CIs of {bt.display_name(a['model'])} and {bt.display_name(b['model'])} overlap; no ranking claimed.")
    for key in ("dice", "iou"):
        for a, b in itertools.combinations(seg_rows, 2):
            sa, sb = a[key].get("std"), b[key].get("std")
            if sa is None or sb is None:
                continue
            if abs(a[key]["mean"] - b[key]["mean"]) <= sa + sb:
                notes.append(f"Segmentation: {key} of {bt.display_name(a['model'])} and {bt.display_name(b['model'])} differ by less than the summed std; no ranking claimed.")
    return notes


def write_report() -> str:
    t = bt.write_tables()
    problems = leakage_audit(t["det_rows"])
    notes = overlap_notes(t["det_rows"], t["seg_rows"])
    det_md = bt.to_markdown(bt.DET_HEADER, t["detection"], t["det_notes"])
    seg_md = bt.to_markdown(bt.SEG_HEADER, t["segmentation"], t["seg_notes"])
    audit = "PASS: no test-split leakage found." if not problems else "FAIL:\n" + "\n".join(f"- {p}" for p in problems)
    text = (f"# Model comparison report\n\n## Detection\n\n{det_md}\n## Segmentation\n\n{seg_md}\n"
            f"## Uncertainty notes\n\n{chr(10).join('- ' + n for n in notes) or '- No overlapping intervals detected among compared rows.'}\n\n"
            f"## Test-leakage audit\n\n{audit}\n")
    (common.results_root() / "report.md").write_text(text, encoding="utf-8")
    return text

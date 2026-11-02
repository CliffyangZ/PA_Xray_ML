"""U-Net family training + prediction export (T032)."""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import common
from src.data.seg_dataset import SegDataset, unletterbox
from src.evaluation import seg_metrics
from src.models import build_unet


def dice_bce_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def predict_probs(model, ds: SegDataset, device, batch: int = 4):
    """Yield (stem, full-resolution probability map) for every item in `ds`."""
    model.eval()
    for x, _, meta, stems in DataLoader(ds, batch_size=batch, shuffle=False):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x.to(device))).float().cpu().numpy()[:, 0]
        for pi, m, s in zip(p, meta.tolist(), stems):
            yield s, unletterbox(pi, (m[0], m[1]), (m[2], m[3]))


def _val_dice(model, ds, device, thr: float) -> float:
    vals = []
    for stem, prob in predict_probs(model, ds, device):
        gt = cv2.imread(str(ds.mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        vals.append(seg_metrics.overlap_metrics(prob >= thr, gt)["dice"])
    return float(np.mean(vals))


def export_masks(model_name: str, seed: int, splits=("val", "test"), device=None) -> None:
    cfg = common.model_cfg(model_name)
    thr = common.data_cfg()["evaluation"]["seg_threshold"]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = common.seg_run_dir(model_name, seed)
    model = build_unet(cfg).to(device)
    model.load_state_dict(torch.load(run / "weights" / "best.pt", map_location=device)["model"])
    for split in splits:
        out = run / "pred_masks" / split
        out.mkdir(parents=True, exist_ok=True)
        for stem, prob in predict_probs(model, SegDataset(split, cfg["imgsz"]), device):
            cv2.imwrite(str(out / f"{stem}.png"), (prob >= thr).astype(np.uint8))


def train_unet(model_name: str, seed: int, epochs: int | None = None, limit: int | None = None, device=None) -> dict:
    cfg = common.model_cfg(model_name)
    if epochs:
        cfg["epochs"] = epochs
    common.set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thr = common.data_cfg()["evaluation"]["seg_threshold"]
    run = common.seg_run_dir(model_name, seed)
    (run / "weights").mkdir(parents=True, exist_ok=True)

    train_ds = SegDataset("train", cfg["imgsz"], limit)
    val_ds = SegDataset("val", cfg["imgsz"], limit)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True, generator=g, drop_last=len(train_ds) > cfg["batch"])
    model = build_unet(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best, best_ep, history, t0 = -1.0, -1, [], time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        tot = 0.0
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = dice_bce_loss(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x)
        sched.step()
        vd = _val_dice(model, val_ds, device, thr)
        history.append({"epoch": ep, "train_loss": tot / len(train_ds), "val_dice": vd})
        print(f"[{model_name} s{seed}] epoch {ep + 1}/{cfg['epochs']} loss={tot / len(train_ds):.4f} val_dice={vd:.4f}", flush=True)
        if vd > best:
            best, best_ep = vd, ep
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep}, run / "weights" / "best.pt")

    info = {"model": model_name, "family": "unet", "seed": seed, "config": cfg, "config_hash": common.stable_hash(cfg),
            "data_hash": common.data_hash(), "best_epoch": best_ep, "best_val_dice": best, "device": str(device),
            "train_images": len(train_ds), "train_seconds": time.time() - t0, "history": history}
    common.write_json(run / "config.json", info)
    export_masks(model_name, seed, device=device)
    return info

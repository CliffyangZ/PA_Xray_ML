"""Mask R-CNN training + prediction export (T042)."""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import common
from src.data import convert
from src.evaluation import seg_metrics
from src.evaluation.merge_instances import encode_mask, merge_instances
from src.models.maskrcnn import build_maskrcnn


class InstanceDataset(Dataset):
    """Prepared segmentation split -> (image tensor, torchvision target)."""

    def __init__(self, split: str, limit: int | None = None):
        self.root = common.data_prep_root() / "segmentation" / split
        self.gt = common.read_json(self.root / "gt.json")
        self.images = self.gt["images"][: limit or None]
        self.by_img: dict[int, list] = {}
        for a in self.gt["annotations"]:
            self.by_img.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        gray = cv2.imread(str(self.root / "images" / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        x = torch.from_numpy(gray).float().div(255.0).unsqueeze(0).repeat(3, 1, 1)
        anns = self.by_img.get(im["id"], [])
        h, w = gray.shape
        masks = [convert.polygons_to_mask(a["segmentation"], h, w) for a in anns]
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
        t = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
             "labels": torch.ones(len(anns), dtype=torch.int64),
             "masks": torch.from_numpy(np.stack(masks)) if masks else torch.zeros((0, h, w), dtype=torch.uint8)}
        return x, t, im["file_name"].rsplit(".", 1)[0]


def _collate(b):
    return tuple(zip(*b))


@torch.no_grad()
def predict_instances(model, ds: InstanceDataset, device, mask_thr: float = 0.5):
    model.eval()
    for i in range(len(ds)):
        x, _, stem = ds[i]
        out = model([x.to(device)])[0]
        recs = []
        for b, s, m in zip(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["masks"].cpu().numpy()):
            recs.append({"image_id": stem, "category_id": 1, "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                         "score": float(s), "segmentation": encode_mask(m[0] >= mask_thr)})
        yield stem, recs, tuple(x.shape[-2:])


def _val_dice(model, ds, device) -> float:
    vals = []
    for stem, recs, shape in predict_instances(model, ds, device):
        gt = cv2.imread(str(ds.root / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        pred = merge_instances(recs, 0.5, shape)
        vals.append(seg_metrics.overlap_metrics(pred, gt)["dice"])
    return float(np.mean(vals))


def export_predictions(model_name: str, seed: int, splits=("val", "test"), device=None) -> None:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = common.seg_run_dir(model_name, seed)
    model, _ = build_maskrcnn(pretrained=False)
    model.load_state_dict(torch.load(run / "weights" / "best.pt", map_location=device)["model"])
    model.to(device)
    for split in splits:
        recs = []
        for _, r, _ in predict_instances(model, InstanceDataset(split), device):
            recs.extend(r)
        common.write_json(run / f"predictions_{split}.json", recs)


def train_maskrcnn(model_name: str, seed: int, epochs: int | None = None, limit: int | None = None, device=None) -> dict:
    cfg = common.model_cfg(model_name)
    if epochs:
        cfg["epochs"] = epochs
    common.set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = common.seg_run_dir(model_name, seed)
    (run / "weights").mkdir(parents=True, exist_ok=True)
    model, init_note = build_maskrcnn(cfg["pretrained"])
    model.to(device)
    train_ds, val_ds = InstanceDataset("train", limit), InstanceDataset("val", limit)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True, collate_fn=_collate, generator=g)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best, best_ep, history, t0 = -1.0, -1, [], time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        tot = 0.0
        for xs, ts, _ in loader:
            xs = [x.to(device) for x in xs]
            ts = [{k: v.to(device) for k, v in t.items()} for t in ts]
            with torch.autocast(device.type, enabled=use_amp):
                loss = sum(model(xs, ts).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        sched.step()
        vd = _val_dice(model, val_ds, device)
        history.append({"epoch": ep, "train_loss": tot / max(len(loader), 1), "val_dice": vd})
        print(f"[maskrcnn s{seed}] epoch {ep + 1}/{cfg['epochs']} loss={tot / max(len(loader), 1):.4f} val_dice={vd:.4f}", flush=True)
        if vd > best:
            best, best_ep = vd, ep
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep}, run / "weights" / "best.pt")

    info = {"model": model_name, "family": "maskrcnn", "seed": seed, "config": cfg, "config_hash": common.stable_hash(cfg),
            "data_hash": common.data_hash(), "init": init_note, "best_epoch": best_ep, "best_val_dice": best,
            "train_images": len(train_ds), "train_seconds": time.time() - t0, "history": history}
    common.write_json(run / "config.json", info)
    export_predictions(model_name, seed, device=device)
    return info

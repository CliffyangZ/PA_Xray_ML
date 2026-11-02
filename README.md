# X-ray PA tooth detection & segmentation model comparison

Training / evaluation framework for the spec in `specs/001-xray-tooth-model-comparison/`.
Everything runs in the Conda env `mcv_env`.

```bash
conda activate mcv_env
```

## Models
| Task | Models |
|---|---|
| Detection | YOLO v8s, v11s, v26s |
| Segmentation | U-Net, U-Net++, Attention U-Net, YOLO v8s/v11s/v26s-seg |
| Cross-task (both tables, marked †) | Mask R-CNN, YOLO-seg models |

## Pipeline
```bash
python train/prepare_data.py          # fix data.yaml path, CLAHE, offline augmentation (train only), derived labels, leakage check

# detection (seed 42)
for m in yolov8s yolo11s yolo26s; do
  python train/train_detection.py --model $m --seed 42
  python train/evaluate.py --task detection --model $m
done

# segmentation + cross-task (3 seeds)
for m in unet unetpp attention_unet yolov8s-seg yolo11s-seg yolo26s-seg maskrcnn; do
  for s in 42 43 44; do
    python train/train_segmentation.py --model $m --seed $s
    python train/evaluate.py --task segmentation --model $m --seed $s
  done
done

python train/report.py                # tables + results/report.md from saved results only
pytest tests/
```
`evaluate.py --task segmentation` on a cross-task model also writes its detection row (seed 42).

## Layout
- `configs/` hyperparameters (`data.yaml` holds CLAHE / augmentation / evaluation settings)
- `src/data` preparation, augmentation, conversion, leakage checks
- `src/models` U-Net family, Mask R-CNN
- `src/training` YOLO, U-Net, Mask R-CNN trainers (+ prediction export)
- `src/evaluation` shared metrics, bootstrap CI, instance merging, cross-task evaluation
- `src/reporting` tables and report
- `train/` command-line entry points
- `data_prep/` generated data (CLAHE'd, augmented); `data/` is never modified except the `path:` fix in `data/detection_dataset/data.yaml`
- `results/` outputs, see `specs/001-xray-tooth-model-comparison/contracts/result-schemas.md`

## Smoke tests (fast, do not overwrite real results)
Use `--results-root <dir>` with small `--epochs` and `--limit` / `--fraction`, e.g.
```bash
python train/train_segmentation.py --model unet --epochs 2 --limit 8 --results-root /tmp/smoke
python train/evaluate.py --task segmentation --model unet --results-root /tmp/smoke
```

## Notes
- Test split is used only by `evaluate.py`; tau (detection / cross-task) is chosen on val.
- YOLO built-in augmentation is disabled; all models see the same offline-augmented training set.
- `*-seg.pt` and Mask R-CNN COCO weights are downloaded on first use (network); if a `*-seg.pt` download fails, the matching detection weights are loaded into the `-seg` architecture and the fallback is recorded in `config.json`.
# PA_Xray_ML

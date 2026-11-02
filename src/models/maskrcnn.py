"""Mask R-CNN (torchvision, ResNet50-FPN), 2 classes: background + tooth (T041)."""
import warnings

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_maskrcnn(pretrained: bool = True, box_score_thresh: float = 0.001, detections_per_img: int = 100):
    """Returns (model, init_note). Falls back to random init if COCO weights cannot be loaded."""
    note = "coco_pretrained"
    try:
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=weights, weights_backbone=None if not pretrained else torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
            box_score_thresh=box_score_thresh, box_detections_per_img=detections_per_img)
        if not pretrained:
            note = "random_init"
    except Exception as e:  # offline: weights download failed
        warnings.warn(f"Mask R-CNN pretrained weights unavailable ({e}); using random init")
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, box_score_thresh=box_score_thresh,
            box_detections_per_img=detections_per_img)
        note = "random_init_fallback"
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, 2)
    return model, note

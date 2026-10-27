"""Segmentation losses shared by both retained training methods."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from model.sam2.postprocess import upsample_mask_logits


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_objects: float,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    probability = inputs.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    probability_t = probability * targets + (1 - probability) * (1 - targets)
    loss = cross_entropy * ((1 - probability_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.flatten(2).mean(-1) / num_objects


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_objects: float,
) -> torch.Tensor:
    probabilities = inputs.sigmoid().flatten(2)
    flattened_targets = targets.flatten(2)
    numerator = 2 * (probabilities * flattened_targets).sum(-1)
    denominator = probabilities.sum(-1) + flattened_targets.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)) / num_objects


def segmentation_loss(logits, target, cfg):
    truth = target[:, None].float()
    count = float(target.shape[0])
    focal = sigmoid_focal_loss(
        logits, truth, count, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma
    ).sum()
    dice = dice_loss(logits, truth, count).sum()
    return cfg.focal_weight * focal + cfg.dice_weight * dice, {
        "focal_loss": focal,
        "dice_loss": dice,
    }


def lora_loss(logits, target, predicted_iou, cfg):
    high_resolution = upsample_mask_logits(logits, target.shape[-2:])
    truth = target[:, None].float()
    count = float(target.shape[0])
    focal = sigmoid_focal_loss(high_resolution, truth, count, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma).sum()
    dice = dice_loss(high_resolution, truth, count).sum()
    predicted_masks = high_resolution.flatten(2) > 0
    target_masks = truth.flatten(2) > 0
    intersection = (predicted_masks & target_masks).sum(-1).float()
    union = (predicted_masks | target_masks).sum(-1).float()
    actual_iou = intersection / union.clamp_min(1.0)
    quality = (F.l1_loss(predicted_iou, actual_iou, reduction="none") / count).sum()
    loss = cfg.focal_weight * focal + cfg.dice_weight * dice + cfg.iou_weight * quality
    return loss, dict(focal_loss=focal, dice_loss=dice, iou_loss=quality), high_resolution

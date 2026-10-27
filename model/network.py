"""Frozen SAM2+LoRA backbone with a trainable ROI refiner."""

import torch
from dataclasses import asdict
from torch import nn
from model.sam2.build_sam import build_sam2
from model.lora import QueryValueLoRALinear
from model.refiner import Block1Head, expand_boxes
from model.refiner import ReferenceHead


def make_head(settings):
    """Historical Full evaluation path, preserving checkpoint keys."""
    return Block1Head(width=settings["refiner_width"],
                      max_logit_residual=settings["max_logit_residual"])


class Refiner(nn.Module):
    def __init__(self, base_model, head, context_scale):
        super().__init__()
        self.base_model = base_model.requires_grad_(False).eval()
        self.refine_head = head
        self.context_scale = context_scale

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward_refined(self, images, boxes, *, roi_boxes=None):
        if roi_boxes is None:
            roi_boxes = expand_boxes(boxes, images.shape[-2:], self.context_scale)
        coarse, _, features = self.refine_head.extract_frozen_inputs(
            self.base_model, images, boxes
        )
        return self.refine_head(
            coarse, roi_boxes, images.shape[-2:], decoder_features=features
        )

    def forward(self, images, boxes, *, roi_boxes=None):
        return self.forward_refined(images, boxes, roi_boxes=roi_boxes).logits


def build_model(cfg, device="cuda", source_lora=None, *, training=False, base_state=None):
    base = build_sam2(base_state if base_state is not None else cfg.checkpoint, device=device, image_size=cfg.image_size)
    source = source_lora or torch.load(
        cfg.source_lora_checkpoint, map_location="cpu", weights_only=False
    )
    source = source.get("source_lora", source)
    settings = source["config"]
    for index, block in enumerate(base.image_encoder.trunk.blocks):
        block.attn.qkv = QueryValueLoRALinear(
            block.attn.qkv,
            rank=settings["rank"],
            alpha=settings["alpha"],
            dropout=settings["dropout"],
            layer_name=f"blocks.{index}.attn.qkv",
        )
    suffixes = tuple(
        f".{factor}.weight" for factor in QueryValueLoRALinear.factor_names
    )
    with torch.no_grad():
        for name, parameter in base.named_parameters():
            if name.endswith(suffixes):
                parameter.copy_(source["lora_state"][name])
    # Full's historical training used the reference forward; evaluation uses
    # the unified stage implementation loaded from the same checkpoint keys.
    head = (
        ReferenceHead(width=cfg.refiner_width, max_logit_residual=cfg.max_logit_residual)
        if training
        else make_head(asdict(cfg))
    )
    model = Refiner(base, head, cfg.context_scale).to(device)
    model.source_lora = {key: source[key] for key in ("config", "lora_state", "epoch")}
    return model


def load_checkpoint(checkpoint, sam2_checkpoint=None, device="cuda"):
    """Load a standalone bundled checkpoint, or a legacy checkpoint plus backbone."""
    from utils.config import Config
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**payload["config"])
    if "sam2_state" not in payload and sam2_checkpoint is None:
        from pathlib import Path
        sam2_checkpoint = Path(__file__).resolve().parents[1] / "checkpoints/physeg.pt"
    if sam2_checkpoint is not None:
        cfg.checkpoint = str(sam2_checkpoint)
    model = build_model(cfg, device=device, source_lora=payload["source_lora"], base_state=payload.get("sam2_state"))
    model.refine_head.load_state_dict(payload["head_state"], strict=True)
    return model.eval(), payload

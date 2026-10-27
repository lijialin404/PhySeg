"""Train the LoRA or Refiner stage of PhySeg."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model.lora import (
    build_model as build_lora_model,
    load_lora_state,
    lora_state_dict,
    set_pure_lora_mode,
)
from model.network import build_model as build_refiner_model
from model.sam2.postprocess import upsample_mask_logits
from utils.config import Config, LoRAConfig
from utils.data import build_loaders, build_lora_loaders, perturb_boxes
from utils.losses import lora_loss, segmentation_loss

ROOT = Path(__file__).resolve().parent


class LoRA:
    config = LoRAConfig
    loaders = staticmethod(build_lora_loaders)

    @staticmethod
    def build(cfg, saved):
        return build_lora_model(cfg)

    @staticmethod
    def train_mode(model):
        set_pure_lora_mode(model, True)

    @staticmethod
    def loss(model, batch, cfg):
        target = batch["mask"].cuda(non_blocking=True)
        boxes = perturb_boxes(
            batch["box"].cuda(non_blocking=True), cfg.image_size,
            noise=cfg.box_noise, noise_bound=cfg.box_noise_bound,
        )
        logits, quality = model.forward_logits(
            batch["image"].cuda(non_blocking=True), boxes=boxes,
            encoder_trainable=True,
        )
        loss, _, _ = lora_loss(logits, target, quality, cfg)
        return loss

    @staticmethod
    def logits(model, batch):
        low, _ = model.forward_logits(
            batch["image"].cuda(non_blocking=True),
            boxes=batch["box"].cuda(non_blocking=True), encoder_trainable=False,
        )
        return upsample_mask_logits(low, batch["mask"].shape[-2:])

    @staticmethod
    def state(model):
        return dict(format="sam2_box_qv_lora_v1", lora_state=lora_state_dict(model))

    @staticmethod
    def restore(model, saved):
        load_lora_state(model, saved["lora_state"])


class Refiner:
    config = Config
    loaders = staticmethod(build_loaders)

    @staticmethod
    def build(cfg, saved):
        source = saved["source_lora"] if saved else None
        return build_refiner_model(cfg, source_lora=source, training=True)

    @staticmethod
    def train_mode(model):
        model.train()

    @staticmethod
    def logits(model, batch):
        return model.forward_refined(
            batch["image"].cuda(non_blocking=True),
            batch["box"].cuda(non_blocking=True),
            roi_boxes=batch["roi_box"].cuda(non_blocking=True),
        ).logits

    @staticmethod
    def loss(model, batch, cfg):
        target = batch["mask"].cuda(non_blocking=True)
        logits = Refiner.logits(model, batch)
        loss, _ = segmentation_loss(logits, target, cfg)
        return loss

    @staticmethod
    def state(model):
        return dict(
            format="physeg_ablation_cos15_v1",
            head_state=model.refine_head.state_dict(), source_lora=model.source_lora,
        )

    @staticmethod
    def restore(model, saved):
        model.refine_head.load_state_dict(saved["head_state"])


def train_epoch(stage, model, loader, optimizer, parameters, cfg, epoch):
    stage.train_mode(model)
    total = torch.zeros((), dtype=torch.float64, device="cuda")
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = stage.loss(model, batch, cfg)
        window_start = ((step - 1) // cfg.accumulation_steps) * cfg.accumulation_steps
        window_size = min(cfg.accumulation_steps, len(loader) - window_start)
        (loss / window_size).backward()
        if step % cfg.accumulation_steps == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total.add_(loss.detach().double())
        if step == 1 or step % 50 == 0 or step == len(loader):
            print(f"epoch={epoch:02d} step={step}/{len(loader)} loss={float(total / step):.6f}", flush=True)
    return float(total / len(loader))


@torch.inference_mode()
def validate(stage, model, loader, cfg):
    model.eval()
    values = []
    for batch in loader:
        target = batch["mask"].cuda(non_blocking=True).bool()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = stage.logits(model, batch)
        prediction = logits[:, 0] > cfg.mask_threshold
        intersection = (prediction & target).flatten(1).sum(1)
        area = prediction.flatten(1).sum(1) + target.flatten(1).sum(1)
        values.extend(
            inter / max(union, 1)
            for inter, union in torch.stack((intersection, area - intersection), 1).tolist()
        )
    return sum(values) / len(values)


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lora", "refiner"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--sam2-checkpoint", type=Path, default=ROOT / "checkpoints/physeg.pt")
    parser.add_argument("--lora-checkpoint", type=Path, default=ROOT / "checkpoints/physeg.pt")
    args = parser.parse_args()
    stage = {"lora": LoRA, "refiner": Refiner}[args.stage]
    saved = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    cfg = stage.config(**saved["config"]) if saved else stage.config()
    output = (args.output_dir or (args.resume.parent if args.resume else ROOT / "runs" / args.stage)).resolve()
    cfg.data_root = str(args.data_root.resolve())
    cfg.output_dir = str(output)
    cfg.checkpoint = str(args.sam2_checkpoint.resolve())
    if args.stage == "refiner":
        cfg.source_lora_checkpoint = str(args.lora_checkpoint.resolve())
    output.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    train_loader, val_loader = stage.loaders(cfg, torch.device("cuda"))
    model = stage.build(cfg, saved)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.cosine_epochs, eta_min=cfg.minimum_learning_rate,
        ) if args.stage == "refiner" else None
    )
    start, best = 1, -1.0
    if saved:
        stage.restore(model, saved)
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler"])
        start, best = saved["epoch"] + 1, saved["best_val_instance_macro_iou"]

    history = output / "metrics.jsonl"
    if saved and history.exists():
        records = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
        history.write_text("".join(json.dumps(row) + "\n" for row in records if row["epoch"] < start), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    for epoch in range(start, cfg.epochs + 1):
        loss = train_epoch(stage, model, train_loader, optimizer, parameters, cfg, epoch)
        iou = validate(stage, model, val_loader, cfg)
        improved = iou > best
        best = max(best, iou)
        if scheduler is not None:
            scheduler.step()
        payload = dict(
            stage=args.stage, epoch=epoch, config=asdict(cfg),
            optimizer=optimizer.state_dict(), best_val_instance_macro_iou=best,
            **stage.state(model),
        )
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        save(output / "last.pt", payload)
        if improved:
            save(output / "best.pt", payload)
        record = dict(epoch=epoch, train_loss=loss, val_instance_macro_iou=iou, best_val_instance_macro_iou=best)
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

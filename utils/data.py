"""COCO instances with aligned, context-expanded ROI coordinates."""

import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def seed_worker(_):
    random.seed(torch.initial_seed() % 2**32)


def decode_mask(ann, record):
    segmentation = ann["segmentation"]
    height, width = record["height"], record["width"]
    if isinstance(segmentation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation["counts"], list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = dict(segmentation)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
    mask = mask_utils.decode(rle)
    return mask


def transform_box(box, size, horizontal=False, vertical=False, rotations=0):
    """Transform continuous XYXY edges alongside flips and torch.rot90."""
    x0, y0, x1, y1 = box.unbind()
    if horizontal:
        x0, x1 = size - x1, size - x0
    if vertical:
        y0, y1 = size - y1, size - y0
    for _ in range(rotations % 4):
        x0, y0, x1, y1 = y0, size - x1, y1, size - x0
    return torch.stack((x0, y0, x1, y1))


def roi_box_from_prompt(box, original_size, image_size, context_scale):
    height, width = original_size
    x0, y0, x1, y1 = box.float()
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    bw = (x1 - x0 + 1).clamp_min(2) * context_scale
    bh = (y1 - y0 + 1).clamp_min(2) * context_scale
    x0, x1 = (
        (cx - bw * 0.5).clamp(0, image_size - 1),
        (cx + bw * 0.5).clamp(0, image_size - 1),
    )
    y0, y1 = (
        (cy - bh * 0.5).clamp(0, image_size - 1),
        (cy + bh * 0.5).clamp(0, image_size - 1),
    )
    x1 = torch.maximum(x1, (x0 + 1).clamp_max(image_size - 1))
    y1 = torch.maximum(y1, (y0 + 1).clamp_max(image_size - 1))
    x0, y0 = torch.minimum(x0, x1 - 1), torch.minimum(y0, y1 - 1)
    sx, sy = (width - 1) / (image_size - 1), (height - 1) / (image_size - 1)
    left, top = max(int(torch.floor(x0 * sx)), 0), max(int(torch.floor(y0 * sy)), 0)
    right = max(min(int(torch.ceil(x1 * sx)), width - 1), min(left + 1, width - 1))
    bottom = max(min(int(torch.ceil(y1 * sy)), height - 1), min(top + 1, height - 1))
    return torch.tensor(
        (
            left / max(sx, 1e-8),
            top / max(sy, 1e-8),
            right / max(sx, 1e-8),
            bottom / max(sy, 1e-8),
        )
    )


class COCOInstanceDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        image_size=1024,
        *,
        train=False,
        cache_size=2,
        context_scale=1.25,
        box_noise=0.1,
        box_noise_bound=20,
    ):
        self.root, self.split = Path(root), split
        self.image_size, self.train = image_size, train
        self.cache_size, self.context_scale = cache_size, context_scale
        self.box_noise, self.box_noise_bound = box_noise, box_noise_bound
        data = json.loads(
            (self.root / "annotations" / f"instances_{split}.json").read_text(
                encoding="utf-8"
            )
        )
        self.images = {row["id"]: row for row in data["images"]}
        self.categories = {row["id"]: row["name"] for row in data["categories"]}
        self.annotations = data["annotations"]
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        record = self.images[ann["image_id"]]
        image = self._cache.get(record["id"])
        if image is None:
            with Image.open(
                self.root / "images" / self.split / record["file_name"]
            ) as source:
                image = torch.from_numpy(np.array(source.convert("RGB"))).permute(
                    2, 0, 1
                )
            if self.cache_size:
                self._cache[record["id"]] = image
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        elif self.cache_size:
            self._cache.move_to_end(record["id"])
        target = torch.from_numpy(decode_mask(ann, record).astype(bool))
        horizontal = self.train and random.random() < 0.5
        vertical = self.train and random.random() < 0.5
        rotations = random.randrange(4) if self.train else 0
        if horizontal:
            image, target = TF.hflip(image), TF.hflip(target)
        if vertical:
            image, target = TF.vflip(image), TF.vflip(target)
        if rotations:
            image, target = (
                torch.rot90(image, rotations, (-2, -1)),
                torch.rot90(target, rotations, (-2, -1)),
            )
        original_size = target.shape
        image = TF.resize(
            image.float().div_(255),
            [self.image_size] * 2,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        target = TF.resize(
            target[None], [self.image_size] * 2, interpolation=InterpolationMode.NEAREST
        )[0].bool()
        x, y, w, h = ann["bbox"]
        box = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
        box *= box.new_tensor(
            [self.image_size / record["width"], self.image_size / record["height"]] * 2
        )
        box = transform_box(box, self.image_size, horizontal, vertical, rotations)
        if self.train:
            box = perturb_boxes(
                box[None],
                self.image_size,
                noise=self.box_noise,
                noise_bound=self.box_noise_bound,
            )[0]
        roi = roi_box_from_prompt(
            box, original_size, self.image_size, self.context_scale
        )
        if self.train:
            image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
            image = TF.adjust_contrast(image, random.uniform(0.97, 1.03))
            image = TF.adjust_saturation(image, random.uniform(0.97, 1.03))
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return dict(
            image=image,
            mask=target,
            box=box,
            roi_box=roi,
            image_id=record["id"],
            instance_id=ann["id"],
            class_id=ann["category_id"],
            label=self.categories[ann["category_id"]],
        )


def build_loaders(cfg, device):
    datasets = [
        COCOInstanceDataset(
            cfg.data_root,
            split,
            cfg.image_size,
            train=split == "train",
            cache_size=cfg.cache_size,
            context_scale=cfg.context_scale,
            box_noise=cfg.box_noise,
            box_noise_bound=cfg.box_noise_bound,
        )
        for split in ("train", "val")
    ]
    options = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    if cfg.num_workers:
        options["prefetch_factor"] = cfg.prefetch_factor
    return tuple(
        DataLoader(
            dataset,
            shuffle=i == 0,
            generator=torch.Generator().manual_seed(cfg.seed + i + 1),
            **options,
        )
        for i, dataset in enumerate(datasets)
    )


class LoRACOCOInstanceDataset(Dataset):
    def __init__(self, root, split, image_size, *, train, cache_size):
        self.root = Path(root).resolve()
        self.split = split
        self.image_size = int(image_size)
        self.train = bool(train)
        self.cache_size = max(int(cache_size), 0)
        path = self.root / "annotations" / f"instances_{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.images = {row["id"]: row for row in data["images"]}
        self.categories = {row["id"]: row["name"] for row in data["categories"]}
        self.annotations = data["annotations"]
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.annotations)

    def _load_image(self, record):
        key = record["id"]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = self.root / "images" / self.split / record["file_name"]
        with Image.open(path) as source:
            image = TF.to_tensor(source.convert("RGB"))
        image = TF.resize(
            image,
            [self.image_size] * 2,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if self.cache_size:
            self._cache[key] = image
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return image

    def __getitem__(self, index):
        ann = self.annotations[index]
        record = self.images[ann["image_id"]]
        image = self._load_image(record)
        mask = decode_mask(ann, record)
        target = TF.resize(
            torch.from_numpy(mask.copy())[None],
            [self.image_size] * 2,
            interpolation=InterpolationMode.NEAREST,
        )[0].bool()
        x, y, w, h = ann["bbox"]
        box = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
        box *= box.new_tensor(
            [self.image_size / record["width"], self.image_size / record["height"]] * 2
        )
        if self.train:
            horizontal, vertical = random.random() < 0.5, random.random() < 0.5
            rotations = random.randrange(4)
            if horizontal:
                image, target = TF.hflip(image), TF.hflip(target)
            if vertical:
                image, target = TF.vflip(image), TF.vflip(target)
            if rotations:
                image = torch.rot90(image, rotations, dims=(-2, -1))
                target = torch.rot90(target, rotations, dims=(-2, -1))
            box = transform_box(box, self.image_size, horizontal, vertical, rotations)
            image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
            image = TF.adjust_contrast(image, random.uniform(0.97, 1.03))
            image = TF.adjust_saturation(image, random.uniform(0.97, 1.03))
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return dict(
            image=image,
            mask=target,
            box=box,
            label=self.categories[ann["category_id"]],
            class_id=ann["category_id"],
            instance_id=ann["id"],
            image_id=record["id"],
        )


def build_lora_loaders(cfg, device):
    train = LoRACOCOInstanceDataset(
        cfg.data_root, "train", cfg.image_size, train=True, cache_size=cfg.cache_size
    )
    val = LoRACOCOInstanceDataset(
        cfg.data_root, "val", cfg.image_size, train=False, cache_size=cfg.cache_size
    )
    options = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )
    if cfg.num_workers > 0:
        options["prefetch_factor"] = cfg.prefetch_factor
    train_loader = DataLoader(
        Subset(train, list(range(len(train)))),
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed + 1),
        **options,
    )
    val_loader = DataLoader(
        Subset(val, list(range(len(val)))),
        shuffle=False,
        generator=torch.Generator().manual_seed(cfg.seed + 2),
        **options,
    )
    return train_loader, val_loader


def perturb_boxes(boxes, image_size, *, noise=0.1, noise_bound=20):
    result = boxes.clone()
    if noise == 0 or noise_bound == 0:
        return result
    width = result[:, 2] - result[:, 0]
    height = result[:, 3] - result[:, 1]
    bound = result.new_tensor(float(noise_bound))
    dx = torch.minimum(width * noise, bound)
    dy = torch.minimum(height * noise, bound)
    result.add_((2 * torch.rand_like(result) - 1) * torch.stack((dx, dy, dx, dy), 1))
    image_bound = result.new_tensor((image_size - 1,) * 4)
    return result.clamp_(torch.zeros_like(image_bound), image_bound)

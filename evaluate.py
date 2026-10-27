"""Evaluate the PhySeg main model on a COCO instance validation or test split."""

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.data import COCOInstanceDataset
from model.network import load_checkpoint

METHODS = ("Full",)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--limit", type=int, help="Limit instances for a smoke check.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/physeg.pt")
    args = parser.parse_args()
    from utils.metrics import score, summarize

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(20260825)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = METHODS if args.split == "val" else ("Full",)
    model, payload = load_checkpoint(args.checkpoint)
    heads = {"Full": model.refine_head}
    epochs = {"Full": payload["epoch"]}
    del payload

    dataset = COCOInstanceDataset(args.data_root, args.split, 1024, context_scale=1.25)
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False, pin_memory=True)
    rows = {name: [] for name in names}
    started = time.time()
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=6) as pool, ExitStack() as stack:
        streams = {}
        for name in rows:
            destination = args.output_dir / name
            destination.mkdir(parents=True, exist_ok=True)
            streams[name] = stack.enter_context((destination / "instances.jsonl").open("w", encoding="utf-8"))
        for index, batch in enumerate(loader, 1):
            image = batch["image"].cuda()
            box = batch["box"].cuda()
            roi = batch["roi_box"].cuda()
            truth = batch["mask"][0].numpy().astype(bool)
            meta = dict(instance_id=int(batch["instance_id"][0]), image_id=int(batch["image_id"][0]), label=batch["label"][0])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                coarse, _, features = heads["Full"].extract_frozen_inputs(model.base_model, image, box)
                pending = {}
                for name, head in heads.items():
                    output = head(coarse, roi, (1024, 1024), decoder_features=features)
                    prediction = (output.logits[0, 0] > 0).cpu().numpy()
                    pending[name] = pool.submit(score, prediction, truth, meta)
            for name, future in pending.items():
                row = future.result()
                rows[name].append(row)
                streams[name].write(json.dumps(row) + "\n")
            if index == 1 or index % 25 == 0 or index == len(dataset):
                for stream in streams.values():
                    stream.flush()
                status = dict(state="running", split=args.split, instances=index, total=len(dataset), elapsed_seconds=time.time() - started)
                write_json(args.output_dir / "status.json", status)
                print(json.dumps(status), flush=True)
    for name, records in rows.items():
        result = dict(model=name, split=args.split, epoch=epochs.get(name), **summarize(records))
        write_json(args.output_dir / name / "summary.json", result)
    write_json(args.output_dir / "status.json", dict(state="complete", split=args.split, instances=len(dataset), limit=args.limit, elapsed_seconds=time.time() - started))


if __name__ == "__main__":
    main()

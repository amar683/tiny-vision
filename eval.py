"""Benchmark a model on a Hugging Face caption test set: BLEU-4 and CIDEr-D.

  python eval.py --run runs/flickr30k --dataset umaru97/flickr30k_train_val_test
  python eval.py --run amartomar683/tiny-vision --dataset jxie/coco_captions --beams 3

One caption is generated per image and scored against all of that image's references.
Works with one-row-per-image datasets (caption column is a list) and one-row-per-caption
datasets (rows grouped by an id column such as cocoid / img_id / filename).
Writes <run>/eval_<dataset>_<split>.json when --run is a local folder, else ./eval_*.json.
"""
import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import torch
from datasets import load_dataset

from caption import caption_images, load_run
from vlm.data import find_columns, row_captions
from vlm.metrics import bleu, cider_d

ID_COLS = ("cocoid", "image_id", "img_id", "imgid", "filename", "file_name")


def group_by_image(split, image_col, cap_cols):
    """Return [(row_index_with_image, [ref captions])] — one entry per unique image."""
    id_col = next((c for c in ID_COLS if c in split.column_names), None)
    caps = split.select_columns(cap_cols)
    if id_col is None:
        return [(i, row_captions(r, cap_cols)) for i, r in enumerate(caps)]
    groups = OrderedDict()
    for i, (key, r) in enumerate(zip(split[id_col], caps)):
        g = groups.setdefault(key, [i, []])
        g[1].extend(row_captions(r, cap_cols))
    return [tuple(g) for g in groups.values()]


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="local run/export folder or HF repo id")
    ap.add_argument("--dataset", default="umaru97/flickr30k_train_val_test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--beams", type=int, default=1, help="beam search width (1 = greedy)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    model, tok, cfg = load_run(args.run, device)
    split = load_dataset(args.dataset, split=args.split)
    image_col, cap_cols = find_columns(split)
    items = group_by_image(split, image_col, cap_cols)
    if args.limit:
        items = items[:args.limit]
    print(f"{args.dataset}[{args.split}]: {len(items)} images, "
          f"{sum(len(r) for _, r in items) / len(items):.1f} refs/image, device={device}")

    t0, hyps = time.time(), []
    for s in range(0, len(items), args.batch_size):
        imgs = [split[i][image_col] for i, _ in items[s:s + args.batch_size]]
        hyps += caption_images(model, tok, cfg, imgs, device, batch_size=args.batch_size, num_beams=args.beams)
    refs = [r for _, r in items]
    b = bleu(hyps, refs)
    cider, per = cider_d(hyps, refs)
    distinct = len(set(hyps)) / len(hyps)
    result = {
        "run": str(args.run), "dataset": args.dataset, "split": args.split, "images": len(items),
        "beams": args.beams,
        "BLEU-1": b[0], "BLEU-4": b[3], "CIDEr": cider, "distinct_captions": distinct,
        "seconds": round(time.time() - t0, 1),
        "samples": [{"pred": h, "refs": r[:2]} for h, r in list(zip(hyps, refs))[:20]],
    }
    print(f"BLEU-1 {b[0]:.3f}  BLEU-4 {b[3]:.3f}  CIDEr {cider:.3f}  "
          f"distinct captions {distinct:.0%}  ({result['seconds']}s)")

    name = f"eval_{args.dataset.replace('/', '_')}_{args.split}" + (f"_beam{args.beams}" if args.beams > 1 else "") + ".json"
    out = Path(args.out) if args.out else (Path(args.run) / name if Path(args.run).is_dir() else Path(name))
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

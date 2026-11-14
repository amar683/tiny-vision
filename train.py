"""Train the vision encoder + caption decoder from scratch.

  python train.py --dataset shapes --preset nano --steps 600          # sanity check, minutes
  python train.py --dataset jxie/flickr8k --preset tiny --steps 20000  # first real model
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vlm import Captioner, WordTokenizer, get_config
from vlm.data import (HFCaptionDataset, ShapesDataset, find_columns, iter_caption_texts,
                      shape_caption_texts)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_data(args, cfg_overrides):
    if args.dataset == "shapes":
        tok = WordTokenizer.train(shape_caption_texts(), min_freq=1)
        cfg = get_config(args.preset, vocab_size=len(tok), **cfg_overrides)
        train_ds = ShapesDataset(args.limit or 20000, tok, cfg, seed=1)
        val_ds = ShapesDataset(256, tok, cfg, seed=2)
        return tok, cfg, train_ds, val_ds

    from datasets import load_dataset

    ds = load_dataset(args.dataset)
    train_split = ds["train"]
    val_name = next((s for s in ("validation", "val", "test") if s in ds), None)
    if val_name is None:
        parts = train_split.train_test_split(test_size=0.05, seed=0)
        train_split, val_split = parts["train"], parts["test"]
    else:
        val_split = ds[val_name]
    if args.limit:
        train_split = train_split.select(range(min(args.limit, len(train_split))))
    image_col, cap_cols = find_columns(train_split)
    print(f"dataset {args.dataset}: {len(train_split)} train / {len(val_split)} val, "
          f"image={image_col!r} captions={cap_cols}")

    tok = WordTokenizer.train(iter_caption_texts(train_split, cap_cols), max_vocab=args.max_vocab)
    cfg = get_config(args.preset, vocab_size=len(tok), **cfg_overrides)
    train_ds = HFCaptionDataset(train_split, tok, cfg, True, image_col, cap_cols)
    val_ds = HFCaptionDataset(val_split, tok, cfg, False, image_col, cap_cols)
    return tok, cfg, train_ds, val_ds


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, loader, device, tok, val_ds, amp_ctx, max_batches=20, n_samples=4):
    model.eval()
    losses = []
    for b, (px, ids, tgt) in enumerate(loader):
        if b >= max_batches:
            break
        with amp_ctx():
            _, loss = model(px.to(device), ids.to(device), tgt.to(device))
        losses.append(loss.item())
    px = torch.stack([val_ds[i][0] for i in range(n_samples)]).to(device)
    with amp_ctx():
        out = model.generate(px, tok.bos, tok.eos)
    samples = [(tok.decode(o), val_ds.references(i)[0]) for i, o in enumerate(out.tolist())]
    model.train()
    return sum(losses) / len(losses), samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="jxie/flickr8k", help="'shapes' or a Hugging Face dataset id")
    ap.add_argument("--preset", default="tiny", choices=["nano", "tiny", "small"])
    ap.add_argument("--image-size", type=int, default=None, help="override preset image size")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="use only N training examples")
    ap.add_argument("--max-vocab", type=int, default=8000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast (faster, try it on MPS/CUDA)")
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    overrides = {"image_size": args.image_size} if args.image_size else {}
    tok, cfg, train_ds, val_ds = build_data(args, overrides)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(out / "tokenizer.json")
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    model = Captioner(cfg).to(device)
    print(f"device={device} params={model.num_params() / 1e6:.1f}M vocab={len(tok)} "
          f"visual_tokens/frame={model.encoder.tokens_per_frame}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    def amp_ctx():
        if args.amp and device.type in ("mps", "cuda"):
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        return torch.autocast(device_type="cpu", enabled=False)

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     persistent_workers=args.workers > 0, drop_last=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **{**loader_kw, "drop_last": False})

    def batches():
        while True:
            yield from train_loader

    it = batches()
    model.train()
    t0, seen, best = time.time(), 0, float("inf")
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        px, ids, tgt = (x.to(device, non_blocking=True) for x in next(it))
        with amp_ctx():
            _, loss = model(px, ids, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += px.size(0)

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:6d}  loss {loss.item():.3f}  lr {lr_at(step, args):.2e}  {seen / dt:.0f} img/s")
            t0, seen = time.time(), 0

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            val_loss, samples = evaluate(model, val_loader, device, tok, val_ds, amp_ctx)
            print(f"--- eval step {step + 1}: val_loss {val_loss:.3f}")
            for pred, ref in samples:
                print(f"    pred: {pred}\n    ref:  {ref}")
            if val_loss < best:
                best = val_loss
                torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step + 1,
                            "val_loss": val_loss}, out / "ckpt.pt")
                print(f"--- new best, saved {out / 'ckpt.pt'}")
            else:
                print(f"--- no improvement (best {best:.3f}); keeping previous checkpoint")


if __name__ == "__main__":
    main()

"""Compare tiny-vision with other small open captioning models on the same COCO test images.

  python bench.py --model tiny-vision --run runs/coco_dinov2 --beams 3
  python bench.py --model florence-2-base
  python bench.py --model smolvlm-256m
  python bench.py --report            # print the table from bench_results.json

Same images, same references, same BLEU/CIDEr scorer (vlm/metrics.py) for every model.
Speed: images/sec over the whole set on the GPU (batch 16), plus single-image latency on CPU.
Each model runs in its own process (one --model per call) so memory numbers don't mix.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset

from eval import group_by_image
from vlm.data import find_columns
from vlm.metrics import bleu, cider_d

RESULTS = Path("bench_results.json")
OTHERS = {
    "florence-2-base": "florence-community/Florence-2-base",
    "smolvlm-256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "smolvlm-500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
}
SMOL_PROMPT = "Describe this image in one short sentence."


def load_captioner(name, args, device):
    """Return (caption_fn(list[PIL]) -> list[str], torch module, model id)."""
    if name == "tiny-vision":
        from caption import caption_images, load_run
        model, tok, cfg = load_run(args.run, device)
        return (lambda ims: caption_images(model, tok, cfg, ims, device, batch_size=len(ims),
                                           num_beams=args.beams)), model, str(args.run)

    from transformers import AutoProcessor
    repo = OTHERS[name]
    proc = AutoProcessor.from_pretrained(repo)
    if name.startswith("florence"):
        from transformers import Florence2ForConditionalGeneration
        model = Florence2ForConditionalGeneration.from_pretrained(repo).to(device).eval()

        @torch.no_grad()
        def run(ims):
            x = proc(text=["<CAPTION>"] * len(ims), images=ims, return_tensors="pt").to(device)
            out = model.generate(**x, max_new_tokens=40, num_beams=args.beams, do_sample=False)
            return [t.strip() for t in proc.batch_decode(out, skip_special_tokens=True)]
        return run, model, repo

    from transformers import AutoModelForImageTextToText
    proc.tokenizer.padding_side = "left"  # decoder-only: pad on the left for batched generate
    model = AutoModelForImageTextToText.from_pretrained(repo).to(device).eval()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": SMOL_PROMPT}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)

    @torch.no_grad()
    def run(ims):
        x = proc(text=[prompt] * len(ims), images=[[im] for im in ims], return_tensors="pt",
                 padding=True).to(device)
        out = model.generate(**x, max_new_tokens=40, num_beams=args.beams, do_sample=False)
        texts = proc.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)
        return [t.strip() for t in texts]
    return run, model, repo


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bench(args):
    device = pick_device(args.device)
    split = load_dataset(args.dataset, split="test")
    image_col, cap_cols = find_columns(split)
    items = group_by_image(split, image_col, cap_cols)[:args.limit]
    images = [split[i][image_col].convert("RGB") for i, _ in items]
    refs = [r for _, r in items]

    caption, model, model_id = load_captioner(args.model, args, device)
    params = sum(p.numel() for p in model.parameters())
    trainable_note = None
    if args.model == "tiny-vision":
        trainable_note = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("encoder.vit."))

    caption(images[:args.batch_size])  # warm-up (kernels, caches)
    if device.type == "mps":
        torch.mps.synchronize()
    t0, hyps, peak = time.time(), [], 0
    for s in range(0, len(images), args.batch_size):
        hyps += caption(images[s:s + args.batch_size])
        if (s // args.batch_size) % 10 == 0:
            print(f"  {len(hyps)}/{len(images)} images, {len(hyps) / (time.time() - t0):.1f} img/s", flush=True)
        if device.type == "mps":
            peak = max(peak, torch.mps.driver_allocated_memory())
    if device.type == "mps":
        torch.mps.synchronize()
    gpu_secs = time.time() - t0

    # single-image CPU latency (what a laptop without a GPU, or a server CPU, would see)
    cpu_ms = None
    if args.cpu_images:
        model.to("cpu")
        if args.model == "tiny-vision":
            from caption import caption_images, load_run
            m, tok, cfg = load_run(args.run, torch.device("cpu"))
            cpu_fn = lambda im: caption_images(m, tok, cfg, [im], torch.device("cpu"), num_beams=args.beams)
        else:
            cpu_args = argparse.Namespace(**{**vars(args), "device": "cpu"})
            cpu_fn, _, _ = load_captioner(args.model, cpu_args, torch.device("cpu"))
            cpu_fn = (lambda f: (lambda im: f([im])))(cpu_fn)
        cpu_fn(images[0])
        t1 = time.time()
        for im in images[:args.cpu_images]:
            cpu_fn(im)
        cpu_ms = (time.time() - t1) / args.cpu_images * 1000

    b = bleu(hyps, refs)
    cider, _ = cider_d(hyps, refs)
    row = {
        "model": args.model + (f" (beam {args.beams})" if args.beams > 1 else ""),
        "id": model_id, "images": len(images), "params_M": round(params / 1e6, 1),
        "trainable_M": round(trainable_note / 1e6, 1) if trainable_note else None,
        "weights_MB": round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6),
        "BLEU-4": round(b[3], 3), "CIDEr": round(cider, 3),
        "distinct": round(len(set(hyps)) / len(hyps), 2),
        "gpu_img_per_s": round(len(images) / gpu_secs, 1),
        "gpu_peak_MB": round(peak / 1e6) if peak else None,
        "cpu_ms_per_img": round(cpu_ms) if cpu_ms else None,
        "device": str(device), "samples": list(zip(hyps[:8], [r[0] for r in refs[:8]])),
    }
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[row["model"]] = row
    RESULTS.write_text(json.dumps(data, indent=2))
    print(json.dumps({k: v for k, v in row.items() if k != "samples"}, indent=2))
    for h, r in row["samples"][:4]:
        print(f"  pred: {h}\n  ref:  {r}")


def report():
    data = json.loads(RESULTS.read_text())
    cols = ["model", "params_M", "weights_MB", "BLEU-4", "CIDEr", "distinct", "gpu_img_per_s",
            "gpu_peak_MB", "cpu_ms_per_img"]
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for row in sorted(data.values(), key=lambda r: r["params_M"]):
        print("| " + " | ".join(str(row.get(c)) for c in cols) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tiny-vision", *OTHERS])
    ap.add_argument("--run", default="runs/coco_dinov2", help="tiny-vision run folder or HF repo id")
    ap.add_argument("--dataset", default="jxie/coco_captions")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cpu-images", type=int, default=20, help="images for CPU latency (0 = skip)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
    else:
        bench(args)


if __name__ == "__main__":
    main()

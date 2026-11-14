"""Caption an image, or turn a video into a timestamped text timeline.

  python caption.py --run runs/flickr8k photo.jpg            # local training run
  python caption.py --run <hf-user>/tiny-vision-flickr8k photo.jpg  # Hugging Face model
  python caption.py --run runs/flickr8k clip.mp4 --every 2
"""
import argparse
import dataclasses
import json
from pathlib import Path

import torch
from PIL import Image

from vlm import Captioner, ModelConfig, WordTokenizer
from vlm.data import image_transform
from vlm.video import build_timeline, format_timeline, sample_frames

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def resolve_run(spec):
    """A local run/export folder, or a Hugging Face repo id ("user/name") to download."""
    path = Path(str(spec)).expanduser()
    if path.is_dir():
        return path
    s = str(spec)
    if s.count("/") == 1 and not s.startswith((".", "/", "~")):
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=s, allow_patterns=["*.json", "*.safetensors"]))
    raise FileNotFoundError(f"no model folder at {path} (and {s!r} is not a Hugging Face repo id)")


def load_run(run_dir, device):
    """Load a model from ckpt.pt (training run) or model.safetensors (exported / Hugging Face)."""
    run = resolve_run(run_dir)
    if (run / "model.safetensors").exists():
        from safetensors.torch import load_file
        state = load_file(run / "model.safetensors")
        cfg_dict = json.loads((run / "config.json").read_text())
    else:
        ckpt = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=True)
        state, cfg_dict = ckpt["model"], ckpt["config"]
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in fields})
    model = Captioner(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "head.weight"]  # tied to tok.weight; may be omitted
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model.to(device).eval(), WordTokenizer.load(run / "tokenizer.json"), cfg


@torch.no_grad()
def caption_images(model, tok, cfg, images, device, batch_size=16):
    tf = image_transform(cfg.image_size, train=False)
    out = []
    for i in range(0, len(images), batch_size):
        px = torch.stack([tf(im.convert("RGB")) for im in images[i:i + batch_size]]).to(device)
        out += [tok.decode(ids) for ids in model.generate(px, tok.bos, tok.eos).tolist()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--run", default="runs/latest")
    ap.add_argument("--every", type=float, default=2.0, help="video: seconds between sampled frames")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--merge", type=float, default=1.0,
                    help="video: word-overlap needed to merge neighbouring captions (1.0 = identical only)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, tok, cfg = load_run(args.run, device)
    path = Path(args.path)

    if path.suffix.lower() in VIDEO_EXT:
        frames, duration = sample_frames(path, args.every, args.max_frames)
        caps = caption_images(model, tok, cfg, [im for _, im in frames], device)
        segments = build_timeline([(t, c) for (t, _), c in zip(frames, caps)], duration, args.merge)
        if args.json:
            print(json.dumps([{"start": s, "end": e, "caption": c} for s, e, c in segments], indent=2))
        else:
            print(format_timeline(segments))
    else:
        cap = caption_images(model, tok, cfg, [Image.open(path)], device)[0]
        print(json.dumps({"caption": cap}) if args.json else cap)


if __name__ == "__main__":
    main()

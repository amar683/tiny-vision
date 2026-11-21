"""Package a training run as a Hugging Face model folder.

  python export_hf.py --run runs/coco_dinov2 --out hf_export/tiny-vision

Writes model.safetensors, config.json, tokenizer.json and README.md (the model card,
from MODEL_CARD.md). Upload the folder with:  hf upload <user>/tiny-vision hf_export/tiny-vision
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/coco_dinov2")
    ap.add_argument("--out", default="hf_export/tiny-vision")
    ap.add_argument("--card", default="MODEL_CARD.md")
    args = ap.parse_args()

    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=True)
    state = dict(ckpt["model"])

    # head.weight is tied to tok.weight; safetensors can't store shared tensors, and the
    # loader re-ties them, so store it once.
    if "head.weight" in state:
        assert torch.equal(state["head.weight"], state["tok.weight"]), "head and embedding not tied"
        del state["head.weight"]
    save_file({k: v.contiguous() for k, v in state.items()}, out / "model.safetensors",
              metadata={"format": "pt"})

    config = dict(ckpt["config"])
    config.update({"model_type": "tiny-vision-captioner", "tie_word_embeddings": True,
                   "train_step": ckpt.get("step"), "val_loss": ckpt.get("val_loss")})
    (out / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copy(run / "tokenizer.json", out / "tokenizer.json")
    shutil.copy(args.card, out / "README.md")

    n = sum(v.numel() for v in state.values())
    print(f"exported {n / 1e6:.1f}M params to {out}/ (step {ckpt.get('step')}, val_loss {ckpt.get('val_loss'):.3f})")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:20s} {f.stat().st_size / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()

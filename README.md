# tiny-vision

A small vision model trained **from scratch**, built to be universal. It turns images
and video into:

- **text** (captions and timestamped video timelines) that any language model can read, and
- **visual tokens** that open LLMs can use through a small per-model adapter (planned).

```
image / video frames → VisionEncoder (ViT) → Adapter → tiny caption decoder → text
                                            └→ planned: adapters into Qwen / Llama embeddings
```

> Status: **v0.1, experimental.** The first model is trained on Flickr8k (6,000 photos).
> It gets scene types roughly right and subjects often wrong.

## Setup

```bash
git clone https://github.com/amar683/tiny-vision && cd tiny-vision
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

Runs on Apple Silicon (MPS), NVIDIA (CUDA) or CPU.

## Use the pretrained model

```bash
.venv/bin/python caption.py --run amartomar683/tiny-vision-flickr8k photo.jpg
.venv/bin/python caption.py --run amartomar683/tiny-vision-flickr8k clip.mp4 --every 2 --json
```

`--run` takes a Hugging Face repo id (downloaded on first use) or a local folder.

## Use it from an MCP client

`mcp_server.py` exposes two tools over stdio:

| Tool | Does |
|---|---|
| `caption_image(path)` | One-sentence caption of a local image |
| `describe_video(path, every_seconds=2, max_frames=120)` | Timestamped text timeline of a local video |

Add it to your MCP client's config (use absolute paths):

```json
{
  "mcpServers": {
    "tiny-vision": {
      "command": "/path/to/tiny-vision/.venv/bin/python",
      "args": ["/path/to/tiny-vision/mcp_server.py"]
    }
  }
}
```

Model source: the `TINY_VISION_MODEL` env var (local folder or HF repo id), else
`runs/flickr8k` if you trained locally, else the published Hugging Face model.

## Train your own

```bash
.venv/bin/python smoke_test.py                       # ~1 min end-to-end check
.venv/bin/python train.py --dataset jxie/flickr8k --preset tiny --steps 6000 --amp --out runs/flickr8k
.venv/bin/python train.py --dataset umaru97/flickr30k_train_val_test --preset tiny --steps 15000 \
    --warmup 1000 --eval-every 1000 --amp --out runs/flickr30k
```

`--dataset` takes any Hugging Face image-caption dataset; image and caption columns are
auto-detected. The best checkpoint (by validation loss) is kept.

| preset | params | image | use |
|---|---|---|---|
| nano  | ~1.3M | 64px  | smoke tests |
| tiny  | ~25M  | 224px | first real model (~40 min on an M5 for Flickr8k) |
| small | ~56M  | 224px | larger datasets / cloud GPU |

## Publish a model to Hugging Face

```bash
.venv/bin/python export_hf.py --run runs/flickr8k --out hf_export/tiny-vision-flickr8k
hf upload amartomar683/tiny-vision-flickr8k hf_export/tiny-vision-flickr8k
```

The export writes `model.safetensors`, `config.json`, `tokenizer.json` and the model card
(`MODEL_CARD.md` → `README.md`).

## Layout

- `vlm/encoder.py`: ViT with fixed 2D sin-cos positions, token pooling, per-frame time embedding
- `vlm/captioner.py`: adapter + prefix-LM caption decoder
- `vlm/video.py`: frame sampling and text timeline
- `vlm/data.py`: Hugging Face loader and a synthetic shapes dataset
- `vlm/tokenizer.py`: word-level tokenizer built from captions
- `caption.py`, `train.py`, `export_hf.py`, `mcp_server.py`, `smoke_test.py`

## Roadmap

- Train on Flickr30k / larger data for better captions
- Change-based video frame sampling and real timestamps for variable-frame-rate video
- Embedding adapters for open LLMs (Qwen, Llama), exportable as llama.cpp `mmproj.gguf`
- Clip-level training for motion understanding

## License

Code: Apache-2.0 (see `LICENSE`). Model weights trained on Flickr8k: see the model card
for training-data terms.

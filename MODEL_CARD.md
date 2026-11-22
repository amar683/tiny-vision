---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-to-text
datasets:
  - jxie/coco_captions
tags:
  - image-captioning
  - video
  - dinov2
  - tiny
  - edge
  - mcp
---

# tiny-vision v1.0

**Let your AI assistant watch videos.** Many assistants, Claude included, can look at
images but can't open a video file. tiny-vision turns a local video into a scene-by-scene
description the assistant can read, through MCP (Claude Desktop, Claude Code, Cursor,
VS Code, LM Studio and other MCP apps) or as plain text for any LLM. It runs locally, so
nothing is uploaded.

It's a **36M-parameter** image and video captioner, fast on a laptop CPU. It uses Meta's
DINOv2-small as a frozen image encoder (its weights are unchanged, so this is not a
fine-tune of DINOv2); the adapter and caption decoder on top (14M parameters) were
trained from scratch on COCO.

![tiny-vision turning a video into a scene-by-scene breakdown](https://raw.githubusercontent.com/amar683/tiny-vision/main/assets/demo.gif)

**Try it in your browser, no install:**
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amar683/tiny-vision/blob/main/notebooks/demo.ipynb)

- **89% of Florence-2-base's COCO caption score at 1/6 the size**
- **8× faster on GPU, 75× faster on CPU** (54 ms per image on an Apple M5 CPU)
- 1.2 GB peak GPU memory, 144 MB weights

## Benchmarks

**COCO Karpathy test (5,000 images)**

| Decoding | BLEU-4 | CIDEr |
|---|---|---|
| greedy | 0.342 | 1.145 |
| beam 3 (default) | 0.351 | 1.163 |

**Against other small open models.** Same 1,000 COCO test images, same scorer, Apple M5
(MPS GPU batch 16; CPU = single image). Reproduce with `bench.py`.

| Model | Params | CIDEr | BLEU-4 | GPU img/s | GPU peak memory | CPU ms/img |
|---|---|---|---|---|---|---|
| **tiny-vision v1.0** (greedy) | **36M** | 1.196 | 0.339 | **52** | **1.2 GB** | **54** |
| **tiny-vision v1.0** (beam 3) | 36M | 1.213 | 0.348 | 24 | 1.2 GB | 101 |
| Florence-2-base | 231M | 1.348 | 0.364 | 6.7 | 7.0 GB | 4,063 |
| Florence-2-base (beam 3) | 231M | 1.364 | 0.368 | 5.3 | 7.0 GB | 4,228 |
| SmolVLM-256M-Instruct ¹ | 256M | 0.126 | 0.051 | 0.5 | 23 GB | 55,800 |

¹ SmolVLM is a chat model. Even when asked for one short sentence, it writes long
descriptions, which caption metrics punish, so its quality score isn't a fair comparison.
Its speed and memory numbers are. Florence-2 is a multi-task model (detection, OCR,
grounding and more) and does more than captioning.

**Progress across versions** (greedy decoding)

| Version | Encoder | Train data | COCO CIDEr | Flickr30k CIDEr |
|---|---|---|---|---|
| v0.1 | from scratch | Flickr8k (6k images) | 0.061 | 0.131 |
| v0.2 (unreleased) | from scratch | Flickr30k (29k) | 0.121 | 0.204 |
| **v1.0** | frozen DINOv2-small | COCO (113k) | **1.145** | 0.423 |

Metrics are corpus BLEU-4 and CIDEr-D, computed with a pure-Python reimplementation of the
coco-caption scorers. It matches the reference code exactly on the same tokens, but
tokenization is simple lowercase plus punctuation stripping rather than the PTB tokenizer,
so scores are close to, but not identical with, numbers in papers.

## Architecture

```
image / video frames → DINOv2-small (frozen) → 2×2 pool → adapter → 6-layer decoder → text
                        22M, Meta, Apache-2.0    64 tokens   MLP      prefix-LM, 384-dim
```

| Part | Details | Trained here |
|---|---|---|
| Vision encoder | DINOv2-small ViT-S/14, 224px → 16×16 patches, frozen | no (pretrained by Meta) |
| Pooling + time | 2×2 average pool → 64 tokens per frame, sin-cos time embedding per video frame | — |
| Adapter | 2-layer MLP (384 → 384) | yes |
| Caption decoder | 6-layer prefix-LM transformer, 384-dim, RMSNorm + SwiGLU, 8,000-word vocabulary, tied embeddings | yes |

14M of the 36M parameters are trained, all on COCO. The encoder weights are bundled in
`model.safetensors`, so the model runs fully offline after download.

## Training

- Data: COCO Karpathy train split (113k images, 566,747 captions)
- 40,000 steps, batch 64 (~4.5 caption-epochs), AdamW (lr 3e-4, warmup 1,500, cosine to 3e-5),
  weight decay 0.05, bfloat16 autocast, random-resized-crop augmentation
- One Apple M5 MacBook Pro GPU, ~3 hours. Final validation loss 2.117

## Video: scene-by-scene summaries

`summarize_video` (MCP) and `caption.py clip.mp4 --summary` (CLI) do three things:
- split the video into shots (hard cuts plus slow camera drift)
- caption the start, middle and end of each shot
- group consecutive shots that show the same thing into scenes

The output is compact text that an LLM turns into a real summary:

```
video: clip.mp4 · 00:09 long · 3 scenes (3 shots) · avg scene 3s
recurring: dogs (2 scenes)

scene 1 [00:00–00:03] (3s)
  seen: a group of dogs running in a field. | three dogs running in a field with a bench in the background.
  keywords: dogs, running, field, bench
...
```

Over MCP, the connected assistant writes the summary. From the CLI, `--summarize` uses
a local Qwen2.5-1.5B-Instruct. Frames are captioned independently and there's no audio
yet, so it describes what's seen, not what's said or how things move.

## Example outputs (COCO test, beam 3)

| tiny-vision | Human reference |
|---|---|
| a man riding a motorcycle down a dirt road. | A man with a red helmet on a small moped on a dirt road. |
| a man riding a bike next to a train on a track. | A man on a bicycle riding next to a train |
| a group of people riding bikes down a street. | people on bicycles ride down a busy street |
| a bathroom with a shower, toilet, and sink. | A bathroom with an enclosed shower next to a sink and a toilet. |
| a woman sitting at a table with a cup of coffee. | A young girl inhales with the intent of blowing out a candle. *(wrong)* |

## Usage

Quickest, with [uv](https://docs.astral.sh/uv/) (no clone):

```bash
uvx --from git+https://github.com/amar683/tiny-vision tiny-vision photo.jpg
uvx --from git+https://github.com/amar683/tiny-vision tiny-vision clip.mp4 --summary
```

As an MCP server, add this to your app's MCP config:

```json
{"mcpServers": {"tiny-vision": {"command": "uvx",
  "args": ["--from", "git+https://github.com/amar683/tiny-vision", "tiny-vision-mcp"]}}}
```

From a clone:

```bash
git clone https://github.com/amar683/tiny-vision && cd tiny-vision
uv venv && uv pip install -r requirements.txt
python caption.py --run amartomar683/tiny-vision photo.jpg
python caption.py --run amartomar683/tiny-vision clip.mp4 --scenes
python eval.py --run amartomar683/tiny-vision --dataset jxie/coco_captions --beams 3
```

As an MCP server (`mcp_server.py`), it provides the `summarize_video`, `caption_image` and
`describe_video` tools. Add it to any app that runs local MCP servers, and that app's LLM
can work with local images and videos. Then ask things like *"what happens in
~/Downloads/talk.mp4?"*.

## Limitations

- **COCO-style captions only**: one short sentence about everyday scenes. No OCR, no precise
  counting, no names of people, places or brands, and no question answering.
- Domain shift: it scores lower on Flickr30k (CIDEr 0.42) than on COCO. Unusual content
  (screenshots, documents, art, medical images) gives confident but wrong captions.
- English only, word-level vocabulary (unknown words become `<unk>`).
- Video frames are captioned independently. It doesn't understand actions across frames.
- Like any model trained on COCO, it inherits COCO's biases (for example, guessing gender
  from appearance).

## License and data

- Code and the trained adapter/decoder weights: Apache-2.0.
- Encoder: DINOv2-small by Meta AI, Apache-2.0.
- Training data: COCO captions (annotations CC BY 4.0). The images come from Flickr under
  their own licenses; see the COCO terms of use.

Previous version: [amartomar683/tiny-vision-flickr8k](https://huggingface.co/amartomar683/tiny-vision-flickr8k) (v0.1, from scratch).

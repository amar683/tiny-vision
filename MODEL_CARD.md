---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-to-text
datasets:
  - jxie/flickr8k
tags:
  - image-captioning
  - video
  - vision-transformer
  - from-scratch
  - tiny
  - mcp
---

# tiny-vision (Flickr8k, v0.1)

A ~24M-parameter image captioning model trained **from scratch** (no pretrained weights)
on Flickr8k. It is the first checkpoint of *tiny-vision*, a small vision model meant to
plug into any language model: as text (captions and video timelines) today, and as
embeddings through small per-LLM adapters later.

> **Experimental.** Trained on only 6,000 photos. It describes the scene type reasonably
> (snow, grass, water, outdoors) but often gets the subject wrong. Treat outputs as rough hints.

## Architecture

| Part | Details |
|---|---|
| Vision encoder | ViT, 224px input, 16px patches, 384-dim, 6 layers, 6 heads, RMSNorm + SwiGLU, fixed 2D sin-cos positions, 2×2 token pooling → 49 tokens per frame |
| Adapter | 2-layer MLP (384 → 384) |
| Caption decoder | 6-layer prefix-LM transformer, 384-dim, word-level vocab (4,461 words), tied embeddings |
| Video | Frames sampled over time, captioned individually, merged into a timestamped timeline |

## Training

- Data: Flickr8k train split (6,000 images × 5 captions)
- 6,000 steps, batch 64, AdamW (lr 3e-4, cosine), bfloat16 autocast, Apple M5 GPU (~42 min)
- Best checkpoint: step 3,000, validation loss **2.80** (later steps overfit)

## Example outputs (unseen test images)

| Model | Human reference |
|---|---|
| a black dog is running through a field. | A group of dogs runs beside a pond through a field. |
| a brown dog is playing with a red and white toy. | A dog with its mouth opened. |
| a man in a blue shirt and jeans is standing in the snow. | The dogs are in the snow in front of a fence. |
| a dog jumps over a hurdle. | Children participate in a sport on a green field. |

## Usage

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/tiny-vision && cd tiny-vision
uv venv && uv pip install -r requirements.txt
python caption.py --run amartomar683/tiny-vision-flickr8k photo.jpg
python caption.py --run amartomar683/tiny-vision-flickr8k clip.mp4 --every 2
```

It also ships as an MCP server (`mcp_server.py`) with `caption_image` and `describe_video`
tools, so MCP clients can use it for images and video.

## Limitations

- Small training set: frequent wrong subjects, and it falls back on common phrases
  ("a man in a blue shirt", "a dog jumps over a hurdle").
- Content outside Flickr8k (indoor scenes, food, text, screenshots, animals other than dogs)
  gives confident but wrong captions.
- English only, word-level vocabulary: unknown words become `<unk>`.
- Video: frames are captioned independently, so there's no motion understanding yet.

## License and training data

Code and weights are released under Apache-2.0. The model was trained on Flickr8k, whose
images come from Flickr and were collected for research use. Check the dataset's terms
before any commercial use.

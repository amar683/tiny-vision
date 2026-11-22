---
title: tiny-vision
emoji: 🎞️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - amartomar683/tiny-vision
short_description: Turn a video into scene-by-scene text any LLM can read
---

Demo of [tiny-vision](https://github.com/amar683/tiny-vision): a 36M-parameter local
image/video captioner (frozen DINOv2-small encoder + caption decoder trained on COCO) that
turns videos into a scene-by-scene text breakdown, with an MCP server so AI assistants can
"watch" local videos. Model: [amartomar683/tiny-vision](https://huggingface.co/amartomar683/tiny-vision).

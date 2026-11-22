"""tiny-vision demo: caption a photo, or turn a video of any length into a scene-by-scene breakdown."""
import time

import gradio as gr
import torch

from caption import caption_images, load_run
from vlm.scenes import describe_scenes, video_length
from vlm.video import _ts

MODEL = "amartomar683/tiny-vision"
BEAMS = 3
DEVICE = torch.device("cpu")

model, tok, cfg = load_run(MODEL, DEVICE)


def _caption(images):
    return caption_images(model, tok, cfg, images, DEVICE, num_beams=BEAMS)


def caption_photo(image):
    if image is None:
        raise gr.Error("Upload a photo first.")
    t0 = time.time()
    cap = _caption([image])[0]
    return cap, f"{(time.time() - t0) * 1000:.0f} ms on CPU"


def summarize_video(path, progress=gr.Progress()):
    if not path:
        raise gr.Error("Upload a video first.")
    sec = video_length(path)
    t0 = time.time()

    def show(t, total):
        progress(min(t / total, 1.0) if total else None, desc=f"reading video {_ts(t)} / {_ts(total)}")

    text, _ = describe_scenes(path, _caption, progress=show)
    return text, f"{time.time() - t0:.1f} s for a {sec:.0f} s video on CPU"


INTRO = """
# tiny-vision
**Let your AI assistant watch videos.** A 36M-parameter captioner (frozen DINOv2-small
encoder + a small caption decoder trained on COCO) that turns a video into a
scene-by-scene text breakdown any LLM can read. Paste the output into ChatGPT, Claude or a
local model and ask *"what happens in this video?"*

Runs on a laptop CPU, and locally as an MCP server so Claude Desktop, LM Studio, Cursor and
others can read your videos directly:
[GitHub](https://github.com/amar683/tiny-vision) ·
[Model](https://huggingface.co/amartomar683/tiny-vision)

*Captions are short and approximate (COCO style: no text reading, names or counting), and
it's visual only: no audio yet.*
"""

with gr.Blocks(title="tiny-vision") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Video → scene breakdown"):
        with gr.Row():
            with gr.Column():
                vid = gr.Video(label="Video (any length; long videos take a few minutes on CPU)", sources=["upload"])
                vbtn = gr.Button("Describe scenes", variant="primary")
            with gr.Column():
                vout = gr.Textbox(label="Scene breakdown (paste into any LLM)", lines=22,
                                  buttons=["copy"])
                vtime = gr.Markdown()
        vbtn.click(summarize_video, vid, [vout, vtime])
    with gr.Tab("Photo → caption"):
        with gr.Row():
            with gr.Column():
                img = gr.Image(label="Photo", type="pil", sources=["upload", "clipboard"])
                ibtn = gr.Button("Caption", variant="primary")
            with gr.Column():
                iout = gr.Textbox(label="Caption")
                itime = gr.Markdown()
        ibtn.click(caption_photo, img, [iout, itime])

if __name__ == "__main__":
    demo.queue(max_size=16).launch()

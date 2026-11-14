"""MCP server: lets any MCP client use the local tiny-vision model.

Tools:
  caption_image(path)                      -> one-sentence caption
  describe_video(path, every_seconds, ...) -> timestamped text timeline

Runs over stdio. Model source, in order:
  1. TINY_VISION_MODEL env var: a local folder or a Hugging Face repo id ("user/name")
  2. runs/flickr8k, if you trained locally
  3. the published Hugging Face model (HF_REPO below), downloaded on first use
Never print to stdout here: stdout is the MCP protocol channel.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from caption import VIDEO_EXT, caption_images, load_run  # noqa: E402
from vlm.video import build_timeline, format_timeline, sample_frames  # noqa: E402

HF_REPO = "amartomar683/tiny-vision-flickr8k"  # set after uploading the model
LOCAL_RUN = ROOT / "runs" / "flickr8k"
MODEL_SOURCE = os.environ.get("TINY_VISION_MODEL") or os.environ.get("TINY_VISION_RUN") or (
    str(LOCAL_RUN) if LOCAL_RUN.is_dir() else HF_REPO)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

server = MCPServer(
    name="tiny-vision",
    instructions=(
        "Local, from-scratch tiny vision model (~25M params) trained on Flickr8k: photos of "
        "people and dogs outdoors. Captions are short and often wrong on other content; treat "
        "them as rough hints, not ground truth. Use describe_video for videos, which many "
        "language models cannot read directly."),
)

_cache = {}


def _model():
    if "m" not in _cache:
        _cache["m"] = load_run(MODEL_SOURCE, DEVICE)
    return _cache["m"]


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.is_file():
        raise ToolError(f"file not found: {p}")
    return p


@server.tool()
def caption_image(path: str) -> str:
    """Caption a local image file (jpg, png, ...) with the tiny-vision model.

    Args:
        path: Absolute path to the image on this computer (~ is allowed).
    """
    p = _resolve(path)
    try:
        img = Image.open(p)
        img.load()
    except UnidentifiedImageError:
        raise ToolError(f"not a readable image: {p}")
    model, tok, cfg = _model()
    return caption_images(model, tok, cfg, [img], DEVICE)[0]


@server.tool()
def describe_video(path: str, every_seconds: float = 2.0, max_frames: int = 120,
                   merge_threshold: float = 1.0) -> str:
    """Turn a local video into a timestamped text timeline of what is on screen.

    Samples one frame every `every_seconds`, captions each, and merges neighbouring
    identical captions into segments like "[00:04–00:10] a dog runs on the beach."

    Args:
        path: Absolute path to the video (mp4, mov, mkv, avi, webm, m4v).
        every_seconds: Seconds between sampled frames (smaller = finer, slower).
        max_frames: Upper bound on frames sampled, to keep long videos fast.
        merge_threshold: Word overlap (0-1) needed to merge neighbouring captions; 1.0 = identical only.
    """
    p = _resolve(path)
    if p.suffix.lower() not in VIDEO_EXT:
        raise ToolError(f"unsupported video type {p.suffix!r}; expected one of {sorted(VIDEO_EXT)}")
    if every_seconds <= 0:
        raise ToolError("every_seconds must be > 0")
    frames, duration = sample_frames(p, every_seconds, max_frames)
    if not frames:
        raise ToolError(f"no frames could be read from {p}")
    model, tok, cfg = _model()
    caps = caption_images(model, tok, cfg, [im for _, im in frames], DEVICE)
    segments = build_timeline([(t, c) for (t, _), c in zip(frames, caps)], duration, merge_threshold)
    note = ""
    if max_frames and len(frames) >= max_frames:
        note = (f"\n(note: hit max_frames={max_frames}; only the first {duration:.0f}s were described. "
                f"Raise max_frames or every_seconds to cover more.)")
    return f"video: {p.name}, {len(frames)} frames sampled\n" + format_timeline(segments) + note


if __name__ == "__main__":
    server.run("stdio")

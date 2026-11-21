"""MCP server: lets any MCP client use the local tiny-vision model.

Tools:
  caption_image(path)                      -> one-sentence caption
  summarize_video(path, ...)               -> scene-by-scene breakdown for the LLM to summarize
  describe_video(path, mode, ...)          -> timestamped text timeline (one caption per sample)

Runs over stdio. Model source, in order:
  1. TINY_VISION_MODEL env var: a local folder or a Hugging Face repo id ("user/name")
  2. runs/coco_dinov2, if you trained v1.0 locally
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
from vlm.scenes import describe_scenes  # noqa: E402
from vlm.video import build_timeline, format_timeline, sample_frames, sample_scenes  # noqa: E402

HF_REPO = "amartomar683/tiny-vision"  # v1.0 (v0.1: amartomar683/tiny-vision-flickr8k)
LOCAL_RUN = ROOT / "runs" / "coco_dinov2"
BEAMS = int(os.environ.get("TINY_VISION_BEAMS", "3"))  # beam search width; 1 = greedy, fastest
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
        "Local tiny vision model (36M params: frozen DINOv2-small encoder + caption decoder "
        "trained on COCO). Gives one-sentence, COCO-style captions of everyday scenes; it "
        "does not read text (OCR), count precisely, or know names. Treat captions as a quick "
        "summary, not ground truth. For videos (which many language models cannot read "
        "directly), use summarize_video to understand a video scene by scene, or "
        "describe_video for a simple timeline."),
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
    return caption_images(model, tok, cfg, [img], DEVICE, num_beams=BEAMS)[0]


@server.tool()
def describe_video(path: str, mode: str = "scenes", scene_threshold: float = 0.3,
                   every_seconds: float = 2.0, max_frames: int = 120,
                   merge_threshold: float = 1.0) -> str:
    """Turn a local video into a timestamped text timeline of what is on screen.

    Picks frames, captions each, and merges neighbouring identical captions into
    segments like "[00:04–00:10] a dog runs on the beach." Timestamps are the video's
    real timestamps (correct for variable-frame-rate phone video).

    Args:
        path: Absolute path to the video (mp4, mov, mkv, avi, webm, m4v).
        mode: "scenes" (default) captions a frame whenever the picture changes (cuts, new
            shots), plus one every 10 s on long static shots; segment boundaries land on
            the actual cuts and static stretches cost one caption. "interval" captions one
            frame every `every_seconds`.
        scene_threshold: scenes mode only. Change score (0-1) that counts as a new scene;
            lower = more sensitive (0.15 catches camera moves, 0.5 only hard cuts).
        every_seconds: interval mode only. Seconds between sampled frames.
        max_frames: Upper bound on frames captioned, to keep long videos fast.
        merge_threshold: Word overlap (0-1) needed to merge neighbouring captions; 1.0 = identical only.
    """
    p = _resolve(path)
    if p.suffix.lower() not in VIDEO_EXT:
        raise ToolError(f"unsupported video type {p.suffix!r}; expected one of {sorted(VIDEO_EXT)}")
    if mode == "scenes":
        if not 0 < scene_threshold < 1:
            raise ToolError("scene_threshold must be between 0 and 1")
        frames, duration = sample_scenes(p, threshold=scene_threshold, max_frames=max_frames)
    elif mode == "interval":
        if every_seconds <= 0:
            raise ToolError("every_seconds must be > 0")
        frames, duration = sample_frames(p, every_seconds, max_frames)
    else:
        raise ToolError(f"mode must be 'scenes' or 'interval', got {mode!r}")
    if not frames:
        raise ToolError(f"no frames could be read from {p}")
    model, tok, cfg = _model()
    caps = caption_images(model, tok, cfg, [im for _, im in frames], DEVICE, num_beams=BEAMS)
    segments = build_timeline([(t, c) for (t, _), c in zip(frames, caps)], duration, merge_threshold)
    note = ""
    if max_frames and len(frames) >= max_frames:
        knob = "scene_threshold" if mode == "scenes" else "every_seconds"
        note = (f"\n(note: hit max_frames={max_frames}; only the first {duration:.0f}s were described. "
                f"Raise max_frames or {knob} to cover more.)")
    return f"video: {p.name}, {len(frames)} frames sampled ({mode})\n" + format_timeline(segments) + note


@server.tool()
def summarize_video(path: str, scene_threshold: float = 0.3, frames_per_shot: int = 3,
                    max_shots: int = 300) -> str:
    """Understand a local video scene by scene, so you can summarize or answer questions about it.

    Splits the video into shots (hard cuts and slow camera drift), captions several frames
    per shot (start, middle, end), and groups consecutive shots that show the same thing
    into scenes. Returns the video length, the subjects that recur across scenes, and for
    each scene its time range, what is seen and its keywords. Use it to write a summary in
    your own words, scene by scene. Visual only: there is no audio or speech.

    Args:
        path: Absolute path to the video (mp4, mov, mkv, avi, webm, m4v).
        scene_threshold: Change score (0-1) that counts as a cut; lower = more scenes.
        frames_per_shot: Frames captioned per shot (1-5); more = richer but slower.
        max_shots: Upper bound on shots, to keep very long or fast-cut videos fast.
    """
    p = _resolve(path)
    if p.suffix.lower() not in VIDEO_EXT:
        raise ToolError(f"unsupported video type {p.suffix!r}; expected one of {sorted(VIDEO_EXT)}")
    if not 0 < scene_threshold < 1:
        raise ToolError("scene_threshold must be between 0 and 1")
    if not 1 <= frames_per_shot <= 5:
        raise ToolError("frames_per_shot must be 1-5")
    model, tok, cfg = _model()
    text, data = describe_scenes(
        p, lambda ims: caption_images(model, tok, cfg, ims, DEVICE, num_beams=BEAMS),
        threshold=scene_threshold, frames_per_shot=frames_per_shot, max_shots=max_shots)
    if not data["scenes"]:
        raise ToolError(f"no frames could be read from {p}")
    if data["overview"]["shots"] >= max_shots:
        text += f"\n(note: hit max_shots={max_shots}; the rest of the video was not described.)"
    return text


if __name__ == "__main__":
    server.run("stdio")

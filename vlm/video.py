"""Video -> sampled frames -> per-frame captions -> merged text timeline any LLM can read."""
from PIL import Image


def sample_frames(path, every_sec=2.0, max_frames=None):
    """Return ([(time_sec, PIL.Image), ...], duration_sec). Needs opencv-python."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * every_sec))
    frames, idx = [], 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                frames.append((idx / fps, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
                if max_frames and len(frames) >= max_frames:
                    idx += 1
                    break
        idx += 1
    cap.release()
    return frames, idx / fps


def _similar(a, b, threshold):
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return a == b
    return len(wa & wb) / len(wa | wb) >= threshold


def build_timeline(timed_captions, duration, merge_threshold=1.0):
    """Merge consecutive captions into segments [(start, end, caption)].

    merge_threshold is word-overlap (Jaccard). 1.0 = merge only identical captions (safe
    default: "green square" vs "green triangle" differ by one word but are different
    scenes). Lower it (e.g. 0.6) for a model whose wording drifts on a static scene.
    """
    segments = []
    for t, cap in timed_captions:
        if segments and _similar(segments[-1][2], cap, merge_threshold):
            continue
        if segments:
            segments[-1][1] = t
        segments.append([t, duration, cap])
    return [tuple(s) for s in segments]


def _ts(sec):
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def format_timeline(segments):
    return "\n".join(f"[{_ts(s)}–{_ts(e)}] {c}" for s, e, c in segments)

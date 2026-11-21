"""Video -> sampled frames -> per-frame captions -> merged text timeline any LLM can read."""
from PIL import Image


def _open(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, (fps if fps and fps > 0 else 30.0)


def _pos_sec(cap, idx, fps):
    """Timestamp of the frame just grabbed. Uses the container's real timestamps (correct for
    variable-frame-rate video); falls back to idx / fps when the backend reports 0."""
    import cv2

    ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    return ms / 1000.0 if ms > 0 or idx == 0 else idx / fps


def _to_pil(frame):
    import cv2

    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def sample_frames(path, every_sec=2.0, max_frames=None):
    """Fixed-interval sampling: one frame every `every_sec` seconds of real video time.

    Returns ([(time_sec, PIL.Image), ...], duration_sec). Needs opencv-python.
    """
    if every_sec <= 0:
        raise ValueError("every_sec must be > 0")
    cap, fps = _open(path)
    frames, idx, t, next_t = [], 0, 0.0, 0.0
    while cap.grab():
        t = _pos_sec(cap, idx, fps)
        idx += 1
        if t + 0.5 / fps >= next_t:
            ok, frame = cap.retrieve()
            if ok:
                frames.append((t, _to_pil(frame)))
                while next_t <= t + 0.5 / fps:
                    next_t += every_sec
                if max_frames and len(frames) >= max_frames:
                    break
    cap.release()
    return frames, (t + 1.0 / fps if idx else 0.0)


def _signature(frame):
    """Cheap per-frame fingerprint: 64x36 grayscale thumbnail + normalised HSV histogram."""
    import cv2

    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return gray, hist


def scene_change_score(a, b):
    """0 (same picture) .. 1 (completely different) between two signatures from _signature.

    Mean of pixel change (mean-abs-diff of grayscale thumbnails, x2 and capped so a
    50%-brightness swing already counts as maximal) and colour change (Bhattacharyya
    distance of HSV histograms). Pixels catch layout/motion; colours catch cuts between
    shots with similar brightness.
    """
    import cv2
    import numpy as np

    pix = min(1.0, 2.0 * float(np.abs(a[0] - b[0]).mean()))
    col = float(cv2.compareHist(a[1], b[1], cv2.HISTCMP_BHATTACHARYYA))
    return 0.5 * pix + 0.5 * min(1.0, max(0.0, col))


def sample_scenes(path, threshold=0.3, check_fps=5, min_gap=0.5, max_gap=10.0, max_frames=None):
    """Change-based sampling: keep a frame when the picture changes, not on a fixed clock.

    Only ~`check_fps` frames per second are decoded (the rest are skipped with grab(), so
    high-frame-rate video stays cheap). Each checked frame is compared with the last KEPT
    frame (so slow drift still adds up); it is kept when the change score exceeds
    `threshold` and at least `min_gap` seconds passed since the last keep, or when
    `max_gap` seconds passed with no keep (so long static shots still get a timestamp).
    The first frame is always kept.

    Returns ([(time_sec, PIL.Image), ...], duration_sec), like sample_frames.
    """
    cap, fps = _open(path)
    stride = max(1, round(fps / check_fps)) if check_fps else 1
    frames, idx, t = [], 0, 0.0
    last_sig, last_t = None, None
    while cap.grab():
        t = _pos_sec(cap, idx, fps)
        idx += 1
        if (idx - 1) % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        sig = _signature(frame)
        if last_sig is None:
            keep = True
        else:
            gap = t - last_t
            keep = gap >= max_gap or (gap >= min_gap and scene_change_score(last_sig, sig) > threshold)
        if keep:
            frames.append((t, _to_pil(frame)))
            last_sig, last_t = sig, t
            if max_frames and len(frames) >= max_frames:
                break
    cap.release()
    return frames, (t + 1.0 / fps if idx else 0.0)


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

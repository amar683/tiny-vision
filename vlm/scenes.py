"""Scene-by-scene video breakdown: what each scene shows, so an LLM can summarize the video.

A plain timeline gives one caption per sampled frame. Here, instead, the video is split
into scenes (shots), a few frames per scene are captioned (start, middle, end), and
consecutive shots that show the same thing are grouped. Each scene gets its captions and
keywords, plus an overview of what recurs across the video. The result is compact text
that Claude or any LLM can turn into a real summary, or that a local LLM can summarize
(see vlm/summarize.py).

Visual only for now: no audio or speech.
"""
import re
from collections import Counter

from .video import _open, _pos_sec, _signature, _to_pil, _ts, scene_change_score

# words that carry no content in COCO-style captions
STOP = set("""
a an the and or of in on at to with without by for from into onto over under near next
is are was were be being been it its this that there their his her he she they them
some many lots lot few several two three four five six group couple bunch pair one
while as up down out off front back top side behind around through across along
very large small big little white black red blue green yellow brown gray grey orange
pink purple dark light other each who what which where has have having sitting standing
looking holding wearing sits stands looks holds wears made filled full shown image picture
close view background foreground area topped
""".split())

_WORD = re.compile(r"[a-z]+")


def keywords(captions, k=5):
    words = Counter()
    for c in captions:
        words.update(set(w for w in _WORD.findall(c.lower()) if w not in STOP and len(w) > 2))
    return [w for w, _ in words.most_common(k)]


def _shrink(frame, short=320):
    """Downscale a BGR frame so a scene can buffer several of them cheaply."""
    import cv2

    h, w = frame.shape[:2]
    s = short / min(h, w)
    return frame if s >= 1 else cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def detect_shots(path, threshold=0.3, drift=0.55, check_fps=5, min_shot=1.0,
                 frames_per_shot=3, max_shots=300):
    """Split a video into shots. Returns ([(start, end, [(t, PIL.Image), ...]), ...], duration).

    A new shot starts on a hard cut (change vs the previous checked frame > threshold) or
    after slow drift (change vs the shot's first frame > drift, e.g. a long pan). Shots
    shorter than min_shot are absorbed. Up to frames_per_shot frames are picked evenly
    from each shot. Only ~check_fps frames/s are decoded, and at most 16 small frames per
    shot are buffered, so long and high-frame-rate videos stay cheap.
    """
    cap, fps = _open(path)
    stride = max(1, round(fps / check_fps)) if check_fps else 1
    shots, buf = [], []
    start_t = prev_sig = start_sig = None
    idx, t = 0, 0.0

    def close(end):
        if buf:
            n = min(frames_per_shot, len(buf))
            picks = [buf[round(i * (len(buf) - 1) / max(n - 1, 1))] for i in range(n)]
            shots.append((start_t, end, [(pt, _to_pil(f)) for pt, f in picks]))

    while cap.grab():
        t = _pos_sec(cap, idx, fps)
        idx += 1
        if (idx - 1) % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        sig = _signature(frame)
        if start_t is None:
            start_t, start_sig = t, sig
        elif t - start_t >= min_shot and (scene_change_score(prev_sig, sig) > threshold
                                          or scene_change_score(start_sig, sig) > drift):
            close(t)
            if len(shots) >= max_shots:
                break
            buf, start_t, start_sig = [], t, sig
        prev_sig = sig
        buf.append((t, _shrink(frame)))
        if len(buf) > 16:          # keep an even spread of at most 16 frames per shot
            buf = buf[::2]
    duration = t + 1.0 / fps if idx else 0.0
    if len(shots) < max_shots:
        close(duration)
    cap.release()
    return shots, duration


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a and b else 0.0


def build_scenes(shots, captions, group_threshold=0.5, intercut_max=5.0):
    """Group shots into scenes.

    shots: from detect_shots; captions: one list of captions per shot (same order).
    Two shots "show the same thing" when their keyword sets overlap by >= group_threshold
    (Jaccard). A shot joins the current scene when it matches any of the scene's views,
    and an A-B-A pattern (cutting back and forth, e.g. two people talking, or an interview
    with a cutaway) folds into one "intercut" scene, but only when the B shot is short
    (<= intercut_max seconds), so a real scene between two similar ones stays separate.
    """
    def matches(sc, kw):
        return any(_jaccard(v, kw) >= group_threshold for v in sc["_views"])

    def absorb(sc, s_end, n_shots, caps, views):
        sc["end"], sc["shots"] = s_end, sc["shots"] + n_shots
        sc["captions"] = list(dict.fromkeys(sc["captions"] + caps))
        for v in views:
            if not any(_jaccard(v, w) >= group_threshold for w in sc["_views"]):
                sc["_views"].append(v)

    scenes = []
    for (s, e, _), caps in zip(shots, captions):
        kw = keywords(caps)
        if scenes and matches(scenes[-1], kw):
            absorb(scenes[-1], e, 1, caps, [kw])
        elif (len(scenes) >= 2 and matches(scenes[-2], kw) and scenes[-1]["shots"] == 1
              and scenes[-1]["end"] - scenes[-1]["start"] <= intercut_max):
            b = scenes.pop()                                    # A, B, A -> one intercut scene
            absorb(scenes[-1], b["end"], 1, b["captions"], b["_views"])
            absorb(scenes[-1], e, 1, caps, [kw])
        else:
            scenes.append({"start": s, "end": e, "shots": 1,
                           "captions": list(dict.fromkeys(caps)), "_views": [kw]})
    for sc in scenes:
        sc["intercut"] = len(sc["_views"]) > 1
        sc["keywords"] = keywords(sc["captions"])
        # show one caption per view first, so every side of an intercut is represented
        firsts = [next((c for c in sc["captions"] if set(keywords([c])) & set(v)), None) for v in sc["_views"]]
        firsts = [c for c in dict.fromkeys(firsts) if c]
        sc["captions"] = (firsts + [c for c in sc["captions"] if c not in firsts])[:4]
        del sc["_views"]
    return scenes


def overview(scenes, duration):
    seen = Counter()
    for sc in scenes:
        seen.update(set(sc["keywords"]))
    lengths = [sc["end"] - sc["start"] for sc in scenes] or [0]
    return {
        "duration": duration, "scenes": len(scenes), "shots": sum(sc["shots"] for sc in scenes),
        "avg_scene_sec": sum(lengths) / len(lengths), "longest_scene_sec": max(lengths),
        "recurring": [(w, n) for w, n in seen.most_common(8) if n > 1],
    }


def format_scenes(name, scenes, ov):
    lines = [f"video: {name} · {_ts(ov['duration'])} long · {ov['scenes']} scenes "
             f"({ov['shots']} shots) · avg scene {ov['avg_scene_sec']:.0f}s"]
    if ov["recurring"]:
        lines.append("recurring: " + ", ".join(f"{w} ({n} scenes)" for w, n in ov["recurring"]))
    for i, sc in enumerate(scenes, 1):
        shots = (f", {sc['shots']} shots, cutting back and forth" if sc.get("intercut")
                 else f", {sc['shots']} shots" if sc["shots"] > 1 else "")
        lines.append(f"\nscene {i} [{_ts(sc['start'])}–{_ts(sc['end'])}] "
                     f"({sc['end'] - sc['start']:.0f}s{shots})")
        lines.append("  seen: " + " | ".join(sc["captions"]))
        if sc["keywords"]:
            lines.append("  keywords: " + ", ".join(sc["keywords"]))
    lines.append("\n(captions come from a small local vision model: approximate, visual only, no audio)")
    return "\n".join(lines)


def describe_scenes(path, caption_fn, **detect_kw):
    """Full pipeline. caption_fn(list[PIL.Image]) -> list[str]. Returns (text, data dict)."""
    from pathlib import Path

    shots, duration = detect_shots(path, **detect_kw)
    flat = [im for _, _, frames in shots for _, im in frames]
    caps = caption_fn(flat) if flat else []
    per_shot, i = [], 0
    for _, _, frames in shots:
        per_shot.append(caps[i:i + len(frames)])
        i += len(frames)
    scenes = build_scenes(shots, per_shot)
    ov = overview(scenes, duration)
    return format_scenes(Path(path).name, scenes, ov), {"overview": ov, "scenes": scenes}

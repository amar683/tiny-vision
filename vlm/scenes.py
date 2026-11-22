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


def video_length(path):
    """Duration in seconds from the container (frame count / fps), without decoding. 0 if unknown."""
    import cv2

    cap, fps = _open(path)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps if n and n > 0 else 0.0


def iter_shots(path, threshold=0.3, drift=0.55, check_fps=5, min_shot=1.0,
               frames_per_shot=3, max_shots=0, info=None, progress=None):
    """Yield shots (start, end, [(t, PIL.Image), ...]) one by one as the video is read.

    A new shot starts on a hard cut (change vs the previous checked frame > threshold) or
    after slow drift (change vs the shot's first frame > drift, e.g. a long pan). Shots
    shorter than min_shot are absorbed. Up to frames_per_shot frames are picked evenly
    from each shot. Only ~check_fps frames/s are decoded, and at most 16 small frames per
    shot are buffered, so long and high-frame-rate videos stay cheap. Yielding shots as
    they close lets the caller caption and drop the images, so memory doesn't grow with
    the video's length.

    max_shots: stop after this many shots (0 = no limit).
    info: optional dict, filled with "duration" (whole video, s), "covered" (seconds
        described) and "truncated" (True if max_shots stopped it early).
    progress: optional callback(seconds_read) called about once per second of video read.
    """
    info = {} if info is None else info
    cap, fps = _open(path)
    stride = max(1, round(fps / check_fps)) if check_fps else 1
    buf, n_shots = [], 0
    start_t = prev_sig = start_sig = None
    idx, t, last_report = 0, 0.0, 0.0
    truncated = False

    def picks():
        n = min(frames_per_shot, len(buf))
        return [(pt, _to_pil(f)) for pt, f in
                (buf[round(i * (len(buf) - 1) / max(n - 1, 1))] for i in range(n))]

    try:
        while cap.grab():
            t = _pos_sec(cap, idx, fps)
            idx += 1
            if (idx - 1) % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            if progress and t - last_report >= 1.0:
                progress(t)
                last_report = t
            sig = _signature(frame)
            if start_t is None:
                start_t, start_sig = t, sig
            elif t - start_t >= min_shot and (scene_change_score(prev_sig, sig) > threshold
                                              or scene_change_score(start_sig, sig) > drift):
                yield start_t, t, picks()
                n_shots += 1
                if max_shots and n_shots >= max_shots:
                    truncated = True
                    break
                buf, start_t, start_sig = [], t, sig
            prev_sig = sig
            buf.append((t, _shrink(frame)))
            if len(buf) > 16:          # keep an even spread of at most 16 frames per shot
                buf = buf[::2]
        end = t + 1.0 / fps if idx else 0.0
        if truncated:
            info.update(covered=t, truncated=True, duration=max(video_length(path), end))
        else:
            if buf:
                yield start_t, end, picks()
            info.update(covered=end, truncated=False, duration=end)
    finally:
        cap.release()


def detect_shots(path, threshold=0.3, drift=0.55, check_fps=5, min_shot=1.0,
                 frames_per_shot=3, max_shots=0):
    """All shots at once: ([(start, end, [(t, PIL.Image), ...]), ...], duration). See iter_shots."""
    info = {}
    shots = list(iter_shots(path, threshold, drift, check_fps, min_shot, frames_per_shot,
                            max_shots, info))
    return shots, info.get("duration", 0.0)


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


def _pick_captions(captions, k):
    """Up to k captions that together cover the most distinct keywords (greedy)."""
    chosen, covered = [], set()
    pool = list(dict.fromkeys(captions))
    while pool and len(chosen) < k:
        best = max(pool, key=lambda c: len(set(keywords([c], 10)) - covered))
        chosen.append(best)
        covered |= set(keywords([best], 10))
        pool.remove(best)
    return chosen


def condense(scenes, max_scenes, captions_per_scene=3):
    """Merge neighbouring scenes until at most max_scenes remain, so long videos stay readable.

    Repeatedly merges the adjacent pair that is most alike (keyword overlap), preferring
    short pairs on ties, so similar stretches collapse first and distinct moments survive.
    Scenes are left untouched when there are already max_scenes or fewer.
    """
    if not max_scenes or len(scenes) <= max_scenes:
        return scenes
    sc = [dict(x, _caps=list(x["captions"]), _kw=set(x["keywords"])) for x in scenes]
    while len(sc) > max_scenes:
        def score(i):
            a, b = sc[i], sc[i + 1]
            return _jaccard(a["_kw"], b["_kw"]) - 1e-4 * (b["end"] - a["start"])
        i = max(range(len(sc) - 1), key=score)
        a, b = sc[i], sc.pop(i + 1)
        a["end"], a["shots"] = b["end"], a["shots"] + b["shots"]
        a["_caps"] = list(dict.fromkeys(a["_caps"] + b["_caps"]))
        a["_kw"] = set(keywords(a["_caps"], 8))
        a["intercut"] = False
    for x in sc:
        caps = x.pop("_caps")
        x.pop("_kw")
        x["keywords"] = keywords(caps)
        x["captions"] = _pick_captions(caps, captions_per_scene)
    return sc


def overview(scenes, duration, covered=None, truncated=False, condensed=False):
    seen = Counter()
    for sc in scenes:
        seen.update(set(sc["keywords"]))
    lengths = [sc["end"] - sc["start"] for sc in scenes] or [0]
    return {
        "duration": duration, "covered": duration if covered is None else covered,
        "truncated": truncated, "condensed": condensed,
        "scenes": len(scenes), "shots": sum(sc["shots"] for sc in scenes),
        "avg_scene_sec": sum(lengths) / len(lengths), "longest_scene_sec": max(lengths),
        "recurring": [(w, n) for w, n in seen.most_common(8) if n > 1],
    }


def format_scenes(name, scenes, ov):
    lines = [f"video: {name} · {_ts(ov['duration'])} long · {ov['scenes']} scenes "
             f"({ov['shots']} shots) · avg scene {ov['avg_scene_sec']:.0f}s"]
    if ov.get("truncated"):
        lines.append(f"NOTE: only 00:00–{_ts(ov['covered'])} of {_ts(ov['duration'])} is described "
                     f"(shot limit reached); the rest of the video is missing.")
    if ov.get("condensed"):
        lines.append(f"(long video: similar neighbouring scenes were merged to keep this to "
                     f"{ov['scenes']} scenes)")
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


def auto_frames_per_shot(duration):
    """Frames captioned per shot by video length: 3 up to 10 min, 2 up to 30 min, then 1."""
    return 3 if duration <= 600 else 2 if duration <= 1800 else 1


def describe_scenes(path, caption_fn, frames_per_shot=None, max_scenes=100, progress=None,
                    batch=32, **detect_kw):
    """Full pipeline. caption_fn(list[PIL.Image]) -> list[str]. Returns (text, data dict).

    Shots are captioned in batches as they are found and their images dropped, so a long
    video is covered end to end with flat memory. frames_per_shot=None picks it from the
    video's length (auto_frames_per_shot). Videos with more than max_scenes scenes are
    condensed (see condense) so the text stays a readable size; 0 = never condense.
    progress: optional callback(seconds_read, total_seconds).
    """
    from pathlib import Path

    total = video_length(path)
    fps_shot = frames_per_shot or auto_frames_per_shot(total)
    info, spans, per_shot, pending = {}, [], [], []   # pending: (shot index, image)

    def flush():
        caps = caption_fn([im for _, im in pending]) if pending else []
        for (k, _), c in zip(pending, caps):
            per_shot[k].append(c)
        pending.clear()

    report = (lambda t: progress(t, total)) if progress else None
    for s, e, frames in iter_shots(path, frames_per_shot=fps_shot, info=info, progress=report,
                                   **detect_kw):
        spans.append((s, e, None))
        per_shot.append([])
        pending.extend((len(spans) - 1, im) for _, im in frames)
        if len(pending) >= batch:
            flush()
    flush()
    scenes = build_scenes(spans, per_shot)
    condensed = bool(max_scenes) and len(scenes) > max_scenes
    scenes = condense(scenes, max_scenes)
    ov = overview(scenes, info.get("duration", 0.0), info.get("covered"), info.get("truncated", False),
                  condensed)
    return format_scenes(Path(path).name, scenes, ov), {"overview": ov, "scenes": scenes}

"""Fast end-to-end check: shapes, video tensors, timeline merging, and that the model really learns.

  python smoke_test.py
"""
import math
import random
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vlm import Captioner, WordTokenizer, get_config
from vlm.data import ShapesDataset, shape_caption_texts
from vlm.video import build_timeline, format_timeline, sample_frames, sample_scenes


def _write_cut_video(path, fps=30, size=(160, 90)):
    """~20 s video: three static shots with hard cuts at 3.1 s and 16.7 s. Returns cut times."""
    import cv2
    import numpy as np

    w, h = size
    a = np.full((h, w, 3), (40, 40, 200), np.uint8)              # red, white circle
    cv2.circle(a, (50, 45), 25, (255, 255, 255), -1)
    b = np.full((h, w, 3), (180, 90, 20), np.uint8)              # blue, yellow box
    cv2.rectangle(b, (90, 20), (140, 70), (0, 220, 255), -1)
    c = np.full((h, w, 3), (60, 170, 60), np.uint8)              # green, black triangle
    cv2.fillPoly(c, [np.array([[30, 80], [80, 10], [130, 80]], np.int32)], (0, 0, 0))
    shots = [(3.1, a), (13.6, b), (3.3, c)]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert out.isOpened(), "cv2.VideoWriter could not open MJPG writer"
    n = 0
    for sec, img in shots:
        end = round((n / fps + sec) * fps)
        while n < end:
            out.write(img)
            n += 1
    out.release()
    return [3.1, 3.1 + 13.6], n / fps


def check_video_sampling():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cuts.avi"
        cuts, total = _write_cut_video(path)

        frames, dur = sample_frames(path, every_sec=2.0)
        times = [t for t, _ in frames]
        assert abs(dur - total) < 0.1, (dur, total)
        assert len(times) == 10 and all(abs(t - 2 * i) < 0.05 for i, t in enumerate(times)), times

        frames, dur = sample_scenes(path, threshold=0.3, check_fps=5, min_gap=0.5, max_gap=10.0)
        times = [t for t, _ in frames]
        assert abs(dur - total) < 0.1, (dur, total)
        assert times[0] == 0.0, times
        for cut in cuts:  # every cut is caught, at most one check step (0.2 s) late
            assert any(0 <= t - cut <= 0.4 for t in times), (cut, times)
        # Long static shot 3.1-16.7 s: besides the cut, only the max_gap keep ~10 s later.
        inside = [t for t in times if cuts[0] + 0.4 < t < cuts[1]]
        first = next(t for t in times if t >= cuts[0])
        assert len(inside) == 1 and abs(inside[0] - (first + 10.0)) < 0.25, (inside, times)
        assert len(times) == 4, times  # start, cut 1, max_gap keep, cut 2
        print("ok: video sampling, scene keeps at " + ", ".join(f"{t:.1f}s" for t in times))


def check_scene_summary():
    """Shot detection finds the two cuts; a fake captioner groups shots into scenes."""
    from vlm.scenes import describe_scenes, detect_shots

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cuts.avi"
        cuts, total = _write_cut_video(path)
        shots, duration = detect_shots(path, frames_per_shot=3)
        starts = [s for s, _, _ in shots]
        assert len(shots) == 3, starts
        for cut, s in zip(cuts, starts[1:]):
            assert 0 <= s - cut <= 0.4, (cut, s)
        assert all(1 <= len(f) <= 3 for _, _, f in shots)
        # shots 1 and 2 get the same captions -> grouped into one scene; shot 3 differs
        names = iter(["a dog on grass"] * 3 + ["a dog on grass"] * 3 + ["a car on a road"] * 3)
        text, data = describe_scenes(path, lambda ims: [next(names) for _ in ims])
        assert [sc["shots"] for sc in data["scenes"]] == [2, 1], data["scenes"]
        assert "scene 2" in text and "car" in text
        assert not data["overview"]["truncated"] and abs(data["overview"]["duration"] - total) < 0.1
        # Shot limit: says how much was covered and still reports the full length.
        fake = lambda ims: ["a dog on grass"] * len(ims)
        text, data = describe_scenes(path, fake, max_shots=1)
        ov = data["overview"]
        assert ov["truncated"] and abs(ov["covered"] - cuts[0]) <= 0.4, ov
        assert abs(ov["duration"] - total) < 0.1 and "NOTE: only 00:00–00:03 of 00:20" in text, text
        # Condensing: a scene budget of 1 merges everything into one scene spanning the video.
        caps = iter(["a dog on grass"] * 3 + ["a cat on a sofa"] * 3 + ["a car on a road"] * 3)
        text, data = describe_scenes(path, lambda ims: [next(caps) for _ in ims], max_scenes=1)
        (sc,) = data["scenes"]
        assert data["overview"]["condensed"] and sc["shots"] == 3 and sc["start"] == 0, sc
        assert abs(sc["end"] - total) < 0.1 and len(sc["captions"]) == 3, sc
        print(f"ok: scene summary, shots start at " + ", ".join(f"{t:.1f}s" for t in starts)
              + "; shot limit and condensing work")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    random.seed(0)

    tok = WordTokenizer.train(shape_caption_texts(), min_freq=1)
    cfg = get_config("nano", vocab_size=len(tok))
    model = Captioner(cfg).to(device)
    print(f"device={device} params={model.num_params() / 1e6:.2f}M")

    # 1. Shapes: image and video inputs.
    img = torch.randn(2, 3, cfg.image_size, cfg.image_size, device=device)
    vid = torch.randn(2, 4, 3, cfg.image_size, cfg.image_size, device=device)
    n = model.encoder.tokens_per_frame
    assert model.encoder(img).shape == (2, n, cfg.enc_dim)
    assert model.encoder(vid).shape == (2, 4 * n, cfg.enc_dim)
    ids = torch.ones(2, 5, dtype=torch.long, device=device)
    assert model(vid, ids)[0].shape == (2, 5, len(tok))
    print("ok: image and video shapes")

    # 2. Timeline merging.
    segs = build_timeline([(0, "a dog runs on grass ."), (2, "a dog runs on the grass ."),
                           (4, "a man rides a bike ."), (6, "a man rides a bike .")], duration=8,
                          merge_threshold=0.6)
    assert len(segs) == 2 and segs[0][1] == 4 and segs[1][1] == 8, segs
    strict = build_timeline([(0, "a green square on the left ."), (3, "a green triangle on the left .")], 6)
    assert len(strict) == 2, "default must not merge captions that differ in a key word"
    print("ok: timeline\n" + format_timeline(segs))

    # 2b. Video frame sampling on a synthetic video with hard cuts.
    check_video_sampling()
    check_scene_summary()

    # 3. Learning: it must read color, shape AND position from pixels.
    train = ShapesDataset(20000, tok, cfg, seed=1)
    test = ShapesDataset(200, tok, cfg, seed=2)
    loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    model.train()
    steps = 1200
    t0, step, first = time.time(), 0, None
    while step < steps:
        for px, ids, tgt in loader:
            for g in opt.param_groups:  # warmup + cosine decay
                g["lr"] = 1e-3 * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
            _, loss = model(px.to(device), ids.to(device), tgt.to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            first = first or loss.item()
            if step % 200 == 0:
                print(f"step {step:4d} loss {loss.item():.3f}")
            step += 1
            if step >= steps:
                break
    print(f"trained {steps} steps in {time.time() - t0:.0f}s, loss {first:.2f} -> {loss.item():.3f}")

    model.eval()
    px = torch.stack([test[i][0] for i in range(len(test))]).to(device)
    preds = [tok.decode(o) for o in model.generate(px, tok.bos, tok.eos).tolist()]
    refs = [test.references(i)[0] for i in range(len(test))]
    norm = lambda s: s.replace(" .", ".")
    acc = sum(norm(p) == norm(r) for p, r in zip(preds, refs)) / len(refs)
    for p, r in list(zip(preds, refs))[:5]:
        print(f"  pred: {p:32s} ref: {r}")
    print(f"exact-match accuracy on unseen images: {acc:.0%} (chance ~4%)")
    assert acc > 0.5, "model failed to learn the shapes task"
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()

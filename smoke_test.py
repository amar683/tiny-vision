"""Fast end-to-end check: shapes, video tensors, timeline merging, and that the model really learns.

  python smoke_test.py
"""
import math
import random
import time

import torch
from torch.utils.data import DataLoader

from vlm import Captioner, WordTokenizer, get_config
from vlm.data import ShapesDataset, shape_caption_texts
from vlm.video import build_timeline, format_timeline


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

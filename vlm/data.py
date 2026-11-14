import random

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def image_transform(size, train):
    from torchvision import transforms as T

    # No horizontal flip: captions often say "left"/"right". No color jitter: they name colors.
    if train:
        resize = [T.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True)]
    else:
        resize = [T.Resize(size, antialias=True), T.CenterCrop(size)]
    return T.Compose(resize + [T.ToTensor(), T.Normalize(MEAN, STD)])


def encode_caption(tok, text, max_len):
    toks = tok.encode(text)[: max_len - 1]
    ids = [tok.bos] + toks
    tgt = toks + [tok.eos]
    pad = max_len - len(ids)
    return torch.tensor(ids + [tok.pad] * pad), torch.tensor(tgt + [-100] * pad)


# ---------------------------------------------------------------- Hugging Face datasets

def find_columns(hf_split):
    """Guess the image column and caption column(s) of an image-caption dataset."""
    from datasets import Image as HFImage

    feats = hf_split.features
    image_col = next(k for k, v in feats.items() if isinstance(v, HFImage))
    cap_cols = [k for k in feats if k != image_col and any(s in k.lower() for s in ("caption", "text", "sentence"))]
    if not cap_cols:
        raise ValueError(f"no caption column found in {list(feats)}")
    return image_col, cap_cols


def row_captions(row, cap_cols):
    caps = []
    for c in cap_cols:
        v = row[c]
        caps.extend(v if isinstance(v, list) else [v])
    return [c for c in caps if isinstance(c, str) and c.strip()]


def iter_caption_texts(hf_split, cap_cols):
    for row in hf_split.select_columns(cap_cols):
        yield from row_captions(row, cap_cols)


class HFCaptionDataset(Dataset):
    def __init__(self, hf_split, tok, cfg, train, image_col, cap_cols):
        self.ds, self.tok, self.cfg, self.train = hf_split, tok, cfg, train
        self.image_col, self.cap_cols = image_col, cap_cols
        self.tf = image_transform(cfg.image_size, train)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        caps = row_captions(row, self.cap_cols)
        cap = random.choice(caps) if self.train else caps[0]
        ids, tgt = encode_caption(self.tok, cap, self.cfg.max_text_len)
        return self.tf(row[self.image_col].convert("RGB")), ids, tgt

    def references(self, i):
        return row_captions(self.ds.select_columns(self.cap_cols)[i], self.cap_cols)


# ---------------------------------------------------------------- synthetic shapes (sanity check)

COLORS = {"red": (220, 40, 40), "green": (40, 170, 60), "blue": (40, 80, 220), "yellow": (230, 200, 30)}
SHAPES = ("square", "circle", "triangle")
SIDES = ("left", "right")


def make_shape_image(rng, size=128):
    color, shape, side = rng.choice(list(COLORS)), rng.choice(SHAPES), rng.choice(SIDES)
    img = Image.new("RGB", (size, size), (235, 235, 235))
    d = ImageDraw.Draw(img)
    r = rng.randint(size // 8, size // 5)
    cx = rng.randint(r, size // 2 - r) if side == "left" else rng.randint(size // 2 + r, size - r)
    cy = rng.randint(r, size - r)
    box = (cx - r, cy - r, cx + r, cy + r)
    if shape == "square":
        d.rectangle(box, fill=COLORS[color])
    elif shape == "circle":
        d.ellipse(box, fill=COLORS[color])
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=COLORS[color])
    return img, f"a {color} {shape} on the {side} ."


def shape_caption_texts():
    return [f"a {c} {s} on the {side} ." for c in COLORS for s in SHAPES for side in SIDES]


class ShapesDataset(Dataset):
    def __init__(self, n, tok, cfg, seed=0):
        self.n, self.tok, self.cfg, self.seed = n, tok, cfg, seed
        self.tf = image_transform(cfg.image_size, train=False)

    def __len__(self):
        return self.n

    def sample(self, i):
        return make_shape_image(random.Random(self.seed * 1_000_003 + i))

    def __getitem__(self, i):
        img, cap = self.sample(i)
        ids, tgt = encode_caption(self.tok, cap, self.cfg.max_text_len)
        return self.tf(img), ids, tgt

    def references(self, i):
        return [self.sample(i)[1]]

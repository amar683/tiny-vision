"""Word-level tokenizer built from the caption corpus. Simple and enough for v1 captions."""
import json
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class WordTokenizer:
    pad, bos, eos, unk = 0, 1, 2, 3

    def __init__(self, vocab):
        assert vocab[:4] == SPECIALS
        self.itos = list(vocab)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def split(text):
        return TOKEN_RE.findall(text.lower())

    @classmethod
    def train(cls, texts, max_vocab=8000, min_freq=2):
        counts = Counter()
        for t in texts:
            counts.update(cls.split(t))
        words = [w for w, n in counts.most_common(max_vocab - len(SPECIALS)) if n >= min_freq]
        return cls(SPECIALS + words)

    def encode(self, text):
        return [self.stoi.get(w, self.unk) for w in self.split(text)]

    def decode(self, ids):
        words = []
        for i in ids:
            i = int(i)
            if i == self.eos:
                break
            if i in (self.pad, self.bos):
                continue
            words.append(self.itos[i])
        return re.sub(r" ([.,!?;:])", r"\1", " ".join(words))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

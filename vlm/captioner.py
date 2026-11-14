"""Encoder + adapter + tiny caption decoder.

The decoder is a prefix-LM: visual tokens attend to each other freely, text tokens
attend causally to all visual tokens and earlier text. Later, the same encoder can
plug into Qwen/Llama by training a new Projector into that model's embedding space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import VisionEncoder
from .layers import Block, init_weights


class Projector(nn.Module):
    """The adapter: maps encoder tokens into a language model's embedding space."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class Captioner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = VisionEncoder(cfg)
        self.adapter = Projector(cfg.enc_dim, cfg.dec_dim)
        self.tok = nn.Embedding(cfg.vocab_size, cfg.dec_dim)
        self.text_pos = nn.Parameter(torch.zeros(1, cfg.max_text_len, cfg.dec_dim))
        self.img_type = nn.Parameter(torch.zeros(1, 1, cfg.dec_dim))
        self.blocks = nn.ModuleList(
            [Block(cfg.dec_dim, cfg.dec_heads, cfg.dropout) for _ in range(cfg.dec_depth)])
        self.norm = nn.RMSNorm(cfg.dec_dim)
        self.head = nn.Linear(cfg.dec_dim, cfg.vocab_size, bias=False)

        for m in (self.adapter, self.tok, self.blocks, self.head):
            m.apply(init_weights)
        self.head.weight = self.tok.weight  # weight tying
        nn.init.trunc_normal_(self.text_pos, std=0.02)
        nn.init.trunc_normal_(self.img_type, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _prefix_mask(P, T, device):
        L = P + T
        i = torch.arange(L, device=device)[:, None]
        j = torch.arange(L, device=device)[None, :]
        return (j <= i) | (j < P)  # True = may attend

    def encode(self, pixels):
        return self.adapter(self.encoder(pixels))

    def decode(self, vis, ids):
        P, T = vis.size(1), ids.size(1)
        x = torch.cat([vis + self.img_type, self.tok(ids) + self.text_pos[:, :T]], dim=1)
        mask = self._prefix_mask(P, T, x.device)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.norm(x[:, P:]))

    def forward(self, pixels, ids, targets=None):
        logits = self.decode(self.encode(pixels), ids)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, pixels, bos, eos, temperature=0.0, top_k=None):
        vis = self.encode(pixels)
        B = vis.size(0)
        ids = torch.full((B, 1), bos, dtype=torch.long, device=vis.device)
        done = torch.zeros(B, dtype=torch.bool, device=vis.device)
        for _ in range(self.cfg.max_text_len - 1):
            logits = self.decode(vis, ids)[:, -1].float()
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, top_k).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, -1), 1).squeeze(1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            done |= nxt == eos
            if done.all():
                break
        return ids[:, 1:]

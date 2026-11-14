"""The universal part: a small ViT that turns images or video frames into visual tokens.

Any language model can sit on top of it through a small adapter (see Projector in
captioner.py). Frames are encoded independently, then tagged with a temporal embedding
so the model above knows which frame each token came from.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Block, init_weights


def sincos_1d(n, dim):
    """Fixed sine-cosine embedding, (n, dim)."""
    assert dim % 2 == 0
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    return torch.cat([torch.sin(pos * freq), torch.cos(pos * freq)], dim=1)


def sincos_2d(grid, dim):
    """Fixed 2D embedding for a grid x grid patch layout (row-major), (grid*grid, dim).

    Fixed (not learned) so the model knows *where* every patch is from step 0. With a
    small learned embedding the tiny model can't tell left from right for a long time.
    """
    assert dim % 4 == 0
    rows = sincos_1d(grid, dim // 2)[:, None, :].expand(grid, grid, dim // 2)
    cols = sincos_1d(grid, dim // 2)[None, :, :].expand(grid, grid, dim // 2)
    return torch.cat([rows, cols], dim=-1).reshape(grid * grid, dim)


class VisionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.image_size % cfg.patch_size == 0
        self.grid = cfg.image_size // cfg.patch_size
        assert self.grid % cfg.pool == 0, "patch grid must be divisible by pool"
        self.pool = cfg.pool
        self.dim = cfg.enc_dim
        self.max_frames = cfg.max_frames

        # Patch embedding as reshape + Linear. Same math as Conv2d(kernel=stride=patch), but
        # ~30x faster on Apple GPUs, where that conv is poorly optimized (measured on M5).
        self.patch_size = cfg.patch_size
        self.patch = nn.Linear(3 * cfg.patch_size ** 2, cfg.enc_dim)
        self.register_buffer("pos", sincos_2d(self.grid, cfg.enc_dim)[None], persistent=False)
        self.register_buffer("time", sincos_1d(cfg.max_frames, cfg.enc_dim)[None, :, None], persistent=False)
        self.blocks = nn.ModuleList(
            [Block(cfg.enc_dim, cfg.enc_heads, cfg.dropout) for _ in range(cfg.enc_depth)])
        self.norm = nn.RMSNorm(cfg.enc_dim)

        self.apply(init_weights)

    @property
    def tokens_per_frame(self):
        return (self.grid // self.pool) ** 2

    def forward(self, pixels):
        """pixels: (B, 3, H, W) image or (B, T, 3, H, W) clip -> (B, T * tokens_per_frame, enc_dim)."""
        if pixels.dim() == 4:
            pixels = pixels.unsqueeze(1)
        B, T = pixels.shape[:2]
        assert T <= self.max_frames, f"got {T} frames, max is {self.max_frames}"

        g, p = self.grid, self.patch_size
        x = pixels.reshape(B * T, 3, g, p, g, p).permute(0, 2, 4, 1, 3, 5)  # (B*T, g, g, 3, p, p)
        x = self.patch(x.reshape(B * T, g * g, 3 * p * p)) + self.pos     # (B*T, g*g, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.pool > 1:
            g = self.grid
            x = x.transpose(1, 2).reshape(B * T, self.dim, g, g)
            x = F.avg_pool2d(x, self.pool)
            x = x.flatten(2).transpose(1, 2)           # (B*T, N, D)

        x = x.view(B, T, -1, self.dim) + self.time[:, :T]
        return x.flatten(1, 2)

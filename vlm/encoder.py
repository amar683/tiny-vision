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


# ---------------------------------------------------------------- frozen pretrained encoder

# facebook/dinov2-small architecture, so the model can be rebuilt offline from our own
# model.safetensors (the weights are bundled there); from_pretrained is only used at train time.
DINOV2_SMALL = dict(hidden_size=384, num_hidden_layers=12, num_attention_heads=6, mlp_ratio=4,
                    patch_size=14, image_size=518, layerscale_value=1.0, layer_norm_eps=1e-6,
                    hidden_act="gelu", qkv_bias=True, use_swiglu_ffn=False)
PRETRAINED = {"dinov2-small": ("facebook/dinov2-small", DINOV2_SMALL)}


class PretrainedEncoder(nn.Module):
    """A frozen pretrained ViT with the same output contract as VisionEncoder.

    Used as a comparison (and possibly the release encoder): only the adapter and decoder
    train on top of it. Patch tokens are pooled and time-tagged exactly like VisionEncoder.
    """

    def __init__(self, cfg, load_weights=False):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model

        repo, arch = PRETRAINED[cfg.encoder]
        assert cfg.enc_dim == arch["hidden_size"], f"enc_dim must be {arch['hidden_size']} for {cfg.encoder}"
        self.vit = Dinov2Model.from_pretrained(repo) if load_weights else Dinov2Model(Dinov2Config(**arch))
        self.vit.requires_grad_(False)
        self.patch_size = arch["patch_size"]
        self.grid = cfg.image_size // self.patch_size
        assert self.grid * self.patch_size == cfg.image_size and self.grid % cfg.pool == 0
        self.pool, self.dim, self.max_frames = cfg.pool, cfg.enc_dim, cfg.max_frames
        # our transforms normalize with mean=std=0.5; DINOv2 expects ImageNet stats
        self.register_buffer("in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("time", sincos_1d(cfg.max_frames, cfg.enc_dim)[None, :, None], persistent=False)

    @property
    def tokens_per_frame(self):
        return (self.grid // self.pool) ** 2

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()  # frozen: no dropout / stochastic depth, ever
        return self

    def forward(self, pixels):
        if pixels.dim() == 4:
            pixels = pixels.unsqueeze(1)
        B, T = pixels.shape[:2]
        assert T <= self.max_frames, f"got {T} frames, max is {self.max_frames}"
        x = pixels.flatten(0, 1) * 0.5 + 0.5                       # back to [0, 1]
        x = (x - self.in_mean) / self.in_std
        with torch.no_grad():
            x = self.vit(pixel_values=x).last_hidden_state[:, 1:]  # drop CLS -> (B*T, g*g, D)
        g = self.grid
        x = x.transpose(1, 2).reshape(B * T, self.dim, g, g)
        if self.pool > 1:
            x = F.avg_pool2d(x, self.pool)
        x = x.flatten(2).transpose(1, 2)
        x = x.view(B, T, -1, self.dim) + self.time[:, :T]
        return x.flatten(1, 2)


def build_encoder(cfg, load_weights=False):
    if cfg.encoder == "scratch":
        return VisionEncoder(cfg)
    return PretrainedEncoder(cfg, load_weights=load_weights)

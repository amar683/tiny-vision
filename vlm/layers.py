import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        p = self.dropout if self.training else 0.0
        if x.device.type == "mps":
            # PyTorch's fused SDPA is ~1.5x slower than plain matmul+softmax on Apple GPUs
            # at these sizes (measured on M5), so use the manual path there.
            att = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
            if mask is not None:
                att = att.masked_fill(~mask, float("-inf"))
            att = F.dropout(att.softmax(-1), p, self.training)
            out = att @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        return self.proj(out.transpose(1, 2).reshape(B, L, D))


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = int(8 * dim / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    """Pre-norm transformer block (RMSNorm + attention + SwiGLU)."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.n1 = nn.RMSNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.n2 = nn.RMSNorm(dim)
        self.mlp = SwiGLU(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.n1(x), mask))
        x = x + self.drop(self.mlp(self.n2(x)))
        return x


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
    elif isinstance(module, nn.Conv2d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

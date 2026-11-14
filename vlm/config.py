from dataclasses import dataclass, asdict, replace


@dataclass
class ModelConfig:
    # Vision encoder (ViT)
    image_size: int = 224
    patch_size: int = 16
    enc_dim: int = 384
    enc_depth: int = 6
    enc_heads: int = 6
    pool: int = 2            # average-pool the patch grid: 2 -> 4x fewer visual tokens
    max_frames: int = 16     # max video frames per clip (temporal embedding size)
    # Caption decoder (small GPT-style, prefix-LM over visual tokens)
    dec_dim: int = 384
    dec_depth: int = 6
    dec_heads: int = 6
    vocab_size: int = 8000   # overwritten by the tokenizer at train time
    max_text_len: int = 40
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


PRESETS = {
    # ~1M params, 64px images. For smoke tests and pipeline debugging.
    "nano": ModelConfig(image_size=64, patch_size=8, enc_dim=128, enc_depth=3, enc_heads=4,
                        dec_dim=128, dec_depth=3, dec_heads=4, dropout=0.0),
    # ~25M params. First real model; trains on an M-series Mac.
    "tiny": ModelConfig(),
    # ~55M params. Overnight on the Mac, or quick on a cloud GPU.
    "small": ModelConfig(enc_dim=512, enc_depth=8, enc_heads=8,
                         dec_dim=512, dec_depth=8, dec_heads=8),
}


def get_config(name: str, **overrides) -> ModelConfig:
    return replace(PRESETS[name], **overrides)

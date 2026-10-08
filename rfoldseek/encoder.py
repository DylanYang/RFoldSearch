"""The 3Dr tokenizer network: descriptor encoder and nearest-codeword lookup.

The tokenizer was trained as a VQ-VAE; tokenizing needs only its encoder and
the learned 80-state codebook. The decoder and the training-time codebook
updates are not part of this package, and their weights in the checkpoint
are ignored.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorEncoder(nn.Module):
    """MLP encoder: descriptor → latent z (L2-normalised by default)."""

    def __init__(
        self,
        d_in: int = 49,
        d_hidden: int = 256,
        d_out: int = 64,
        n_layers: int = 1,
        l2_normalize: bool = True,
    ):
        super().__init__()
        assert n_layers >= 1, "encoder must have at least one Linear layer"
        self.norm = nn.LayerNorm(d_in)
        mods: list[nn.Module] = []
        if n_layers == 1:
            mods.append(nn.Linear(d_in, d_out))
        else:
            mods.append(nn.Linear(d_in, d_hidden))
            mods.append(nn.SiLU())
            for _ in range(n_layers - 2):
                mods.append(nn.Linear(d_hidden, d_hidden))
                mods.append(nn.SiLU())
            mods.append(nn.Linear(d_hidden, d_out))
        self.net = nn.Sequential(*mods)
        self.l2_normalize = l2_normalize

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        z = self.net(self.norm(descriptor))
        if self.l2_normalize:
            z = F.normalize(z, dim=-1)
        return z


class Tokenizer(nn.Module):
    """Descriptor (B, L, D) → 3Dr token indices (B, L)."""

    def __init__(self, encoder: DescriptorEncoder, n_codes: int, d_latent: int):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("codebook", torch.zeros(n_codes, d_latent))

    @property
    def n_codes(self) -> int:
        return self.codebook.shape[0]

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        z = self.encoder(descriptor)
        flat = z.reshape(-1, z.shape[-1])
        # Squared distance to every codeword, in the same form as training,
        # so ties and rounding resolve to the same token.
        distances = (
            (flat**2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + (self.codebook.t() ** 2).sum(dim=0, keepdim=True)
        )
        return torch.argmin(distances, dim=1).view(*z.shape[:-1])


def tokenizer_from_checkpoint(ckpt: dict) -> Tokenizer:
    """Build a ``Tokenizer`` from a VQ-VAE training checkpoint."""
    sd = ckpt["model"]
    n_enc_layers = ckpt.get("n_enc_layers")
    if n_enc_layers is None:
        n_enc_layers = 1
        while "encoder.net.%d.weight" % (2 * n_enc_layers) in sd:
            n_enc_layers += 1
    codebook = sd["codebook.embeddings"]
    encoder = DescriptorEncoder(
        d_in=ckpt.get("d_descriptor", sd["encoder.norm.weight"].shape[0]),
        d_hidden=ckpt.get("d_hidden", sd["encoder.net.0.weight"].shape[0]),
        d_out=ckpt.get("d_latent", codebook.shape[1]),
        n_layers=n_enc_layers,
        l2_normalize=ckpt.get("l2_normalize_encoder", True),
    )
    model = Tokenizer(encoder, codebook.shape[0], codebook.shape[1])
    state = {k: v for k, v in sd.items() if k.startswith("encoder.")}
    state["codebook"] = codebook
    model.load_state_dict(state)
    return model

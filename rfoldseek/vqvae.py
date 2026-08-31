# Ported from RFoldseek-paper-code src/models/vqvae.py; only imports adjusted.
"""RNA Structure descriptor VQ-VAE.

Per-residue structural descriptors are computed in `descriptors/`
(SHAPE / FRAME / SCALAR families); this file only contains the
VQ-VAE encoder, EMA codebook, decoder, and the reconstruction loss.

See `descriptors/__init__.py` for the descriptor layout, and
`descriptors/compose.py` for `compute_descriptor` / `parse_descriptor`.
"""
from __future__ import annotations

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook import EMACodebook
from .descriptors import compute_descriptor, parse_descriptor


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


class DescriptorDecoder(nn.Module):
    """MLP decoder: quantised latent z_q → descriptor reconstruction."""

    def __init__(self, d_in: int = 64, d_hidden: int = 128, d_out: int = 49, n_layers: int = 4):
        super().__init__()
        assert n_layers >= 1, "decoder must have at least one Linear layer"
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

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.net(z_q)


def infer_n_enc_layers(state_dict) -> int:
    n = 0
    while "encoder.net.%d.weight" % (2 * n) in state_dict:
        n += 1
    return max(1, n)


def infer_n_dec_layers(state_dict) -> int:
    n = 0
    while "decoder.net.%d.weight" % (2 * n) in state_dict:
        n += 1
    return max(1, n)


def infer_decoder_out_dim(state_dict) -> int:
    n_dec = infer_n_dec_layers(state_dict)
    return state_dict["decoder.net.%d.weight" % (2 * (n_dec - 1))].shape[0]


class DescriptorVQVAE(nn.Module):
    """RNA Structure Descriptor VQ-VAE."""

    def __init__(
        self,
        d_descriptor: int = 49,
        d_hidden: int = 256,
        d_latent: int = 64,
        n_codes: int = 60,
        n_enc_layers: int = 1,
        n_dec_layers: int = 4,
        no_random_restart: bool = True,
        ema_decay: float = 0.99,
        l2_normalize_encoder: bool = True,
        l2_normalize_codebook: bool = True,
    ):
        super().__init__()
        self.encoder = DescriptorEncoder(
            d_descriptor,
            d_hidden,
            d_latent,
            n_layers=n_enc_layers,
            l2_normalize=l2_normalize_encoder,
        )
        self.codebook = EMACodebook(
            n_codes,
            d_latent,
            no_random_restart=no_random_restart,
            restart_thres=1.0,
            ema_decay=ema_decay,
            l2_normalize=l2_normalize_codebook,
        )
        self.decoder = DescriptorDecoder(d_latent, d_hidden, d_descriptor, n_layers=n_dec_layers)

    def forward(self, descriptor: torch.Tensor) -> dict[str, torch.Tensor]:
        # Codebook expects 3D (B, L, D); add dummy L=1 dim for flat 2D input
        squeezed = descriptor.ndim == 2
        if squeezed:
            descriptor = descriptor.unsqueeze(1)

        z = self.encoder(descriptor)
        z_q, indices, commitment_loss = self.codebook(z)
        f_hat = self.decoder(z_q)

        if squeezed:
            z = z.squeeze(1)
            z_q = z_q.squeeze(1)
            indices = indices.squeeze(1)
            f_hat = f_hat.squeeze(1)

        return {
            "z": z,
            "z_q": z_q,
            "indices": indices,
            "commitment_loss": commitment_loss,
            "f_hat": f_hat,
        }

    @torch.no_grad()
    def encode(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        pair_idx: torch.Tensor | None = None,
        stack_idx: torch.Tensor | None = None,
        pair_orient: torch.Tensor | None = None,
    ) -> torch.Tensor:
        descriptor, _ = compute_descriptor(
            coords,
            mask,
            pair_idx,
            stack_idx,
            pair_orient=pair_orient,
        )
        z = self.encoder(descriptor)
        _, indices, _ = self.codebook(z)
        return indices


def descriptor_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    beta_commit: float = 0.25,
    lambda_geom: float = 1.0,
    lambda_pair: float = 1.0,
    lambda_stack: float = 1.0,
    lambda_mask: float = 1.0,
    pair_stack_reduce: str = "sum",
    lambda_latent_norm: float = 0.0,
    active_blocks: set | None = None,
) -> dict[str, torch.Tensor]:
    """Three-group reconstruction loss for the 49-D structural descriptor.

    The descriptor is bimodal: g_bp / g_st are zero when no partner
    exists, real geometry otherwise. Pure MSE blurs the two modes,
    so we decompose:

      1. Continuous geometry (g_nn1 + g_nn2 + g_ctx): pure MSE on
         z-score-normalised values.
      2. Conditional FRAME (g_bp, g_st): MSE × binary mask — only count
         residues that actually have a partner; denominator = number of
         positives.
      3. Mask prediction (m_bp, m_st): BCE-with-logits on {0, 1} targets.

    Mask dims (m_bp at 30, m_st at 41 in the 49-D layout) are left
    unnormalised (mean=0 / std=1) in desc_stats.pt.
    """
    if pair_stack_reduce not in {"sum", "mean"}:
        raise ValueError("pair_stack_reduce must be 'sum' or 'mean'")

    pred = parse_descriptor(output["f_hat"])
    gt = parse_descriptor(target)

    # If active_blocks is None, all blocks contribute (default / full model).
    # In feature-ablation runs, pass the subset that should drive the loss so
    # that zeroed dimensions do not bias L_geom magnitudes or L_mask BCE.
    all_blocks = {"g_nn1", "g_nn2", "g_ctx", "g_bp", "g_st"}
    active = all_blocks if active_blocks is None else set(active_blocks)

    # 1. Continuous geometry — per-block MSEs restricted to active blocks.
    #    Aggregated by SUM over active geometry blocks (the production
    #    convention reported in the paper: L_rec = MSE(g_nn1) + MSE(g_ctx)).
    #    Each block contributes its own element-wise MSE; blocks not in
    #    `active` (e.g. the zeroed g_nn2 in the 17-D production descriptor)
    #    are excluded entirely so they never enter the objective.
    geom_terms = []
    if "g_nn1" in active:
        geom_terms.append((gt["g_nn1"] - pred["g_nn1"]).pow(2).mean())
    if "g_nn2" in active:
        geom_terms.append((gt["g_nn2"] - pred["g_nn2"]).pow(2).mean())
    if "g_ctx" in active:
        geom_terms.append((gt["g_ctx"] - pred["g_ctx"]).pow(2).mean())
    if not geom_terms:
        L_geom = target.new_zeros(())
    else:
        L_geom = torch.stack(geom_terms).sum()

    # 2. Conditional FRAME (only residues that have a partner).
    m_bp = gt["m_bp"].squeeze(-1)
    m_st = gt["m_st"].squeeze(-1)
    n_bp = m_bp.sum().clamp(min=1.0)
    n_st = m_st.sum().clamp(min=1.0)

    bp_sq = (gt["g_bp"] - pred["g_bp"]).pow(2)
    st_sq = (gt["g_st"] - pred["g_st"]).pow(2)

    if pair_stack_reduce == "mean":
        d_bp = gt["g_bp"].shape[-1]
        d_st = gt["g_st"].shape[-1]
        L_pair = (bp_sq * m_bp.unsqueeze(-1)).sum() / (n_bp * d_bp)
        L_stack = (st_sq * m_st.unsqueeze(-1)).sum() / (n_st * d_st)
    else:
        L_pair = (bp_sq.sum(-1) * m_bp).sum() / n_bp
        L_stack = (st_sq.sum(-1) * m_st).sum() / n_st

    if "g_bp" not in active:
        L_pair = target.new_zeros(())
    if "g_st" not in active:
        L_stack = target.new_zeros(())

    # 3. Mask prediction (BCE on {0, 1}) — only on active mask blocks.
    mask_terms = []
    if "g_bp" in active:
        mask_terms.append(
            F.binary_cross_entropy_with_logits(pred["m_bp"].squeeze(-1), m_bp, reduction="mean")
        )
    if "g_st" in active:
        mask_terms.append(
            F.binary_cross_entropy_with_logits(pred["m_st"].squeeze(-1), m_st, reduction="mean")
        )
    L_mask = torch.stack(mask_terms).sum() if mask_terms else target.new_zeros(())

    L_recon = (
        lambda_geom * L_geom + lambda_pair * L_pair + lambda_stack * L_stack + lambda_mask * L_mask
    )
    L_commit = beta_commit * output["commitment_loss"]
    L_latent = lambda_latent_norm * output["z"].pow(2).mean()

    return {
        "loss": L_recon + L_commit + L_latent,
        "L_recon": L_recon,
        "L_geom": L_geom,
        "L_pair": L_pair,
        "L_stack": L_stack,
        "L_mask": L_mask,
        "L_commit": L_commit,
        "L_latent": L_latent,
    }

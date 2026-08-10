"""Load the deployed VQ-VAE tokenizer and encode structures to 3Dr tokens.

`load_model` / `prepare_sample` / `encode_entry` are ported unchanged from
RFoldseek-paper-code src/search/build_database.py; the normalisation
statistics travel inside the checkpoint and are applied here.
"""
import numpy as np
import torch

from .vqvae import (
    DescriptorVQVAE,
    compute_descriptor,
    infer_decoder_out_dim,
    infer_n_dec_layers,
    infer_n_enc_layers,
)


def load_model(checkpoint_path: str, device: torch.device = torch.device("cpu")):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["model"]
    d_descriptor = ckpt.get("d_descriptor", infer_decoder_out_dim(sd))
    d_hidden = ckpt.get("d_hidden", sd["encoder.net.0.weight"].shape[0])
    d_latent = ckpt.get("d_latent", sd["codebook.embeddings"].shape[1])
    n_codes = sd["codebook.embeddings"].shape[0]
    model = DescriptorVQVAE(
        d_descriptor=d_descriptor,
        d_hidden=d_hidden,
        d_latent=d_latent,
        n_codes=n_codes,
        n_enc_layers=ckpt.get("n_enc_layers", infer_n_enc_layers(sd)),
        n_dec_layers=ckpt.get("n_dec_layers", infer_n_dec_layers(sd)),
        ema_decay=ckpt.get("ema_decay", 0.99),
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt, n_codes, d_descriptor


def prepare_sample(entry, device="cpu"):
    coords = torch.tensor(entry["coords"], dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.tensor(entry["mask"], dtype=torch.bool).unsqueeze(0).to(device)
    pair_key = "pair_idx" if "pair_idx" in entry else "pair_edges"
    stack_key = "stack_idx" if "stack_idx" in entry else "stack_edges"
    pair_edges = torch.tensor(entry[pair_key], dtype=torch.long).unsqueeze(0).to(device)
    stack_edges = torch.tensor(entry[stack_key], dtype=torch.long).unsqueeze(0).to(device)
    pair_orient = None
    if "pair_orient" in entry:
        pair_orient = torch.tensor(entry["pair_orient"], dtype=torch.long).unsqueeze(0).to(device)
    if mask.sum() == 0:
        return None
    if pair_edges.ndim == 3:
        pair_edges = pair_edges[..., 0]
    if stack_edges.ndim == 3:
        stack_edges = stack_edges[..., 0]
    if pair_orient is not None and pair_orient.ndim == 3:
        pair_orient = pair_orient[..., 0]
    if (coords.shape[1] != mask.shape[1]
            or pair_edges.shape[1] != mask.shape[1]
            or stack_edges.shape[1] != mask.shape[1]
            or (pair_orient is not None and pair_orient.shape[1] != mask.shape[1])):
        return None
    return coords, mask, pair_edges, stack_edges, pair_orient


@torch.no_grad()
def encode_entry(model, entry, device, norm_mean, norm_std):
    sample = prepare_sample(entry, device)
    if sample is None:
        return None
    coords, mask, pair_edges, stack_edges, pair_orient = sample
    descriptor, _info = compute_descriptor(
        coords, mask, pair_edges, stack_edges, pair_orient=pair_orient)
    if norm_mean is not None and norm_std is not None:
        descriptor = (descriptor - norm_mean.to(device)) / norm_std.to(device)
    output = model(descriptor)
    valid_mask = mask[0].cpu()
    tokens = output["indices"][0].cpu()[valid_mask].numpy().astype(np.int16)
    return {
        "id": entry.get("id", ""),
        "tokens": tokens,
        "coords": coords[0].cpu().numpy(),
        "mask": mask[0].cpu().numpy(),
        "n_valid": int(valid_mask.sum()),
    }

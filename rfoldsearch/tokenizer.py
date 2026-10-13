"""Load the deployed 3Dr tokenizer and encode structures to 3Dr tokens.

The normalisation statistics travel inside the checkpoint and are applied
here.
"""
import numpy as np
import torch

from .descriptors import compute_descriptor
from .encoder import tokenizer_from_checkpoint

# Byte layout of the 49-D descriptor produced by compute_descriptor.
# See rfoldsearch/descriptors/compose.py for the docstring / diagram.
_DESCRIPTOR_BLOCK_SPAN = {
    "g_nn1": (0, 10),
    "g_nn2": (10, 20),
    "g_bp":  (20, 30),
    "m_bp":  (30, 31),
    "g_st":  (31, 41),
    "m_st":  (41, 42),
    "g_ctx": (42, 49),
}


def active_block_mask(active_blocks, d_descriptor=49, dtype=torch.float32):
    """Return a (d_descriptor,) tensor with 1s on active-block dims, 0 elsewhere.

    The paper's tokenizer was trained with a subset of descriptor blocks
    contributing to the reconstruction loss (recorded on the checkpoint as
    `active_blocks`), and the same subset was fed to the encoder at inference:
    inactive dims were zeroed after the (mean, std) normalisation so the
    encoder saw a consistent input. Reproducing the shipped token database
    requires the same zeroing at inference. When `active_blocks` is empty or
    None, every block is treated as active (backward compatible).
    """
    if not active_blocks:
        return torch.ones(d_descriptor, dtype=dtype)
    keep = set(active_blocks)
    m = torch.zeros(d_descriptor, dtype=dtype)
    for block, (start, end) in _DESCRIPTOR_BLOCK_SPAN.items():
        if block in keep:
            m[start:end] = 1.0
    return m


def load_model(checkpoint_path: str, device: torch.device = torch.device("cpu")):
    """-> (tokenizer, checkpoint dict, number of states, descriptor width)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = tokenizer_from_checkpoint(ckpt).to(device)
    model.eval()
    return model, ckpt, model.n_codes, model.encoder.norm.normalized_shape[0]


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
def encode_entry(model, entry, device, norm_mean, norm_std, active_blocks=None):
    sample = prepare_sample(entry, device)
    if sample is None:
        return None
    coords, mask, pair_edges, stack_edges, pair_orient = sample
    descriptor, _info = compute_descriptor(
        coords, mask, pair_edges, stack_edges, pair_orient=pair_orient)
    if norm_mean is not None and norm_std is not None:
        descriptor = (descriptor - norm_mean.to(device)) / norm_std.to(device)
    if active_blocks:
        keep = active_block_mask(active_blocks, descriptor.shape[-1],
                                 dtype=descriptor.dtype).to(device)
        descriptor = descriptor * keep
    indices = model(descriptor)
    valid_mask = mask[0].cpu()
    tokens = indices[0].cpu()[valid_mask].numpy().astype(np.int16)
    return {
        "id": entry.get("id", ""),
        "tokens": tokens,
        "coords": coords[0].cpu().numpy(),
        "mask": mask[0].cpu().numpy(),
        "n_valid": int(valid_mask.sum()),
    }

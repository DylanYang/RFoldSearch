import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np


class EMACodebook(nn.Module):
    def __init__(
        self,
        n_codes,
        embedding_dim,
        no_random_restart=False,
        restart_thres=1.0,
        ema_decay=0.99,
        commitment_weight=1.0,
        epsilon=1e-5,
        l2_normalize=True,
    ):
        super().__init__()
        init = torch.randn(n_codes, embedding_dim)
        if l2_normalize:
            init = F.normalize(init, dim=-1)
        self.register_buffer("embeddings", init)
        self.register_buffer("N", torch.zeros(n_codes))
        self.register_buffer("z_avg", self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres
        self.freeze_codebook = False
        self.ema_decay = ema_decay
        self.commitment_weight = commitment_weight
        self.epsilon = epsilon
        self.l2_normalize = l2_normalize

    def reset_parameters(self):
        pass

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # A loaded codebook already has learned embeddings / EMA statistics.
        # Without this guard, resuming training from a checkpoint would
        # re-initialise the codebook from the first new batch.
        self._need_init = False

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        self._need_init = False
        flat_inputs = z.reshape(-1, self.embedding_dim)
        y = self._tile(flat_inputs)

        _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        if self.l2_normalize:
            _k_rand = F.normalize(_k_rand, dim=-1)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        if self._need_init and self.training and not self.freeze_codebook:
            self._init_embeddings(z)

        flat_inputs = z.reshape(-1, self.embedding_dim)

        distances = (
            (flat_inputs**2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embeddings.t()
            + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = encoding_indices.view(*z.shape[:2])  # back to batch × seq_len

        embeddings = F.embedding(
            encoding_indices,
            self.embeddings,
        )  # each index replaced by its 64-dim code vector

        commitment_loss = self.commitment_weight * F.mse_loss(z, embeddings.detach())
        z_q_st = (embeddings - z).detach() + z

        if self.training and not self.freeze_codebook:
            with torch.no_grad():
                enc = encoding_indices.reshape(-1)

                acc_dtype = torch.float32
                n = torch.bincount(enc, minlength=self.n_codes).to(acc_dtype)

                z_sum = torch.zeros(
                    self.n_codes, self.embedding_dim, device=flat_inputs.device, dtype=acc_dtype
                )
                z_sum.index_add_(0, enc, flat_inputs.to(acc_dtype))

                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(n, op=dist.ReduceOp.SUM)
                    dist.all_reduce(z_sum, op=dist.ReduceOp.SUM)

                d = self.ema_decay
                self.N.mul_(d).add_(n, alpha=1.0 - d)
                self.z_avg.mul_(d).add_(z_sum, alpha=1.0 - d)

                if not self.no_random_restart:
                    dead = self.N < self.restart_thres
                    num_dead = int(dead.sum().item())
                    if num_dead > 0 and flat_inputs.shape[0] > 0:
                        perm = torch.randperm(flat_inputs.shape[0], device=flat_inputs.device)[
                            :num_dead
                        ]
                        self.z_avg[dead] = flat_inputs[perm].to(self.z_avg.dtype)
                        self.N[dead] = 1.0

                n_total = self.N.sum()
                smoothed_N = (
                    (self.N + self.epsilon) / (n_total + self.n_codes * self.epsilon) * n_total
                )
                denom = smoothed_N.unsqueeze(1)

                new_emb = (self.z_avg / denom).to(self.embeddings.dtype)
                if self.l2_normalize:
                    new_emb = F.normalize(new_emb, dim=-1)
                self.embeddings.copy_(new_emb)

        return z_q_st, encoding_indices, commitment_loss

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings

    def soft_codebook_lookup(self, weights: torch.Tensor) -> torch.Tensor:
        return weights @ self.embeddings

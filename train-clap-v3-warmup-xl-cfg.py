#!/usr/bin/env python

"""Train a CLAP-conditioned Continuous Autoregressive Model (CAM) (v3-xl).

This script trains a model that generates audio in a *compressed latent space* produced by
`music2latent`. Conditioning is provided by CLAP embeddings.

Key ideas (high level):
- **music2latent**: encodes audio waveforms into a sequence of latent frames.
- **CLAP tokens**: a sequence of embeddings that represent the conditioning signal.
- **CAM model**: an autoregressive Transformer backbone plus a sampler MLP used in a small
  diffusion-like inner loop.

Expected preprocessing output (`.pt` files):
- `music2latent`: Tensor shaped like (channels, latent_dim, T)
- `clap_tokens`: Tensor shaped like (num_segments, L, 512) or (L, 512) for older files
- `clap_mask` (optional): Tensor shaped like (num_segments, L)
"""

import argparse
from collections import OrderedDict
import copy
import glob
import os
import random
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from music2latent import EncoderDecoder


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper for RoPE: swap and negate half of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Positional Embeddings (RoPE) to a query/key tensor."""
    return (x * freqs.cos()) + (rotate_half(x) * freqs.sin())


def get_rotary_embedding(seq_len: int, dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Create the RoPE frequency matrix for a given sequence length and head dimension."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim))
    t = torch.arange(seq_len, dtype=dtype, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.unsqueeze(1)


class PreprocessedAudioDataset(Dataset):
    """Loads preprocessed `.pt` files and exposes fixed-length latent *segments*.

    The model trains on short chunks of latent frames (length=`segment_frames`). Each chunk
    is paired with the CLAP conditioning tokens that correspond to the same time span.
    """

    def __init__(self, preprocessed_dir, segment_frames=128, cache_size=2):
        self.files = glob.glob(os.path.join(preprocessed_dir, "*.pt"))
        if len(self.files) == 0:
            raise ValueError(f"No preprocessed files found in {os.path.abspath(preprocessed_dir)}")
        else:
            print(f"Found {len(self.files)} preprocessed files in {os.path.abspath(preprocessed_dir)}")

        self.segment_frames = segment_frames
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self.index_list: list[tuple[str, int]] = []
        for f in self.files:
            data = torch.load(f, map_location="cpu", weights_only=True)
            latent = data["music2latent"]
            T = int(latent.shape[2])
            num_segs = T // segment_frames
            for i in range(num_segs):
                start = i * segment_frames
                self.index_list.append((f, start))

        self.index_list = random.sample(self.index_list, len(self.index_list))
        print(f"Total segments available: {len(self.index_list)}")

    def __len__(self):
        return len(self.index_list)

    def _load_file(self, file_path: str) -> dict:
        if self.cache_size <= 0:
            return torch.load(file_path, map_location="cpu", weights_only=True)

        cached = self._cache.get(file_path)
        if cached is not None:
            self._cache.move_to_end(file_path)
            return cached

        data = torch.load(file_path, map_location="cpu", weights_only=True)
        self._cache[file_path] = data
        self._cache.move_to_end(file_path)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data

    def __getitem__(self, idx):
        """Return one training example.

        Returns:
        - latent segment: (T, latent_dim*channels)
        - clap_tokens: (L, 512)
        - clap_mask: (L,)
        """
        file_path, start = self.index_list[idx]
        data = self._load_file(file_path)
        latent = data["music2latent"]
        clap_tokens_all = data["clap_tokens"]
        clap_mask_all = data.get("clap_mask", None)

        latent = latent.permute(2, 0, 1)
        T, C, D = latent.shape
        if T < self.segment_frames:
            pad = self.segment_frames - T
            latent = F.pad(latent, (0, 0, 0, 0, 0, pad), mode="replicate")
        else:
            latent = latent[start : start + self.segment_frames, :, :]
        latent = latent.reshape(latent.shape[0], -1)

        seg_idx = int(start // self.segment_frames)
        if clap_tokens_all.dim() == 3:
            if seg_idx >= int(clap_tokens_all.shape[0]):
                raise ValueError(
                    f"Segment index {seg_idx} out of range for clap_tokens with shape {tuple(clap_tokens_all.shape)}"
                )
            clap_tokens = clap_tokens_all[seg_idx]
            if clap_mask_all is not None:
                if clap_mask_all.dim() != 2:
                    raise ValueError(f"Expected clap_mask shape (num_segments,L) but got {tuple(clap_mask_all.shape)}")
                if seg_idx >= int(clap_mask_all.shape[0]):
                    raise ValueError(
                        f"Segment index {seg_idx} out of range for clap_mask with shape {tuple(clap_mask_all.shape)}"
                    )
                clap_mask = clap_mask_all[seg_idx]
            else:
                clap_mask = torch.ones(clap_tokens.shape[0], dtype=torch.bool)
        elif clap_tokens_all.dim() == 2:
            clap_tokens = clap_tokens_all
            clap_mask = torch.ones(clap_tokens.shape[0], dtype=torch.bool)
        else:
            raise ValueError(f"Unexpected clap_tokens shape: {tuple(clap_tokens_all.shape)}")

        if clap_tokens.dim() != 2 or clap_tokens.shape[1] != 512:
            raise ValueError(f"Expected clap_tokens shape (L,512) but got {tuple(clap_tokens.shape)}")
        if clap_mask.dim() != 1 or clap_mask.shape[0] != clap_tokens.shape[0]:
            raise ValueError(
                f"Expected clap_mask shape (L,) matching tokens, got {tuple(clap_mask.shape)} for tokens {tuple(clap_tokens.shape)}"
            )

        return latent.float(), clap_tokens.float(), clap_mask.to(dtype=torch.bool)


def collate_batch(batch):
    """Pad variable-length CLAP token sequences into a dense batch.

    `latents` already have a fixed length (segment_frames). `clap_tokens` can have different
    lengths per example, so we pad to the maximum L in the batch and return a boolean mask.
    """
    latents, tokens_list, mask_list = zip(*batch)
    latents = torch.stack(latents, dim=0)

    max_L = max(int(t.shape[0]) for t in tokens_list)
    B = len(tokens_list)
    tokens = torch.zeros((B, max_L, 512), dtype=torch.float32)
    mask = torch.zeros((B, max_L), dtype=torch.bool)
    for i, (toks, m) in enumerate(zip(tokens_list, mask_list)):
        L = int(toks.shape[0])
        tokens[i, :L, :] = toks
        mask[i, :L] = m

    return latents, tokens, mask


class AdaLayerNorm(nn.Module):
    """LayerNorm with scale/bias conditioned on a scalar noise level `sigma`.

    This is a simple form of conditioning used by the sampler MLP.
    """

    def __init__(self, normalized_shape):
        super(AdaLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape)
        self.scale_proj = nn.Linear(1, normalized_shape)
        self.bias_proj = nn.Linear(1, normalized_shape)

    def forward(self, x, sigma):
        gamma = 1 + self.scale_proj(sigma)
        beta = self.bias_proj(sigma)
        return self.ln(x) * gamma + beta


class RotaryMultiheadAttention(nn.Module):
    """Multi-head self-attention with Rotary Positional Embeddings (RoPE)."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(RotaryMultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, query, key, value, attn_mask=None):
        seq_len, batch, _ = query.size()
        q = (
            query.view(seq_len, batch, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(seq_len, batch * self.num_heads, self.head_dim)
        )
        k = (
            key.view(seq_len, batch, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(seq_len, batch * self.num_heads, self.head_dim)
        )
        freqs = get_rotary_embedding(seq_len, self.head_dim, query.dtype, query.device)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        q = (
            q.view(seq_len, batch, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(seq_len, batch, self.embed_dim)
        )
        k = (
            k.view(seq_len, batch, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(seq_len, batch, self.embed_dim)
        )
        return self.attn(q, k, value, attn_mask=attn_mask)


class TransformerBlock(nn.Module):
    """A standard Transformer block: self-attention + feed-forward network."""

    def __init__(self, model_dim, num_heads, ff_mult=4, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.attn = RotaryMultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, model_dim * ff_mult),
            nn.GELU(),
            nn.Linear(model_dim * ff_mult, model_dim),
        )

    def forward(self, x, attn_mask):
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = residual + attn_out
        residual = x
        x = self.ln2(x)
        x = residual + self.ff(x)
        return x


class CrossAttention(nn.Module):
    """Cross-attention from latent sequence (queries) to conditioning tokens (keys/values)."""

    def __init__(self, query_dim, context_dim, num_heads, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, context_mask=None):
        """Attend from `x` to `context`.

        Inputs:
        - `x`: latent/hidden sequence (B, T, query_dim)
        - `context`: conditioning tokens (B, L, context_dim)
        - `context_mask`: which conditioning tokens are valid (B, L)

        Returns:
        - attended values projected back to (B, T, query_dim)
        """
        B, T, _ = x.shape
        L = context.shape[1]
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if context_mask is not None:
            m = context_mask.to(dtype=torch.bool, device=attn_scores.device)
            m = m.view(B, 1, 1, L)
            attn_scores = attn_scores.masked_fill(~m, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(attn_out)


class Backbone(nn.Module):
    """Autoregressive Transformer backbone operating on latent frames."""

    def __init__(self, latent_dim=128, model_dim=768, num_layers=16, num_heads=4, max_seq_len=512):
        super(Backbone, self).__init__()
        self.input_proj = nn.Linear(latent_dim, model_dim)
        self.layers = nn.ModuleList([TransformerBlock(model_dim, num_heads, ff_mult=4) for _ in range(num_layers)])
        self.model_dim = model_dim

    def forward(self, x):
        """Run a causal Transformer over the latent sequence.

        `x` is assumed to be a sequence of latent frames, and the causal mask ensures each
        position only attends to earlier positions.
        """
        B, T, _ = x.shape
        x = self.input_proj(x)
        x = x.transpose(0, 1)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
        x = x.transpose(0, 1)
        return x


class SamplerMLP(nn.Module):
    """MLP used inside the diffusion-like sampler.

    Inputs are concatenated features (history/state + current noisy latent + conditioning).
    AdaLayerNorm layers are conditioned on the noise level `sigma`.
    """

    def __init__(self, cond_dim, latent_dim=128, hidden_dim=768, num_layers=8, mlp_mult=4, dropout=0.0):
        super(SamplerMLP, self).__init__()
        self.input_dim = cond_dim + latent_dim
        self.hidden_dim = hidden_dim
        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(AdaLayerNorm(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(AdaLayerNorm(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, sigma):
        """Predict denoising updates given features `x` and noise level `sigma`."""
        B, T, _ = x.shape
        x = x.view(B * T, -1)
        sigma = sigma.view(B * T, 1)
        for layer in self.layers:
            if isinstance(layer, AdaLayerNorm):
                x = layer(x, sigma)
            else:
                x = layer(x)
        x = x.view(B, T, -1)
        return x


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Mean-pool a sequence with a boolean mask.

    This is used to convert variable-length CLAP token sequences into a single vector for
    conditioning the sampler.
    """
    m = mask.to(dtype=x.dtype)
    num = (x * m.unsqueeze(-1)).sum(dim=dim)
    den = m.sum(dim=dim).clamp_min(eps).unsqueeze(-1)
    return num / den


class CAMModel(nn.Module):
    """Continuous Autoregressive Model (CAM) conditioned on CLAP token sequences.

    Components:
    - **backbone**: autoregressive Transformer over latent frames.
    - **cross_attn**: injects CLAP conditioning into the backbone activations.
    - **sampler**: predicts denoising updates used by the diffusion-like inner loop.

    Training uses classifier-free guidance style dropout by sometimes replacing conditioning
    tokens with a learned `null_clap` embedding.
    """

    def __init__(self, latent_dim=64, model_dim=768, seq_len=128, channels=2, clap_dim=512, cfg_drop_prob=0.1):
        super(CAMModel, self).__init__()
        self.original_latent_dim = latent_dim
        self.channels = channels
        self.latent_dim = latent_dim * channels
        self.model_dim = model_dim
        self.seq_len = seq_len
        self.cfg_drop_prob = cfg_drop_prob

        self.null_clap = nn.Parameter(torch.empty(clap_dim))
        nn.init.xavier_uniform_(self.null_clap.unsqueeze(0))

        self.backbone = Backbone(latent_dim=self.latent_dim, model_dim=model_dim, num_layers=24, num_heads=6, max_seq_len=seq_len)
        self.clap_proj = nn.Linear(clap_dim, model_dim)
        self.cross_attn = CrossAttention(query_dim=model_dim, context_dim=model_dim, num_heads=4)
        self.sampler = SamplerMLP(cond_dim=2 * model_dim, latent_dim=self.latent_dim, hidden_dim=model_dim, num_layers=12)
        self.zSOS = nn.Parameter(torch.randn(1, model_dim).squeeze(0))

    def forward(self, x, clap_tokens, clap_mask):
        """Compute the training loss for a batch.

        Inputs:
        - `x`: clean latent targets (B, T, D)
        - `clap_tokens`: conditioning token sequence (B, L, 512)
        - `clap_mask`: which conditioning tokens are valid (B, L)
        """
        B, T, D = x.shape
        device = x.device

        if clap_tokens.dim() != 3:
            if torch.jit.is_scripting():
                raise RuntimeError("clap_tokens must be (B,L,512)")
            raise ValueError(f"clap_tokens must be (B,L,512), got {tuple(clap_tokens.shape)}")
        if clap_mask.dim() != 2:
            if torch.jit.is_scripting():
                raise RuntimeError("clap_mask must be (B,L)")
            raise ValueError(f"clap_mask must be (B,L), got {tuple(clap_mask.shape)}")

        clap_tokens = clap_tokens.to(device=device, dtype=torch.float32)
        clap_mask = clap_mask.to(device=device, dtype=torch.bool)

        # Classifier-Free Guidance (CFG) dropout during training:
        # randomly replace the conditioning tokens with a learned "null" token so the
        # model learns both conditional and unconditional behavior.
        if self.training and self.cfg_drop_prob > 0:
            drop_cond = torch.rand(B, device=device) < self.cfg_drop_prob
            if drop_cond.any():
                clap_tokens = clap_tokens.clone()
                clap_mask = clap_mask.clone()
                null_tok = self.null_clap.to(device=device, dtype=torch.float32)
                clap_tokens[drop_cond, :, :] = 0.0
                clap_tokens[drop_cond, 0, :] = null_tok
                clap_mask[drop_cond, :] = False
                clap_mask[drop_cond, 0] = True

        # Mix clean latents `x` with Gaussian noise to create an input for the backbone.
        k = torch.rand(B, T, 1, device=device)
        eps_backbone = torch.randn(B, T, D, device=device)
        x_tilde = k * eps_backbone + (1 - k) * x

        # Backbone predicts next-step hidden states; cross-attention injects CLAP context.
        backbone_out = self.backbone(x_tilde)

        ctx = self.clap_proj(clap_tokens)
        cond_out = self.cross_attn(backbone_out, ctx, context_mask=clap_mask)
        backbone_out = backbone_out + cond_out

        # Build a shifted context `z` so each position sees the previous hidden state.
        z = torch.zeros(B, T, self.model_dim, device=device)
        z[:, 0, :] = self.zSOS
        if T > 1:
            z[:, 1:, :] = backbone_out[:, :-1, :]

        # Second noise-mixing stage used by the sampler objective.
        sigma = torch.sigmoid(torch.randn(B, T, 1, device=device))
        eps_sampler = torch.randn(B, T, D, device=device)
        y = sigma * eps_sampler + (1 - sigma) * x
        v_target = x - eps_sampler

        # Drop some history context to make the model robust.
        drop_mask = (torch.rand(B, T, 1, device=device) < 0.2).float()
        z = drop_mask * self.zSOS.unsqueeze(0).unsqueeze(0) + (1 - drop_mask) * z

        # For the sampler, we pool the CLAP tokens to a single conditioning vector.
        pooled = masked_mean(ctx, clap_mask, dim=1)
        cond_for_sampler = pooled.unsqueeze(1).expand(-1, T, -1)
        sampler_input = torch.cat([z, y, cond_for_sampler], dim=-1)
        v_pred = self.sampler(sampler_input, sigma)
        loss = F.mse_loss(v_pred, v_target)
        return loss


def generate_sample(model, device, seq_len, num_diffusion_steps=48, seed_scale=0.0, temperature=1.0, clap_tokens=None, clap_mask=None, cfg_scale=1.0):
    """Generate a latent sequence (and later decode to audio) conditioned on CLAP.

    This runs a small diffusion-like inner loop for each latent frame, and then advances
    autoregressively using the Transformer backbone.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            generated = []

            if clap_tokens is None:
                clap_tokens = model.null_clap.view(1, 1, -1)
                clap_mask = torch.ones((1, 1), dtype=torch.bool)
            else:
                if clap_tokens.dim() == 2:
                    clap_tokens = clap_tokens.unsqueeze(0)
                if clap_mask is None:
                    clap_mask = torch.ones((clap_tokens.shape[0], clap_tokens.shape[1]), dtype=torch.bool)

            clap_tokens = clap_tokens.to(device=device, dtype=torch.float32)
            clap_mask = clap_mask.to(device=device, dtype=torch.bool)

            cond_ctx = model.clap_proj(clap_tokens)
            cond_pool = masked_mean(cond_ctx, clap_mask, dim=1).unsqueeze(1)

            uncond_tokens = model.null_clap.view(1, 1, -1).to(device=device, dtype=torch.float32)
            uncond_mask = torch.ones((1, 1), device=device, dtype=torch.bool)
            uncond_ctx = model.clap_proj(uncond_tokens)
            uncond_pool = masked_mean(uncond_ctx, uncond_mask, dim=1).unsqueeze(1)

            z0 = model.zSOS.unsqueeze(0).unsqueeze(0)
            if seed_scale:
                z0 = z0 + seed_scale * torch.randn_like(z0)

            # Generate the first latent frame with a diffusion-like inner loop.
            x = torch.randn(1, 1, model.latent_dim, device=device)
            noise_level = torch.ones(1, 1, 1, device=device)

            for i in range(num_diffusion_steps):
                step_size = 1.0 / num_diffusion_steps
                inp_cond = torch.cat([z0, x, cond_pool], dim=-1)
                eps_cond = model.sampler(inp_cond, noise_level)
                inp_uncond = torch.cat([z0, x, uncond_pool], dim=-1)
                eps_uncond = model.sampler(inp_uncond, noise_level)
                eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
                x = x + step_size * (eps / temperature)
                noise_level = noise_level - step_size
            generated.append(x.squeeze(0))

            for t in range(1, seq_len):
                # Autoregressive step: encode previous frames, then sample the next one.
                prev_seq = torch.stack(generated, dim=1)
                backbone_out = model.backbone(prev_seq)
                backbone_out_cond = backbone_out + model.cross_attn(backbone_out, cond_ctx, context_mask=clap_mask)
                backbone_out_uncond = backbone_out + model.cross_attn(backbone_out, uncond_ctx, context_mask=uncond_mask)
                z_t_cond = backbone_out_cond[:, -1:, :]
                z_t_uncond = backbone_out_uncond[:, -1:, :]

                x = torch.randn(1, 1, model.latent_dim, device=device)
                noise_level = torch.ones(1, 1, 1, device=device)
                for i in range(num_diffusion_steps):
                    step_size = 1.0 / num_diffusion_steps
                    inp_cond = torch.cat([z_t_cond, x, cond_pool], dim=-1)
                    eps_cond = model.sampler(inp_cond, noise_level)
                    inp_uncond = torch.cat([z_t_uncond, x, uncond_pool], dim=-1)
                    eps_uncond = model.sampler(inp_uncond, noise_level)
                    eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
                    x = x + step_size * (eps / temperature)
                    noise_level = noise_level - step_size
                generated.append(x.squeeze(0))

            gen_seq = torch.stack(generated, dim=0)
            gen_seq = gen_seq.view(seq_len, model.channels, model.original_latent_dim)
            generated_latents = gen_seq.permute(1, 2, 0)
            return generated_latents
    finally:
        if was_training:
            model.train()


def train(args):
    """Train the v3-xl CAM model with multi-token CLAP conditioning.

    This training script assumes the dataset has already been preprocessed into `.pt` files.
    Each `.pt` contains:
    - `music2latent`: compressed audio latents (what the model learns to model/generate)
    - `clap_tokens` (+ optional `clap_mask`): CLAP conditioning tokens aligned to each latent segment

    High-level flow:
    - Pick a device (CUDA/MPS/CPU)
    - Build a dataset of fixed-length latent segments (`segment_frames`)
    - Train with AdamW + cosine LR schedule
    - Optionally maintain an EMA (exponential moving average) copy of the weights
    - Periodically save a checkpoint and a decoded `.wav` sample into `ckpt_dir`
    """
    # Select a compute device. Prefer GPU when available.
    if getattr(args, "device", None) is None or args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Checkpoints and generated samples will be written here.
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print("Initializing preprocessed dataset...")
    # The dataset indexes *segments* (not full songs). We then split segments into train/val.
    full_dataset = PreprocessedAudioDataset(
        preprocessed_dir=args.preprocessed_dir,
        segment_frames=args.segment_frames,
        cache_size=getattr(args, "dataset_cache_size", 2),
    )

    # Split by track (file) to avoid leakage between train/val.
    file_to_segment_indices: dict[str, list[int]] = {}
    for seg_ds_idx, (fp, _) in enumerate(full_dataset.index_list):
        file_to_segment_indices.setdefault(fp, []).append(seg_ds_idx)

    all_files = list(file_to_segment_indices.keys())
    random.shuffle(all_files)
    num_files = len(all_files)
    num_val_files = int(args.val_ratio * num_files)
    num_val_files = max(0, min(num_files, num_val_files))
    val_files = set(all_files[:num_val_files])
    train_files = set(all_files[num_val_files:])

    train_indices: list[int] = []
    val_indices: list[int] = []
    for fp, seg_idxs in file_to_segment_indices.items():
        if fp in val_files:
            val_indices.extend(seg_idxs)
        else:
            train_indices.extend(seg_idxs)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    print(
        f"Train tracks: {len(train_files)}, Validation tracks: {len(val_files)} | "
        f"Train segments: {len(train_dataset)}, Validation segments: {len(val_dataset)}"
    )

    num_workers = getattr(args, "num_workers", 4)
    if device.type == "mps" and getattr(args, "num_workers", None) is None:
        num_workers = 0

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_batch,
    )

    # Core model. It predicts denoising updates in latent space, conditioned on CLAP tokens.
    model = CAMModel(latent_dim=64, model_dim=768, seq_len=args.segment_frames, channels=2, clap_dim=512, cfg_drop_prob=args.cfg_drop_prob)
    model.to(device)

    ema_model = None
    if getattr(args, "use_ema", False):
        # EMA keeps a smoothed copy of weights that often sounds better at inference.
        ema_model = copy.deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad = False

    # Optimizer + LR schedule.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.lr_decay_steps), eta_min=args.lr_min)

    start_step = 0
    global_step = 0

    if args.resume is not None:
        # Resume model/optimizer/scheduler (and EMA if available) from a checkpoint.
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        if isinstance(ckpt, dict):
            if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                for st in optimizer.state.values():
                    for k, v in st.items():
                        if isinstance(v, torch.Tensor):
                            st[k] = v.to(device)
            if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            global_step = int(ckpt.get("global_step", 0))
            start_step = global_step

            if ema_model is not None:
                ema_sd = ckpt.get("ema_state_dict", None)
                if ema_sd is not None:
                    ema_model.load_state_dict(ema_sd, strict=False)
                else:
                    ema_model.load_state_dict(model.state_dict(), strict=False)

        print(f"Resumed at global_step={global_step}")

    encoder = EncoderDecoder()

    for epoch in range(args.num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for latents, clap_tokens, clap_mask in pbar:
            # Move the current minibatch to the training device.
            latents = latents.to(device)
            clap_tokens = clap_tokens.to(device)
            clap_mask = clap_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            # Forward pass returns the diffusion training loss.
            loss = model(latents, clap_tokens, clap_mask)
            loss.backward()
            # Gradient clipping is optional (set --grad_clip 0 to disable).
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            scheduler.step()

            global_step += 1

            if ema_model is not None and global_step >= args.ema_update_after_step and (global_step % args.ema_update_every == 0):
                # EMA update: ema = decay * ema + (1 - decay) * current
                decay = args.ema_decay
                msd = model.state_dict()
                esd = ema_model.state_dict()
                for k, v in msd.items():
                    if k not in esd:
                        continue
                    if not torch.is_floating_point(v):
                        continue
                    esd[k].mul_(decay).add_(v.detach(), alpha=(1.0 - decay))

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}", "step": global_step})

            if global_step % args.ckpt_every == 0:
                # Save training state.
                ckpt_path = os.path.join(args.ckpt_dir, f"cam_step_{global_step}.pt")
                out = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                }
                if ema_model is not None:
                    out["ema_state_dict"] = ema_model.state_dict()
                torch.save(out, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")
                sys.stdout.flush()

                if len(val_dataset) > 0:
                    # Quick qualitative check: generate a sample conditioned on a random validation segment.
                    # The sample is decoded back to waveform using music2latent.
                    random_idx = random.choice(val_dataset.indices)
                    file_path, start = full_dataset.index_list[random_idx]
                    data = full_dataset._load_file(file_path)
                    clap_tokens_all = data["clap_tokens"].float()
                    clap_mask_all = data.get("clap_mask", None)
                    if clap_mask_all is not None:
                        clap_mask_all = clap_mask_all.to(dtype=torch.bool)

                    seg_idx = int(start // args.segment_frames)
                    if clap_tokens_all.dim() == 3:
                        if seg_idx >= int(clap_tokens_all.shape[0]):
                            raise ValueError(
                                f"Segment index {seg_idx} out of range for clap_tokens with shape {tuple(clap_tokens_all.shape)}"
                            )
                        clap_tokens_val = clap_tokens_all[seg_idx].unsqueeze(0).to(device)
                        if clap_mask_all is not None:
                            clap_mask_val = clap_mask_all[seg_idx].unsqueeze(0).to(device)
                        else:
                            clap_mask_val = torch.ones((1, clap_tokens_val.shape[1]), device=device, dtype=torch.bool)
                    elif clap_tokens_all.dim() == 2:
                        clap_tokens_val = clap_tokens_all.unsqueeze(0).to(device)
                        clap_mask_val = torch.ones((1, clap_tokens_val.shape[1]), device=device, dtype=torch.bool)
                    else:
                        raise ValueError(f"Unexpected clap_tokens shape in val file: {tuple(clap_tokens_all.shape)}")

                    active_model = ema_model if ema_model is not None else model
                    sample_latents = generate_sample(
                        active_model,
                        device,
                        seq_len=args.sample_seq_len,
                        num_diffusion_steps=args.num_diffusion_steps,
                        seed_scale=args.seed_scale,
                        temperature=args.temperature,
                        clap_tokens=clap_tokens_val,
                        clap_mask=clap_mask_val,
                        cfg_scale=args.cfg_scale,
                    )
                    if isinstance(sample_latents, torch.Tensor):
                        sample_latents = sample_latents.detach().to("cpu")

                    max_waveform_length = int(args.sample_seq_len * args.samples_per_latent)
                    audio_ch = encoder.decode(sample_latents, max_waveform_length=max_waveform_length)
                    if isinstance(audio_ch, torch.Tensor):
                        audio_ch = audio_ch.detach().cpu().numpy()
                    if audio_ch.shape[0] < audio_ch.shape[1]:
                        audio_ch = audio_ch.T

                    # Save sample next to checkpoints for easy tracking.
                    sample_path = os.path.join(args.ckpt_dir, f"sample_step_{global_step}.wav")
                    sf.write(sample_path, audio_ch, args.sample_rate)
                    print(f"Saved sample audio: {sample_path}")

        if global_step >= args.max_steps:
            break

    final_path = os.path.join(args.ckpt_dir, "cam_final.pt")
    out = {"model_state_dict": model.state_dict(), "global_step": global_step}
    if ema_model is not None:
        out["ema_state_dict"] = ema_model.state_dict()
    torch.save(out, final_path)
    print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    # CLI entrypoint.
    parser = argparse.ArgumentParser(description="Train v3-xl CAM with multi-token CLAP conditioning.")
    parser.add_argument("--preprocessed_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--sample_dir", type=str, default="./train_samples")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--sample_rate", type=int, default=48_000)
    parser.add_argument("--samples_per_latent", type=int, default=4096)

    parser.add_argument("--segment_frames", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100_000)
    parser.add_argument("--max_steps", type=int, default=100_000_000)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=5e-5)
    parser.add_argument("--lr_decay_steps", type=int, default=100_000_000)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--dataset_cache_size", type=int, default=2)

    parser.add_argument("--cfg_drop_prob", type=float, default=0.1)
    parser.add_argument("--cfg_scale", type=float, default=0.0)

    parser.add_argument("--num_diffusion_steps", type=int, default=48)
    parser.add_argument("--seed_scale", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--sample_seq_len", type=int, default=256)

    parser.add_argument("--ckpt_every", type=int, default=2000)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_update_every", type=int, default=1)
    parser.add_argument("--ema_update_after_step", type=int, default=0)

    parser.add_argument("--num_workers", type=int, default=None)

    args = parser.parse_args()
    train(args)

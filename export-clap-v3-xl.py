#!/usr/bin/env python

import argparse
import importlib.util
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class CAMExportModule(nn.Module):
    def __init__(self, cam: nn.Module):
        super().__init__()
        self.cam = cam

    @torch.jit.export
    def project_clap(self, clap_tokens: torch.Tensor) -> torch.Tensor:
        return self.cam.clap_proj(clap_tokens)

    @torch.jit.export
    def pool_clap(self, ctx: torch.Tensor, clap_mask: torch.Tensor) -> torch.Tensor:
        m = clap_mask.to(dtype=ctx.dtype)
        num = (ctx * m.unsqueeze(-1)).sum(dim=1)
        den = m.sum(dim=1).clamp_min(1e-8).unsqueeze(-1)
        return num / den

    @torch.jit.export
    def backbone(self, prev_seq: torch.Tensor) -> torch.Tensor:
        return self.cam.backbone(prev_seq)

    @torch.jit.export
    def cross_attention(self, backbone_out: torch.Tensor, ctx: torch.Tensor, clap_mask: torch.Tensor) -> torch.Tensor:
        return self.cam.cross_attn(backbone_out, ctx, clap_mask)

    @torch.jit.export
    def sampler_step(self, z: torch.Tensor, x: torch.Tensor, pooled: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, x, pooled], dim=-1)
        return self.cam.sampler(inp, sigma)

    @torch.jit.export
    def get_null_clap(self) -> torch.Tensor:
        return self.cam.null_clap

    @torch.jit.export
    def get_zsos(self) -> torch.Tensor:
        return self.cam.zSOS

    def forward(self, prev_seq: torch.Tensor, clap_tokens: torch.Tensor, clap_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if clap_tokens.dim() == 2:
            clap_tokens = clap_tokens.unsqueeze(0)
        if clap_mask.dim() == 1:
            clap_mask = clap_mask.unsqueeze(0)

        ctx = self.project_clap(clap_tokens)
        pooled = self.pool_clap(ctx, clap_mask).unsqueeze(1)

        backbone_out = self.backbone(prev_seq)
        cond_out = self.cross_attention(backbone_out, ctx, clap_mask)
        backbone_out = backbone_out + cond_out
        z_t = backbone_out[:, -1:, :]
        return z_t, pooled


class CAMGenerateTSModule(nn.Module):
    def __init__(self, cam: nn.Module, latent_dim: int, channels: int, clap_dim: int):
        super().__init__()
        self.cam = cam
        self.latent_dim = int(latent_dim)
        self.channels = int(channels)
        self.clap_dim = int(clap_dim)

    @torch.jit.export
    def _pool_ctx(self, ctx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(dtype=ctx.dtype)
        num = (ctx * m.unsqueeze(-1)).sum(dim=1)
        den = m.sum(dim=1).clamp_min(1e-8).unsqueeze(-1)
        return (num / den).unsqueeze(1)

    @torch.jit.export
    def _diffuse(
        self,
        z_cond: torch.Tensor,
        z_uncond: torch.Tensor,
        cond_pool: torch.Tensor,
        uncond_pool: torch.Tensor,
        cfg_scale: float,
        num_diffusion_steps: int,
        temperature: float,
        d_flat: int,
    ) -> torch.Tensor:
        device = z_cond.device
        x = torch.randn(1, 1, int(d_flat), device=device, dtype=torch.float32)
        noise_level = torch.ones(1, 1, 1, device=device, dtype=torch.float32)
        nsteps = int(max(1, num_diffusion_steps))
        for _ in range(nsteps):
            step_size = 1.0 / float(nsteps)
            inp_cond = torch.cat([z_cond, x, cond_pool], dim=-1)
            eps_cond = self.cam.sampler(inp_cond, noise_level)
            inp_uncond = torch.cat([z_uncond, x, uncond_pool], dim=-1)
            eps_uncond = self.cam.sampler(inp_uncond, noise_level)
            eps = eps_uncond + float(cfg_scale) * (eps_cond - eps_uncond)
            x = x + step_size * (eps / float(temperature))
            noise_level = noise_level - step_size
        return x

    @torch.jit.export
    def forward(
        self,
        seq_len: int,
        clap_tokens: torch.Tensor,
        clap_mask: torch.Tensor,
        cfg_scale: float,
        num_diffusion_steps: int,
        seed_scale: float,
        temperature: float,
        prefix_latents: Optional[torch.Tensor],
        use_conditioning: bool,
    ) -> torch.Tensor:
        device = clap_tokens.device
        d_flat = int(self.latent_dim) * int(self.channels)

        if not use_conditioning:
            clap_tokens = self.cam.null_clap.view(1, 1, -1).to(device=device, dtype=torch.float32)
            clap_mask = torch.ones((1, 1), device=device, dtype=torch.bool)
            cfg_scale = 0.0
        else:
            if clap_tokens.dim() == 2:
                clap_tokens = clap_tokens.unsqueeze(0)
            if clap_mask.dim() == 1:
                clap_mask = clap_mask.unsqueeze(0)
            clap_tokens = clap_tokens.to(device=device, dtype=torch.float32)
            clap_mask = clap_mask.to(device=device, dtype=torch.bool)

        cond_ctx = self.cam.clap_proj(clap_tokens)
        cond_pool = self._pool_ctx(cond_ctx, clap_mask)

        uncond_tokens = self.cam.null_clap.view(1, 1, -1).to(device=device, dtype=torch.float32)
        uncond_mask = torch.ones((1, 1), device=device, dtype=torch.bool)
        uncond_ctx = self.cam.clap_proj(uncond_tokens)
        uncond_pool = uncond_ctx.mean(dim=1, keepdim=True)

        generated = torch.jit.annotate(List[torch.Tensor], [])

        start_t = 0
        if prefix_latents is not None:
            pl = prefix_latents.to(device=device, dtype=torch.float32)
            if pl.dim() != 3:
                raise RuntimeError("prefix_latents must be (C,D,T)")
            T0 = int(pl.shape[2])
            prefix_frames = pl.permute(2, 0, 1).reshape(T0, d_flat)
            max_t = int(min(T0, seq_len))
            for i in range(max_t):
                generated.append(prefix_frames[i : i + 1, :])
            start_t = len(generated)

        z0 = self.cam.zSOS.to(device=device, dtype=torch.float32).view(1, 1, -1)
        if seed_scale != 0.0:
            z0 = z0 + float(seed_scale) * torch.randn_like(z0)

        if start_t == 0:
            x0 = self._diffuse(
                z0,
                z0,
                cond_pool,
                uncond_pool,
                float(cfg_scale),
                int(num_diffusion_steps),
                float(temperature),
                int(d_flat),
            )
            generated.append(x0.squeeze(0))
            start_t = 1

        for _t in range(start_t, int(seq_len)):
            prev_seq = torch.stack(generated, dim=1)
            backbone_out = self.cam.backbone(prev_seq)
            backbone_out_cond = backbone_out + self.cam.cross_attn(backbone_out, cond_ctx, clap_mask)
            backbone_out_uncond = backbone_out + self.cam.cross_attn(backbone_out, uncond_ctx, uncond_mask)
            z_t_cond = backbone_out_cond[:, -1:, :]
            z_t_uncond = backbone_out_uncond[:, -1:, :]
            x_t = self._diffuse(
                z_t_cond,
                z_t_uncond,
                cond_pool,
                uncond_pool,
                float(cfg_scale),
                int(num_diffusion_steps),
                float(temperature),
                int(d_flat),
            )
            generated.append(x_t.squeeze(0))

        gen_seq = torch.stack(generated, dim=1)
        gen_seq = gen_seq.squeeze(0)
        gen_seq = gen_seq.view(int(seq_len), int(self.channels), int(self.latent_dim))
        return gen_seq.permute(1, 2, 0).contiguous()


def _load_train_module(train_script_path: str):
    spec = importlib.util.spec_from_file_location("cam_train_v3_xl", train_script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load training script as module: {train_script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_state_dict(checkpoint_obj: object, use_ema: bool) -> dict:
    if isinstance(checkpoint_obj, dict):
        if use_ema and checkpoint_obj.get("ema_state_dict", None) is not None:
            return checkpoint_obj["ema_state_dict"]
        if checkpoint_obj.get("model_state_dict", None) is not None:
            return checkpoint_obj["model_state_dict"]
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    raise ValueError("Unsupported checkpoint format")


def main(
    checkpoint_path: str,
    out_path: str,
    train_script_path: str,
    use_ema: bool,
    export_method: str,
    trace_seq_len: int,
    export_kind: str,
    segment_frames: int,
    latent_dim: int,
    channels: int,
    model_dim: int,
    clap_dim: int,
    cfg_drop_prob: float,
    strict: bool,
):
    train_mod = _load_train_module(train_script_path)
    if not hasattr(train_mod, "CAMModel"):
        raise RuntimeError("Training script does not define CAMModel")

    cam_model_cls = getattr(train_mod, "CAMModel")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _resolve_state_dict(ckpt, use_ema=use_ema)

    cam = cam_model_cls(
        latent_dim=int(latent_dim),
        model_dim=int(model_dim),
        seq_len=int(segment_frames),
        channels=int(channels),
        clap_dim=int(clap_dim),
        cfg_drop_prob=float(cfg_drop_prob),
    )
    missing, unexpected = cam.load_state_dict(state_dict, strict=bool(strict))
    if len(missing) > 0:
        print("Missing keys:", missing)
    if len(unexpected) > 0:
        print("Unexpected keys:", unexpected)

    cam.eval()

    kind = str(export_kind).lower()
    if kind not in {"components", "generator"}:
        raise ValueError("export_kind must be one of: components, generator")

    if kind == "generator":
        export_module = CAMGenerateTSModule(cam, latent_dim=int(latent_dim), channels=int(channels), clap_dim=int(clap_dim))
    else:
        export_module = CAMExportModule(cam)
    export_module.eval()

    method = str(export_method).lower()
    if method not in {"auto", "script", "trace"}:
        raise ValueError("export_method must be one of: auto, script, trace")

    if method in {"auto", "script"}:
        try:
            scripted = torch.jit.script(export_module)
            scripted = torch.jit.freeze(scripted)
        except Exception as e:
            raise RuntimeError(
                "TorchScript scripting failed. For variable-length inference you must fix scripting. "
                "If you only need fixed-length inference you can use --export_method trace. "
                f"Error: {e}"
            )
    else:
        prev_len = int(max(1, trace_seq_len))
        prev_seq = torch.zeros((1, prev_len, int(latent_dim) * int(channels)), dtype=torch.float32)
        clap_tokens = torch.zeros((1, 2, int(clap_dim)), dtype=torch.float32)
        clap_mask = torch.ones((1, 2), dtype=torch.bool)
        scripted = torch.jit.trace(export_module, (prev_seq, clap_tokens, clap_mask))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    scripted.save(out_path)
    print(f"Saved TorchScript module to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--use_ema", action="store_true")

    parser.add_argument("--export_method", type=str, choices=["auto", "script", "trace"], default="auto")
    parser.add_argument("--trace_seq_len", type=int, default=1)

    parser.add_argument("--export_kind", type=str, choices=["components", "generator"], default="components")

    parser.add_argument("--segment_frames", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--model_dim", type=int, default=768)
    parser.add_argument("--clap_dim", type=int, default=512)
    parser.add_argument("--cfg_drop_prob", type=float, default=0.0)

    parser.add_argument("--strict", action="store_true")

    default_train_script = os.path.join(os.path.dirname(__file__), "train-clap-v3-warmup-xl-cfg.py")
    parser.add_argument("--train_script_path", type=str, default=default_train_script)

    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        train_script_path=args.train_script_path,
        use_ema=bool(args.use_ema),
        export_method=str(args.export_method),
        trace_seq_len=int(args.trace_seq_len),
        export_kind=str(args.export_kind),
        segment_frames=int(args.segment_frames),
        latent_dim=int(args.latent_dim),
        channels=int(args.channels),
        model_dim=int(args.model_dim),
        clap_dim=int(args.clap_dim),
        cfg_drop_prob=float(args.cfg_drop_prob),
        strict=bool(args.strict),
    )

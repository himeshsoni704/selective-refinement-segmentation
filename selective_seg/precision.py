"""Explicit specialist precision shared by training, validation and inference."""
import torch


def specialist_precision(device, cfg):
    requested = cfg.get("hybrid", {}).get("precision", "auto")
    if requested not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("Unknown specialist precision")
    if device.type != "cuda":
        if requested in {"fp16", "bf16"}:
            raise ValueError("Explicit mixed precision requires CUDA")
        return "fp32"
    if requested == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is not supported on this GPU")
    return requested


def specialist_amp(device, cfg):
    mode = specialist_precision(device, cfg)
    return torch.autocast(device_type=device.type, dtype=(torch.bfloat16 if mode == "bf16"
                          else torch.float16), enabled=mode != "fp32")


def require_finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite {name}: {int((~torch.isfinite(value)).sum())} values")
    return value

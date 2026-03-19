"""lm-evaluation-harness model wrapper with D-Flux governor profiles.

Usage:
    lm_eval --include_path src/dflux \
        --model dflux-governed \
        --model_args pretrained=Qwen/Qwen3.5-9B-Base,profile_path=profiles/reasoning.json,dtype=bfloat16 \
        --tasks hellaswag \
        --batch_size 1
"""

import json
import gc
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from dflux.multiscale_telemetry import MultiScaleTelemetry
from dflux.profile import load_profile, compute_profile


def _install_static_hooks(model: nn.Module, scales: Dict[int, float]):
    """Install lightweight forward hooks on attention o_proj layers.

    No telemetry, no governor — just scale multiplication.
    Returns list of hook handles for cleanup.
    """
    layers = MultiScaleTelemetry._find_transformer_layers(model)
    device = next(model.parameters()).device
    hooks = []

    for i, layer in enumerate(layers):
        scale_val = scales.get(i, 1.0)
        if abs(scale_val - 1.0) < 1e-6:
            continue

        attn = MultiScaleTelemetry._find_attn_module(layer)
        if attn is None:
            continue

        o_proj = None
        for name in ("o_proj", "c_proj", "dense", "out_proj"):
            if hasattr(attn, name):
                o_proj = getattr(attn, name)
                break
        if o_proj is None:
            continue

        scale_tensor = torch.tensor([scale_val], device=device, dtype=torch.float32)

        def _make_hook(s):
            def hook(module, input, output):
                return output * s.to(output.dtype)
            return hook

        h = o_proj.register_forward_hook(_make_hook(scale_tensor))
        hooks.append(h)

    return hooks


def _promote_qwen35_config(config):
    """Copy text_config fields to top-level config for Qwen3.5."""
    _SKIP = {"use_return_dict", "output_hidden_states", "output_attentions",
             "torchscript", "pruned_heads", "is_encoder_decoder"}
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is None:
        return
    for attr in dir(text_cfg):
        if attr.startswith("_") or attr in _SKIP:
            continue
        try:
            object.__getattribute__(config, attr)
        except AttributeError:
            try:
                val = getattr(text_cfg, attr)
                if not callable(val):
                    setattr(config, attr, val)
            except Exception:
                pass


def _detect_nested_keys(model_name: str) -> bool:
    """Check if a HF model uses nested language_model.* weight keys."""
    from huggingface_hub import hf_hub_download
    try:
        idx_path = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(idx_path) as f:
            keys = json.load(f)["weight_map"].keys()
        return any("language_model." in k for k in keys)
    except Exception:
        pass
    try:
        from safetensors import safe_open
        path = hf_hub_download(model_name, "model.safetensors")
        with safe_open(path, framework="pt") as f:
            keys = f.keys()
        return any("language_model." in k for k in keys)
    except Exception:
        return False


def _load_qwen35_manual(model_name, config, torch_dtype, device):
    """Load Qwen3.5 with manual weight key remapping."""
    from transformers import Qwen3_5ForCausalLM
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as safe_load

    _promote_qwen35_config(config)
    model = Qwen3_5ForCausalLM(config).to(dtype=torch_dtype)

    try:
        idx = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(idx) as f:
            weight_map = json.load(f)["weight_map"]
        full_sd = {}
        for shard in set(weight_map.values()):
            path = hf_hub_download(model_name, shard)
            for k, v in safe_load(path, device="cpu").items():
                full_sd[k.replace("model.language_model.", "model.")] = v
    except Exception:
        path = hf_hub_download(model_name, "model.safetensors")
        full_sd = {k.replace("model.language_model.", "model."): v
                   for k, v in safe_load(path, device="cpu").items()}

    model.load_state_dict(full_sd, strict=False)
    del full_sd
    gc.collect()
    return model.to(device)


@register_model("dflux-governed", "governed")
class GovernedHFLM(HFLM):
    """HuggingFace model with D-Flux governor scale profiles applied."""

    def __init__(
        self,
        pretrained: str,
        profile_path: Optional[str] = None,
        base_telemetry: Optional[str] = None,
        target_telemetry: Optional[str] = None,
        signal: str = "dilution_survival",
        strategy: str = "ratio",
        blend: float = 0.3,
        cap: float = 2.0,
        layer_type_bias: str = "full_attention",
        **kwargs,
    ):
        self._profile_path = profile_path
        self._base_telemetry_path = base_telemetry
        self._target_telemetry_path = target_telemetry
        self._signal = signal
        self._strategy = strategy
        self._blend = float(blend)
        self._cap = float(cap)
        self._layer_type_bias = layer_type_bias if layer_type_bias != "none" else None
        self._scale_hooks = []

        super().__init__(pretrained=pretrained, **kwargs)

        # Load or compute profile
        scales = self._resolve_scales()

        # Install hooks
        if scales:
            self._scale_hooks = _install_static_hooks(self.model, scales)
            active = {k: v for k, v in scales.items() if abs(v - 1.0) > 1e-6}
            print(f"\n  [dflux] Installed {len(self._scale_hooks)} scale hooks")
            for idx, val in sorted(active.items()):
                arrow = "↑" if val > 1.0 else "↓"
                print(f"    L{idx:2d}: {val:.4f} {arrow}")
            print()

    def _resolve_scales(self) -> Dict[int, float]:
        """Get scales from profile file or telemetry diff."""
        if self._profile_path:
            profile = load_profile(self._profile_path)
            return profile["scales"]

        if self._base_telemetry_path and self._target_telemetry_path:
            with open(self._base_telemetry_path) as f:
                base_t = json.load(f)
            with open(self._target_telemetry_path) as f:
                target_t = json.load(f)
            profile = compute_profile(
                base_t, target_t,
                signal=self._signal, strategy=self._strategy,
                blend=self._blend, cap=self._cap,
                layer_type_bias=self._layer_type_bias,
            )
            return {int(k): v for k, v in profile["scales"].items()}

        return {}

    def _create_model(self, pretrained, revision=None, dtype=None,
                      trust_remote_code=None, **kwargs):
        """Override to handle Qwen3.5 loading with key remapping."""
        import transformers

        config = self._config
        is_qwen35 = getattr(config, "model_type", "") == "qwen3_5"

        if is_qwen35 and _detect_nested_keys(pretrained):
            from lm_eval.models.huggingface import get_dtype
            torch_dtype = get_dtype(dtype)
            device = self._device
            self._model = _load_qwen35_manual(pretrained, config, torch_dtype, device)
        else:
            super()._create_model(
                pretrained, revision=revision, dtype=dtype,
                trust_remote_code=trust_remote_code, **kwargs,
            )

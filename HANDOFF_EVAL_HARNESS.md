# Claude Code Handoff: lm-evaluation-harness Integration for D-Flux Governor Profiles

## Context

D-Flux has a `LiveGovernor` that applies per-layer scale factors to attention output projections (`o_proj`) during inference. We've demonstrated that a static scale profile (7 numbers for Qwen3.5-9B's full attention layers) can shift a base model's behavior toward reasoning-distilled output — no weight changes, no training.

**The task: build an lm-evaluation-harness integration so we can benchmark governed vs ungoverned models on standard reasoning benchmarks and get hard numbers.**

## What Exists

### D-Flux codebase: `dflux/`

- `src/dflux/live_governor.py` — `LiveGovernor` class with:
  - `LiveGovernor.distillation_governor(model, tokenizer, target_scales, blend=1.0)` — applies static per-layer scales
  - `LiveGovernor.from_telemetry_diff(model, tokenizer, base_telem, target_telem, signal=..., strategy=..., blend=..., layer_type_bias=...)` — computes scales from two telemetry JSONs
  - `.detach()` — removes all hooks (WARNING: also calls `reset_scales()`, resets to 1.0)
  - `._scales` — dict[int, torch.Tensor] mapping layer index to mutable scale tensor
  - `._install_scale_hooks(model)` — installs forward hooks on each layer's attn o_proj

- `src/dflux/multiscale_telemetry.py` — `MultiScaleTelemetry` with:
  - `MultiScaleTelemetry._find_transformer_layers(model)` — finds transformer blocks
  - `MultiScaleTelemetry._find_attn_module(layer)` — finds attn sub-module (handles `self_attn`, `linear_attn`, etc.)

- `examples/distillation_governor.py` — has `_detect_nested_keys()` and `_load_qwen35()` for Qwen3.5 loading with config promotion and weight key remapping

### Validated profiles (from experiments)

Best profile so far: `dilution_survival`, `full_attention` bias, `blend 0.3`

Telemetry files in `data/telemetry/`:
- `telemetry_Qwen-Qwen3.5-9B-Base_base.json`
- `telemetry_Qwen-Qwen3.5-9B_instruct.json`
- `telemetry_Jackrong-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled_reasoning.json`

### Qwen3.5 loading quirks

Qwen3.5 has a hybrid architecture (24 DeltaNet + 8 full softmax attention layers). Loading requires:
1. Config promotion: copy `text_config` fields to top-level config (skip deprecated attrs: `use_return_dict`, `output_hidden_states`, `output_attentions`, `torchscript`, `pruned_heads`, `is_encoder_decoder`)
2. Weight key detection: official checkpoints use `model.language_model.*` keys, fine-tunes use flat `model.*` keys. Use `_detect_nested_keys()` to peek at safetensors index before loading.
3. Use `dtype=` not `torch_dtype=` for `from_pretrained` (HuggingFace deprecation)
4. `flash-linear-attention` not installed warning on MPS is expected/harmless

## What to Build

### 1. Custom lm-eval model wrapper: `src/dflux/eval_model.py`

Subclass `HFLM` from `lm_eval.models.huggingface`. The existing `hf_steered.py` in lm-eval is the exact pattern to follow — it subclasses HFLM and applies hooks during `_model_call` and `_model_generate`.

```python
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

@register_model("dflux-governed", "governed")
class GovernedHFLM(HFLM):
    """
    HuggingFace model with D-Flux governor scale profiles applied.

    Usage:
        lm_eval --model dflux-governed \
            --model_args pretrained=Qwen/Qwen3.5-9B-Base,profile_path=profiles/reasoning.json \
            --tasks gpqa_diamond
    """

    def __init__(self, pretrained, profile_path=None,
                 base_telemetry=None, target_telemetry=None,
                 signal="dilution_survival", strategy="ratio",
                 blend=0.3, cap=2.0, layer_type_bias="full_attention",
                 trust_remote_code=True, **kwargs):
        # ... see design below
```

**Two modes for specifying profiles:**

Mode A — **Profile JSON file** (pre-computed scales):
```json
{
  "scales": {"3": 1.18, "7": 1.19, "11": 1.10, "15": 0.99, "19": 1.00, "23": 0.99, "27": 1.01},
  "signal": "dilution_survival",
  "blend": 0.3,
  "source": "telemetry diff: Qwen3.5-9B-Base vs Reasoning-Distilled"
}
```

Mode B — **Telemetry diff on the fly** (provide base + target telemetry JSONs, compute scales at load time):
```bash
--model_args pretrained=Qwen/Qwen3.5-9B-Base,base_telemetry=data/telemetry/base.json,target_telemetry=data/telemetry/reasoning.json,signal=dilution_survival,blend=0.3
```

**Key implementation details:**

1. Do NOT use LiveGovernor directly — it installs telemetry hooks which add overhead and aren't needed for static profiles. Instead, replicate only the scale hook installation:
   - Use `MultiScaleTelemetry._find_transformer_layers(model)` to find layers
   - Use `MultiScaleTelemetry._find_attn_module(layer)` to find attention modules
   - Find `o_proj` (or `c_proj`, `dense`, `out_proj`) on each attention module
   - Install a simple forward hook: `output * scale_tensor`
   - The hook should short-circuit when scale == 1.0 for zero overhead on unscaled layers

2. For Qwen3.5, the model wrapper needs the same loading logic as `_load_qwen35()` in the examples. Override `_get_config()` or handle in `__init__` before calling `super().__init__()`. Alternatively, since HFLM handles model loading, you may need to override `_create_model()` to use the Qwen3.5 loading path.

3. The `_model_call()` and `_model_generate()` don't need overrides — hooks are persistent forward hooks that fire automatically. Just let the parent class handle inference.

4. For `from_telemetry_diff` mode, the scale computation logic is in `LiveGovernor.from_telemetry_diff` — extract the ratio/delta computation into a standalone function that doesn't need a model (just two telemetry dicts → dict of scales).

### 2. Profile file format: `profiles/`

Create a `profiles/` directory with ready-to-use profile JSONs:

```
profiles/
  qwen35_9b_reasoning_dilution_survival_0.3.json
  qwen35_9b_reasoning_entropy_reduction_0.5.json
```

Profile JSON schema:
```json
{
  "model_family": "Qwen3.5-9B",
  "description": "Reasoning boost via dilution_survival, full_attention only",
  "signal": "dilution_survival",
  "strategy": "ratio",
  "blend": 0.3,
  "layer_type_bias": "full_attention",
  "scales": {
    "3": 1.18,
    "7": 1.19,
    "11": 1.10,
    "15": 0.99,
    "19": 1.00,
    "23": 0.99,
    "27": 1.01
  },
  "source_base": "Qwen/Qwen3.5-9B-Base",
  "source_target": "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled"
}
```

### 3. Evaluation runner script: `examples/run_benchmarks.py`

A convenience script that runs benchmarks with and without a profile:

```bash
# Run baseline + governed, report both
python examples/run_benchmarks.py \
    --model Qwen/Qwen3.5-9B-Base \
    --profile profiles/qwen35_9b_reasoning_dilution_survival_0.3.json \
    --tasks gpqa_diamond,hendrycks_math500,aime_2024 \
    --device mps --dtype bfloat16

# Just governed
python examples/run_benchmarks.py \
    --model Qwen/Qwen3.5-9B-Base \
    --profile profiles/qwen35_9b_reasoning_dilution_survival_0.3.json \
    --tasks gpqa_diamond \
    --skip-baseline
```

The script should:
1. Run lm_eval with `--model hf` (baseline, no profile)
2. Run lm_eval with `--model dflux-governed` (with profile)
3. Print side-by-side comparison table
4. Save results to `data/benchmarks/`

### 4. Standalone scale computation utility: `src/dflux/profile.py`

Extract the scale computation from `LiveGovernor.from_telemetry_diff` into a standalone function:

```python
def compute_profile(base_telemetry: dict, target_telemetry: dict,
                    signal="dilution_survival", strategy="ratio",
                    blend=0.3, cap=2.0, layer_type_bias="full_attention") -> dict:
    """Compute a scale profile from two telemetry JSONs. Returns profile dict."""

def save_profile(profile: dict, path: str): ...
def load_profile(path: str) -> dict: ...
```

This lets people generate profiles without loading a model.

### 5. Update `__init__.py` exports

Add: `GovernedHFLM` (conditional import — only if lm_eval is installed), `compute_profile`, `save_profile`, `load_profile`

## Target Benchmarks

Run these in order (fastest to slowest):

1. **`gpqa_diamond_zeroshot`** — 198 questions, multiple choice, fast. Best first test.
2. **`hendrycks_math500`** — 500 math problems, 5 difficulty levels. Good signal.
3. **`aime_2024`** — 30 problems, hard math. The prestige benchmark.
4. **`ifeval`** — instruction following. Quick sanity check.
5. **`mmlu_pro`** — broad knowledge. Slower but comprehensive.

## Expected CLI Usage

```bash
# Install lm-eval-harness
pip install lm-eval

# Register dflux models (either via entry_points in pyproject.toml or explicit import)
# Option 1: pyproject.toml entry point
# Option 2: --include_path flag
lm_eval --model dflux-governed \
    --model_args pretrained=Qwen/Qwen3.5-9B-Base,profile_path=profiles/qwen35_9b_reasoning_dilution_survival_0.3.json,trust_remote_code=True,dtype=bfloat16 \
    --tasks gpqa_diamond_zeroshot \
    --batch_size 1 \
    --output_path data/benchmarks/

# Or with on-the-fly profile computation
lm_eval --model dflux-governed \
    --model_args pretrained=Qwen/Qwen3.5-9B-Base,base_telemetry=data/telemetry/telemetry_Qwen-Qwen3.5-9B-Base_base.json,target_telemetry=data/telemetry/telemetry_Jackrong-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled_reasoning.json,signal=dilution_survival,blend=0.3,layer_type_bias=full_attention,trust_remote_code=True,dtype=bfloat16 \
    --tasks gpqa_diamond_zeroshot \
    --batch_size 1

# lm_eval needs to find our model class. Use --include_path:
lm_eval --include_path src/dflux \
    --model dflux-governed \
    --model_args pretrained=Qwen/Qwen3.5-9B-Base,profile_path=profiles/reasoning.json \
    --tasks gpqa_diamond_zeroshot
```

## Registration

Two options for registering the custom model:

**Option A: `--include_path` (simpler, no install needed)**
lm_eval's `--include_path` auto-imports Python files. Put `eval_model.py` in a discoverable location and use `--include_path src/dflux`.

**Option B: `pyproject.toml` entry point (cleaner)**
```toml
[project.entry-points."lm_eval.models"]
dflux-governed = "dflux.eval_model:GovernedHFLM"
```

Prefer Option A for development, Option B for release.

## Testing Plan

1. Sanity check: run GPT-2 with a dummy profile (all scales 1.0) — score should match `--model hf` baseline exactly
2. Sanity check: run GPT-2 with a non-trivial profile — score should differ from baseline
3. Run Qwen3.5-9B-Base on `gpqa_diamond_zeroshot` baseline vs governed with `dilution_survival blend=0.3 full_attention` profile
4. If GPQA shows improvement, run `hendrycks_math500` and `aime_2024`
5. Compare governed base model scores against published Qwen3.5-9B instruct scores

## Key Gotchas

1. **Don't use LiveGovernor for benchmarks** — it installs telemetry hooks that add overhead and capture snapshots. For static profiles, only the scale hooks are needed.
2. **Qwen3.5 loading** — needs config promotion and weight key detection. See `_load_qwen35()` in `examples/distillation_governor.py`.
3. **`detach()` resets scales** — if you ever use LiveGovernor, save scales before calling detach.
4. **`flash-linear-attention` warning** — expected on MPS, harmless, falls back to PyTorch.
5. **`dtype=` not `torch_dtype=`** — HuggingFace renamed this parameter.
6. **GPQA requires HuggingFace login** — the dataset is gated. Run `huggingface-cli login` first.
7. **Batch size** — start with `--batch_size 1` on MPS. The 9B model on Apple Silicon won't have room for larger batches.

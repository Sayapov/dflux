# DFlux

X-ray for transformers. See what your model is actually doing.

```
pip install dflux
```

DFlux treats neural network internals as fluid dynamics — gradient flow during training, activation energy during inference, attention head power distribution at any time. Six tools, one framework:

| Tool | What it does | When to use it |
|------|-------------|----------------|
| **MultiScaleTelemetry** | 26-field per-token signal capture across 8 scales | During inference — full internal state fingerprint |
| **FineProbe** | Per-head activation X-ray | During inference — see which heads carry energy |
| **HeadSurgeon** | Direct head modification | Post-training — tune head influence without retraining |
| **CausalPrimitives** | Per-head causal contribution measurement | During inference — who actually caused the output |
| **DFluxMeter** | Gradient energy monitoring | During training — early warning for instability |
| **AXE-NS** | Adaptive intervention engine | During training — auto-stabilizes gradient flow |

## Multi-Scale Telemetry: 8 signals, one forward pass

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflux import MultiScaleTelemetry

model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

telem = MultiScaleTelemetry.from_model(model, tokenizer)

with torch.no_grad():
    model.generate(input_ids, max_new_tokens=32, do_sample=False)

print(telem.summary())
telem.save("telemetry.json")
```

Captures per token, per layer, simultaneously:

1. **Residual Stream Trajectory** — L2 norm, delta magnitude, direction change
2. **Logit Lens** — project residual onto unembedding at each layer, watch predictions form
3. **Cross-Layer Influence** — pairwise cosine similarity of layer contributions
4. **MLP Contributions** — MLP vs attention energy, dead neuron fraction, outlier ratios
5. **Entropy Cascade** — prediction entropy at each layer, entropy reduction rate
6. **Activation Outliers** — max magnitude, Gini coefficient, top outlier dimensions
7. **Dimension Channels** — persistent hidden dimensions that carry energy across >50% of layers
8. **Dilution Analysis** — per-layer survival of contributions to final output (inspired by [AttnRes](https://github.com/MoonshotAI/Attention-Residuals/))

All signals are raw numbers. No labels. No classifications. Interpretation is your job.

## Deterministic fingerprint

Under greedy decoding (`do_sample=False`), the telemetry is **bit-for-bit identical** across runs. Verified on:

- GPT-2 (12 layers) — 2 runs, 0 diffs
- Pythia-1.4B (24 layers) — 2 runs, 0 diffs
- Mistral-7B (32 layers) — 5 runs, 10 pairwise comparisons, 115,795 numerical values per run, **zero differences**

This means single-run ablation studies have zero uncertainty. Change one head, prune one layer, quantize one weight — run telemetry once, diff against baseline, and every deviation is a guaranteed causal effect.

```bash
python examples/reproducibility_test.py --model mistralai/Mistral-7B-v0.1 --device mps --runs 5
```

## What we found: dilution scales with depth, not parameters

Cross-architecture telemetry on 5 models reveals how much compute is wasted — layer contributions that get washed out by the additive residual stream:

| Model | Layers | Params | Mean Wasted Work | Peak Survival |
|-------|--------|--------|-----------------|---------------|
| GPT-2 | 12 | 124M | 33.2% | 0.898 |
| Pythia-1.4B | 24 | 1.4B | 79.0% | 0.394 |
| Pythia-2.8B | 32 | 2.8B | 80.9% | 0.316 |
| Phi-2 | 32 | 2.7B | 75.1% | 0.401 |
| Mistral-7B | 32 | 7.2B | 68.6% | 0.489 |

Dilution is an architectural property — doubling from 12 to 24 layers more than doubles waste. Mistral compensates with 137x residual norm growth; Phi-2 with 98.3% entropy reduction. GPT-2 is the outlier: shallow enough that most layer work survives.

```bash
# Run telemetry on multiple models
python examples/multiscale_telemetry.py --model "gpt2,EleutherAI/pythia-1.4b" --device mps

# Compare existing runs
python examples/multiscale_telemetry.py --compare "telemetry_gpt2.json,telemetry_EleutherAI-pythia-1.4b.json"
```

## Quick start: X-ray a model in 10 lines

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflux import FineProbe

model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

probe = FineProbe.from_model(model)
input_ids = tokenizer.encode("The capital of France is", return_tensors="pt")

with torch.no_grad():
    model.generate(input_ids, max_new_tokens=32)

report = probe.report()
print(f"Layers: {report['n_layers']}, Heads: {report['n_heads']}")
print(f"Hallucination risk: {report['mean_risk']:.3f}")
print(f"Head entropy: {report['mean_head_entropy']:.3f}")
print(f"Head energy map: {report['avg_head_energy']}")  # [n_layers × n_heads]
```

Works with GPT-2, LLaMA, Mistral, Pythia, Phi, Qwen, and any model using standard attention projection patterns.

## Quick start: Head surgery in 5 lines

```python
from dflux import HeadSurgeon

surgeon = HeadSurgeon(model)
surgeon.scale_head(layer=11, head=7, factor=0.8)  # dampen dominant output head
surgeon.scale_head(layer=0, head=2, factor=2.0)   # boost upstream skeptic head
# No training. No data. Instant. Reversible.
surgeon.restore()  # undo everything
```

Or let the probe data decide:

```python
# Run probe on factual prompts → factual_report
# Run probe on hallucination prompts → halluc_report
surgeon.auto_calibrate(factual_report, halluc_report)
# Automatically boosts heads that disengage during hallucination
# and dampens heads that spike during hallucination
```

## Causal Primitives: who caused the output?

```python
from dflux import CausalPrimitives

cp = CausalPrimitives.from_model(model)

with torch.no_grad():
    model.generate(input_ids, max_new_tokens=16)

results = cp.results()
for r in results:
    print(f"L{r.layer} H{r.head}: necessity={r.necessity:.3f} sufficiency={r.sufficiency:.3f}")
```

Measures necessity (does removing this head change the output?) and sufficiency (does this head alone produce the output?) for every attention head.

## AXE-NS: training stabilizer

Five-tower adaptive engine that monitors gradient flow and intervenes when training destabilizes.

```python
from dflux import AXEEngine

engine = AXEEngine.from_optimizer(optimizer, window_steps=100, beta=50)

for step in range(num_steps):
    loss.backward()
    action = engine.step(step, loss=loss.item(), lr=lr)

    if action.kind == "extract":
        engine.apply_extract(optimizer)
    elif action.kind == "renormalize":
        engine.apply_renormalize(optimizer)

    optimizer.step()
    optimizer.zero_grad()
```

Regimes: laminar → transitional → turbulent → critical. Interventions are proportional to regime severity.

## DFluxMeter: gradient telemetry

Lightweight gradient energy monitor. No intervention, just measurement.

```python
from dflux import DFluxMeter

meter = DFluxMeter.from_optimizer(optimizer, L_cut=5, window_steps=200)

for step in range(num_steps):
    loss.backward()
    meter.step(step, loss=loss.item(), lr=lr)
    optimizer.step()
    optimizer.zero_grad()

meter.close()
```

## Framework adapters

### HuggingFace Trainer

```python
from dflux.adapters.hf_trainer import DFluxCallback

trainer = Trainer(..., callbacks=[DFluxCallback(run_id="run-001")])
trainer.train()
```

### PyTorch Lightning

```python
from dflux.adapters.lightning import DFluxCallback

trainer = L.Trainer(callbacks=[DFluxCallback(run_id="run-001")])
trainer.fit(model, datamodule)
```

## Install

```bash
pip install dflux              # core (no dependencies)
pip install dflux[hf]          # + HuggingFace Transformers
pip install dflux[lightning]   # + PyTorch Lightning
pip install dflux[all]         # everything
```

## Architecture support

Auto-detection for: GPT-2, LLaMA, Mistral, Pythia (NeoX), Phi, Qwen, MPT, BERT, Falcon.

Adding support for a new architecture = adding one pattern to the detection logic.

## References & related work

DFlux builds on or is inspired by the following:

- **Jansma & Hoel (2025)**, "Engineering Emergence" — the causal primitives framework (determinism, specificity, mutual information as causation measure) that our `CausalPrimitives` and `CPSurgeon` modules implement. [Paper](https://arxiv.org/abs/2502.11753)
- **Moonshot AI (2026)**, "Attention Residuals" — demonstrated that standard additive residual connections dilute early-layer contributions in deep transformers. Our dilution analysis (Scale 8 of MultiScaleTelemetry) empirically measures this per-layer survival. [GitHub](https://github.com/MoonshotAI/Attention-Residuals/)
- **State Flow Machine** (Chang Cheng, 2025) — explores explicit state tracking through computation layers, related to our dimension channel tracking which finds persistent information highways in the residual stream. [GitHub](https://github.com/changcheng967/state-flow-machine)

## License

MIT. Free instrument. Use it, fork it, build on it.

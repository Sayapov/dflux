# Transformer telemetry is bit-for-bit deterministic — and it shows 79% of layer compute in deep models is wasted

Built a tool that hooks into the forward pass and captures everything: residual norms, logit lens projections, entropy, MLP vs attention energy, activation outliers, cross-layer influence, and per-layer dilution survival. 26 signals per token, per layer, all at once.

**The practical result:** under greedy decoding, every number is identical across runs. Tested on four architectures:

- GPT-2 (12L, dense softmax) — 2 runs, 0 diffs
- Pythia-1.4B (24L, dense softmax) — 2 runs, 0 diffs
- Mistral-7B (32L, GQA softmax) — 5 runs, 10 pairs, 115,795 values/run, zero diffs
- **Qwen3.5-9B** (32L, hybrid DeltaNet + softmax) — 2 runs, 0 diffs

The Qwen3.5 one is interesting — it has 24 Gated DeltaNet layers (linear attention with recurrent state updates) mixed with 8 full softmax layers. Even the recurrent path is fully deterministic under greedy. This isn't a quirk of one attention mechanism.

Why this matters for inference: if you're pruning layers, quantizing, swapping heads, or building routing logic — you can run telemetry once before your change, once after, and every single deviation you see is the exact causal effect. No need to average over runs. One run = ground truth.

**Base vs instruct on Qwen3.5-9B:**

Compared the base and instruct versions. The hybrid architecture (3 DeltaNet → 1 full attention, repeating) lets you see where instruction tuning lands:

- Full attention layers took 2.1x larger dilution survival shifts than DeltaNet layers — instruction following rewrites the softmax routing
- DeltaNet layers got more efficient — wasted work dropped 4.7% vs only 1.0% in full attention
- Final entropy goes up 4.5% — instruct model hedges more across candidates instead of slamming on one token
- MLP dominance barely moves (~58% both) — instruction tuning mostly changes attention patterns, not the MLP/attention balance

**The dilution data:**

Ran it across 5 architectures. For each layer, measured how much of what it computed actually survived to the final output:

- GPT-2 (12L): 33% waste, 0.898 peak survival
- Pythia-1.4B (24L): 79% waste, 0.394 peak survival
- Pythia-2.8B (32L): 81% waste, 0.316 peak survival
- Phi-2 (32L): 75% waste, 0.401 peak survival
- Mistral-7B (32L): 69% waste, 0.489 peak survival

Waste scales with depth, not parameters. Directly relevant for layer pruning — the telemetry tells you exactly which layers contribute least to the final output.

**What it captures (all per-token, per-layer):**

- Residual stream norms, deltas, direction changes
- Logit lens: model's prediction forming at each layer
- MLP vs attention energy split + dead neuron fraction
- Entropy cascade: where the model narrows its prediction
- Dimension channels: persistent information highways in the residual stream
- Dilution survival: how much of each layer's work makes it to the end
- Layer type annotation for hybrid architectures (DeltaNet vs full attention)

MIT licensed, works with any HuggingFace model. One-liner setup:

```python
from dflux import MultiScaleTelemetry

telem = MultiScaleTelemetry.from_model(model, tokenizer)
model.generate(input_ids, max_new_tokens=32, do_sample=False)
print(telem.summary())
```

GitHub: https://github.com/iskandersayapov/dflux

Auto-detects GPT-2, LLaMA, Mistral, Pythia, Phi, Qwen (including Qwen3.5 hybrid DeltaNet), MPT, BERT, Falcon.

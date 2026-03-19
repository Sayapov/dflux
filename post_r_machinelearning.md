# [P] We measured per-layer dilution survival across 5 transformer architectures — 79% of compute in 24-layer models produces work that gets washed out

We built an open-source telemetry system that hooks into transformer forward passes and captures 26 raw numerical signals per token across 8 scales simultaneously — residual stream trajectory, logit lens projections, cross-layer influence, MLP internals, entropy cascade, activation outliers, dimension channels, and dilution survival.

Then we ran it on GPT-2, Pythia-1.4B, Pythia-2.8B, Phi-2, and Mistral-7B.

**The dilution finding:**

For each layer, we measure how much of that layer's delta (what it added to the residual stream) actually survives to the final output. Survival = cosine similarity between the layer's contribution and the final hidden state. Wasted work = delta norm × (1 - |survival|).

| Model | Layers | Mean Wasted Work | Peak Survival |
|-------|--------|-----------------|---------------|
| GPT-2 | 12 | 33.2% | 0.898 |
| Pythia-1.4B | 24 | 79.0% | 0.394 |
| Pythia-2.8B | 32 | 80.9% | 0.316 |
| Phi-2 | 32 | 75.1% | 0.401 |
| Mistral-7B | 32 | 68.6% | 0.489 |

Dilution scales with depth, not parameters. Doubling layers from 12 to 24 more than doubles waste. This is the exact problem that Moonshot AI's Attention Residuals paper (2026) addresses by replacing additive residual connections with learned attention over previous layers — our data confirms their thesis empirically.

**The reproducibility finding:**

Under greedy decoding, the telemetry is bit-for-bit deterministic across four architectures:

- GPT-2 (12L, dense softmax) — 2 runs, 0 diffs
- Pythia-1.4B (24L, dense softmax) — 2 runs, 0 diffs
- Mistral-7B (32L, GQA softmax) — 5 runs, 10 pairwise comparisons, 115,795 values per run, zero diffs
- Qwen3.5-9B (32L, hybrid DeltaNet + softmax) — 2 runs, 0 diffs

The Qwen3.5 result matters — its Gated DeltaNet layers use a recurrent state update rule (linear attention), not standard QKV softmax. The deterministic fingerprint holds across standard attention, grouped-query attention, and linear/recurrent attention. It's a property of greedy decoding, not a quirk of one mechanism.

This means single-run ablation studies have zero uncertainty. One run = ground truth.

**Base vs instruct on a hybrid architecture:**

Compared Qwen3.5-9B-Base vs Qwen3.5-9B (instruct) — 24 DeltaNet layers + 8 full attention layers in a 3:1 pattern. Instruction tuning hits the two layer types differently:

- Full attention layers show 2.1x larger dilution survival shifts (12.2% vs 5.7%) — instruction following rewrites *what information survives* through softmax routing
- DeltaNet layers absorbed the efficiency gains — wasted work dropped 4.7% in linear attention vs 1.0% in full attention
- Final-layer entropy increases 4.5% — the instruct model hedges more across candidates
- Norm growth slightly reined in (37.9x → 36.1x)

**Other findings from the telemetry:**

- Pythia shows a three-phase architecture: scaffolding (L0-7, 9.1% survival), decision-making (L8-15, 20% survival + biggest entropy drops), and a paradox zone (L16-23, 32.5% survival but *increases* entropy)
- Mistral compensates for dilution with 137x residual norm growth across its 32 layers
- Phi-2 compensates differently — 98.3% entropy reduction, the most aggressive prediction narrowing of any model tested
- Dimension channels (hidden dims that carry disproportionate energy across >50% of layers) decrease with depth: 11 in GPT-2 → 5 in Pythia-2.8B. Deeper models route through fewer persistent information highways
- Last-layer waste is universal across all deep models

The tool is called DFlux. MIT licensed.

GitHub: https://github.com/iskandersayapov/dflux

All signals are raw numbers — no labels, no classifications, no "this head is a skeptic" narratives. The telemetry captures what's happening; interpretation is separate.

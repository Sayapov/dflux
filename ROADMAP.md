# D-Flux Research Roadmap — March/April 2026

## Current State (March 17)

**What we have:**
- LiveGovernor with static profiles and reactive mode
- One architecture validated (Qwen3.5-9B, hybrid: 24 DeltaNet + 8 full attention)
- One signal validated (dilution_survival, ratio strategy)
- Complete ARC Challenge blend curve: −0.5 to 1.0, monotonically increasing
- Inverse profile causality confirmed
- Profile outperforms its source model by +3.67 points
- Governed base beats instruct by +3.41 points
- Theory paper drafted (information_fluid_dynamics.md)
- HellaSwag 0.5 and 1.0 results pending

**What we don't have yet:**
- Second architecture validated
- Profile search/optimization
- Dynamic governor tested
- Per-head scaling
- Multi-signal profiles
- GSM8K or math benchmarks

---

## Week 1: Solidify the Foundation

### 1. Collect pending results [Day 1]
- [ ] HellaSwag blend 0.5 and 1.0 — confirms whether ARC gains are free or trade-off
- [ ] Update paper with HellaSwag data
- [ ] If HellaSwag is flat: headline is "free lunch." If it drops: headline is "task-specific profiles"

### 2. GSM8K benchmark [Day 1-2]
- [ ] Run GSM8K on: base, instruct, distilled, governed base (blend 0.5 and 1.0)
- [ ] This is the distilled model's home turf (chain-of-thought math reasoning)
- [ ] Key question: does the profile help on tasks where the source model excels?
- [ ] Also tests whether improvement transfers across task types

### 3. Profile stacking experiment [Day 2]
- [ ] Compose dilution_survival and entropy_reduction profiles (element-wise multiply)
- [ ] The two signals produce nearly opposite shapes — stacking tests whether complementary profiles combine constructively
- [ ] Run on ARC Challenge at blend 0.3 and 0.5 for each, composed
- [ ] Tests the superposition prediction from Section 8 of the paper

### 4. Second architecture: Mistral-7B [Day 2-3]
- [ ] We already have Mistral telemetry in `data/telemetry/`
- [ ] Run telemetry on a Mistral-instruct or fine-tuned variant
- [ ] Compute distillation profile
- [ ] Mistral is pure transformer (all stateless layers) — framework predicts ALL layers are governable
- [ ] Run ARC Challenge with Mistral governed profile
- [ ] Key test: does the all-layers-governable prediction hold?

### 5. Push to GitHub [Day 3]
- [ ] Commit all new profiles, benchmark data, paper
- [ ] Clean up README with current results
- [ ] Make the repo presentable for external eyes

---

## Week 2: Expand the Search Space

### 6. Automated profile search [Day 4-5]
- [ ] Implement Bayesian optimization over the 8 scale factors
- [ ] Objective: maximize ARC acc_norm (or weighted multi-benchmark score)
- [ ] Search bounds: [0.5, 1.5] per scale factor, constrained to full-attention layers
- [ ] Budget: 100-200 evaluations (~20-30 GPU-hours on 9B)
- [ ] This will find the empirically optimal static profile for ARC
- [ ] Compare searched profile vs hand-derived profile to measure "search headroom"

### 7. Per-head scaling proof of concept [Day 5-6]
- [ ] Extend LiveGovernor to support per-head scale tensors (not just per-layer)
- [ ] 8 layers × 32 heads = 256 free parameters
- [ ] Start with hand-derived: use per-head telemetry to identify dominant vs redundant heads
- [ ] Run ARC Challenge with per-head profile
- [ ] Key question: does 256 params outperform 8 params on the same task?

### 8. Dynamic governor experiment [Day 6-7]
- [ ] Implement the EMA-smoothed update rule from Section 9.4
- [ ] Use static dilution_survival profile (blend 0.5) as setpoint
- [ ] Parameters to tune: α (EMA smoothing, start at 0.95), κ (gain, start at 0.1)
- [ ] Drift bounds: σ ∈ [0.5, 2.0]
- [ ] Run on ARC Challenge first (compare against static blend 1.0)
- [ ] Then run on FREE GENERATION — this is where dynamic governance matters most
  - Static blend 1.0 breaks free gen (loops). Can the dynamic governor maintain blend 1.0 quality while avoiding turbulence?
  - If yes: the governor automatically finds the per-token optimal regime, staying laminar where the static profile would go turbulent

### 9. More benchmarks [Day 7-8]
- [ ] Winogrande (commonsense reasoning)
- [ ] MMLU-Pro (multi-domain knowledge)
- [ ] Build a multi-benchmark score to prevent overfitting to ARC
- [ ] Run best profile from search (Step 6) on all benchmarks

---

## Week 3+ : Cross-Architecture and Publication

### 10. Architecture survey [Week 3]
- [ ] Llama-3-8B (pure transformer) — predict: fully governable, all layers safe
- [ ] Mamba (pure SSM) — predict: minimally governable, scales break state
- [ ] Mixtral-8x7B (MoE) — predict: governable + router as additional control surface
- [ ] For each: telemetry, distillation profile, ARC benchmark, compare to predictions

### 11. Quantization interaction [Week 3]
- [ ] Run telemetry on Qwen3.5-9B full precision vs GPTQ-Int4
- [ ] Does quantization change the flow profile? Does the same scale profile work on quantized models?
- [ ] Critical for the marketplace: profiles need to transfer across quantization levels

### 12. Paper finalization [Week 3-4]
- [ ] Fill in multi-architecture results
- [ ] Add searched profile results
- [ ] Add dynamic governor results
- [ ] Formalize the Information Reynolds Number with quantitative calibration
- [ ] Submit to arXiv

### 13. Reddit / community launch [Week 3-4]
- [ ] Post to r/MachineLearning (academic framing)
- [ ] Post to r/LocalLLaMA (practical framing — "8 numbers that beat instruct")
- [ ] Include code, profiles, reproduction instructions

---

## Open Research Questions (Longer Term)

- **Signal combination search**: what weighted blend of dilution_survival + entropy_reduction is optimal?
- **Training-time telemetry**: record flow dynamics during pretraining. When does reasoning crystallize?
- **Architecture design for governability**: what makes a model maximally responsive to scale profiles?
- **Profile transfer across model sizes**: does a 9B profile transfer to 35B of the same family?
- **Adversarial profiles**: can a profile degrade a model's safety? What are the defense implications?
- **Biological neural network analogy**: do biological attention mechanisms exhibit similar flow dynamics?

---

## Hardware Requirements

| Experiment | Compute | Time estimate |
|---|---|---|
| Single ARC Challenge run (9B) | 1 GPU | ~10-15 min |
| Full blend sweep (7 points) | 1 GPU | ~90 min |
| Bayesian search (200 evals) | 1-4 GPUs | ~20-30 hrs (parallelizable) |
| Per-head telemetry | 1 GPU | ~30 min |
| Multi-architecture survey (4 models) | 1-2 GPUs | ~1 day |
| Dynamic governor experiment | 1 GPU | ~2-3 hrs |

All experiments run on single GPU. Search is embarrassingly parallelizable.

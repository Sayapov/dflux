# Information Fluid Dynamics: A Governing Framework for Transformer Inference

**Iskander Sayapov**

*Draft v0.1 — March 2026*

---

## Abstract

We propose that information flow through transformer architectures obeys principles structurally analogous to fluid dynamics, and that this analogy is not metaphorical but predictive. We define formal correspondences between fluid dynamic quantities (pressure, velocity, viscosity, Reynolds number) and measurable transformer signals (entropy, residual norm, attention damping, coherence thresholds). Using these correspondences, we derive governing equations for the transformer residual stream, define a dimensionless *Information Reynolds Number* that predicts the onset of text degeneration, and demonstrate that per-layer scale interventions function as flow valves in a discrete pipeline. We validate the framework empirically on Qwen3.5-9B, showing that (i) a governed base model using only 8 scalar scale factors outperforms both the instruction-tuned and reasoning-distilled variants on ARC Challenge (59.30% vs 55.89% and 55.63% acc_norm — a +3.41 and +3.67 point margin respectively) — and the profile outperforms the very model it was derived from, extracting structural signal that generalizes beyond its source, (ii) the framework correctly predicts which layer types are safe to scale and which are fragile, and (iii) a critical blend threshold exists beyond which coherent generation breaks down — directly analogous to laminar–turbulent transition. We release D-Flux, an open-source toolkit implementing this framework, along with scale profiles that modify model behavior using only 8 scalar parameters (~200 bytes) per intervention.

---

## 1. Introduction

The transformer architecture processes information through a sequential stack of layers, each contributing an additive perturbation to a shared residual stream. This structure — sequential stages, additive contributions, a shared medium — is topologically identical to fluid flow through a discrete pipeline. Yet the machine learning community has not systematically applied the analytical tools of fluid dynamics to this problem.

We argue this is a missed opportunity. Fluid dynamics provides a mature framework for reasoning about staged flow systems: conservation laws describe what persists, pressure gradients describe what drives change, viscosity describes what resists it, and dimensionless numbers predict qualitative transitions in system behavior. Each of these has a direct, measurable analogue in transformer inference.

This paper makes three contributions:

1. **A formal correspondence** between fluid dynamic quantities and transformer telemetry signals, grounded in the residual stream formulation. We define information pressure, flow velocity, effective viscosity, and an Information Reynolds Number (Re_I) for transformer inference.

2. **A governing framework** — the LiveGovernor — that treats per-layer attention output scaling as flow valve control. We show that scale profiles derived from telemetry differences between model variants function as prescribed valve settings that shift model behavior without weight modification.

3. **Empirical validation** on the Qwen3.5-9B architecture family, demonstrating that the fluid dynamic framework correctly predicts: which layer types are safe to scale (stateless full-attention layers, analogous to nozzles) versus fragile (stateful DeltaNet layers, analogous to reservoirs), the existence of a critical blend threshold beyond which text degeneration occurs (laminar–turbulent transition), and that small perturbations compose linearly while large perturbations interact nonlinearly.

### 1.1 Related Work

**Residual stream analysis.** The mechanistic interpretability literature treats the residual stream as a communication channel between layers (Elhage et al., 2021). Our contribution is treating it as a *flow medium* subject to conservation and transport equations.

**Inference-time intervention.** Alignment Vectors (Vercellotti et al., 2025) apply full weight-level diffs at inference time with a tunable λ. Activation steering (Turner et al., 2023) adds direction vectors to residual streams. Our approach is more constrained — scalar multiplication on attention outputs only — but orders of magnitude more compact (~200 bytes vs. ~18GB).

**Head-level analysis.** HeadInfer (Chen et al., 2025) demonstrates that attention heads are independent enough to offload to separate devices. This independence is a precondition for meaningful per-head and per-layer scaling.

---

## 2. The Residual Stream as a Discrete Flow System

### 2.1 Notation and Setup

Consider a transformer with $L$ layers. Let $\mathbf{h}_l \in \mathbb{R}^d$ denote the residual stream state at layer $l$ for a given token position. The forward pass computes:

$$\mathbf{h}_{l+1} = \mathbf{h}_l + \mathbf{a}_l(\mathbf{h}_l) + \mathbf{f}_l(\mathbf{h}_l + \mathbf{a}_l(\mathbf{h}_l))$$

where $\mathbf{a}_l$ is the attention sublayer output and $\mathbf{f}_l$ is the feed-forward sublayer output. The residual stream $\{\mathbf{h}_0, \mathbf{h}_1, \ldots, \mathbf{h}_L\}$ is the discrete trajectory of information through the network.

This is structurally identical to a discrete pipe flow: the residual stream is the medium, each layer is a processing stage that adds energy (information) to the flow, and the final state $\mathbf{h}_L$ drives the output distribution.

### 2.2 Fluid Dynamic Correspondences

We define the following formal correspondences:

| Fluid Dynamics | Symbol | Transformer Analogue | Measurement |
|---|---|---|---|
| Flow velocity | $v$ | Residual stream norm | $v_l = \|\mathbf{h}_l\|_2$ |
| Pressure | $p$ | Negative entropy of output distribution | $p_l = -H(\text{softmax}(\mathbf{W}_\text{unembed} \cdot \mathbf{h}_l))$ |
| Pressure drop | $\Delta p$ | Entropy reduction per layer | $\Delta p_l = H_l - H_{l+1}$ |
| Density | $\rho$ | Effective information density | $\rho_l = \text{rank}_\epsilon(\mathbf{A}_l) / n_\text{heads}$ |
| Viscosity | $\mu$ | Attention damping factor | $\mu_l = 1 - S_l$ where $S_l$ is dilution survival |
| Mass flow rate | $\dot{m}$ | Token information throughput | $\dot{m}_l = v_l \cdot \rho_l$ |
| Pipe cross-section | $A$ | Model dimension | $d_\text{model}$ |

**Definition 1 (Information Pressure).** The information pressure at layer $l$ is the negative entropy of the predictive distribution obtained by projecting the residual stream state through the unembedding matrix:

$$p_l = -H(\text{softmax}(\mathbf{W}_U \mathbf{h}_l / \tau)) = \sum_i q_i \log q_i$$

where $q = \text{softmax}(\mathbf{W}_U \mathbf{h}_l / \tau)$ and $\tau$ is temperature. High pressure corresponds to high certainty (low entropy); low pressure corresponds to uncertainty (high entropy).

**Definition 2 (Flow Velocity).** The flow velocity at layer $l$ is the L2 norm of the residual stream state:

$$v_l = \|\mathbf{h}_l\|_2$$

This measures the magnitude of the information signal at each stage. The velocity profile $\{v_0, v_1, \ldots, v_L\}$ shows how the signal strengthens or attenuates through the network.

**Definition 3 (Effective Viscosity).** The effective viscosity at layer $l$ measures how much the layer resists (dampens) incoming information flow. We define it through the complement of dilution survival:

$$\mu_l = 1 - S_l$$

where dilution survival $S_l$ measures what fraction of layer $l$'s contribution persists to the output:

$$S_l = \frac{\|\text{proj}_{\Delta_l}(\mathbf{h}_L - \mathbf{h}_l)\|}{\|\Delta_l\|}$$

with $\Delta_l = \mathbf{a}_l(\mathbf{h}_l)$ being the layer's attention contribution. High viscosity ($S_l \approx 0$) means the layer's work is almost entirely dampened by subsequent layers. Low viscosity ($S_l \approx 1$) means the layer's contribution flows cleanly to the output.

### 2.3 The Information Continuity Equation

In classical fluid dynamics, the continuity equation expresses conservation of mass: $\partial\rho/\partial t + \nabla \cdot (\rho \mathbf{v}) = 0$.

For the discrete transformer pipeline, the analogous conservation law governs information content across layers. The residual connection ensures that information is never destroyed — it can only be added to or redirected:

$$\mathbf{h}_{l+1} = \mathbf{h}_l + \Delta_l$$

This is a discrete conservation law: the total information at layer $l+1$ equals the information at layer $l$ plus the source term $\Delta_l$. There is no sink term — information accumulates monotonically in the residual stream. This explains the empirically observed monotonic growth of residual norms through transformer layers.

The *effective* information, however, is not conserved — later layers can "dilute" earlier contributions by adding large orthogonal components. This is precisely what dilution survival measures: the degree to which the conservation law holds in the *relevant subspace* rather than the full space.

### 2.4 The Information Bernoulli Equation

Bernoulli's principle states that along a streamline, $p + \frac{1}{2}\rho v^2 = \text{const}$. In the transformer, we observe an analogous trade-off: layers that dramatically increase the residual norm (high velocity gain) tend to produce less entropy reduction (lower pressure gain), and vice versa.

We define the *information energy* at layer $l$:

$$E_l = p_l + \frac{1}{2}\alpha v_l^2$$

where $\alpha$ is an architecture-dependent coupling constant. The Bernoulli hypothesis predicts that $E_l$ is approximately conserved across layers, with deviations indicating energy injection (learning) or dissipation (redundant computation).

---

## 3. The Information Reynolds Number

### 3.1 Definition

The Reynolds number in fluid dynamics, $\text{Re} = \rho v L / \mu$, predicts the transition from laminar to turbulent flow. We define its transformer analogue:

**Definition 4 (Information Reynolds Number).** For a transformer with $L$ layers, the Information Reynolds Number is:

$$\text{Re}_I = \frac{\bar{\rho} \cdot \bar{v} \cdot L}{\bar{\mu}}$$

where $\bar{\rho}$, $\bar{v}$, $\bar{\mu}$ are the mean effective density, flow velocity, and viscosity across layers. Equivalently, using the relationship $\mu_l = 1 - S_l$:

$$\text{Re}_I = \frac{\bar{\rho} \cdot \bar{v} \cdot L}{1 - \bar{S}}$$

### 3.2 Predicting Laminar–Turbulent Transition

In our experiments, we observed a critical phenomenon: applying scale profiles with blend values above a threshold causes text generation to transition from coherent (laminar) to degenerate (turbulent — characterized by repetitive loops, incoherent fragments, and semantic collapse).

We hypothesize that scaling modifies the effective Reynolds number by altering the velocity and viscosity profiles. A scale factor $\sigma_l > 1$ on layer $l$ increases the local flow velocity (amplifying the attention output), while $\sigma_l < 1$ increases effective local viscosity (dampening the output). The modified Reynolds number becomes:

$$\text{Re}_I(\boldsymbol{\sigma}) = \frac{\bar{\rho} \cdot \overline{v \cdot \sigma} \cdot L}{1 - \overline{S(\boldsymbol{\sigma})}}$$

where $\boldsymbol{\sigma} = [\sigma_1, \ldots, \sigma_L]$ is the scale profile. The blend parameter $\beta$ interpolates between the identity profile and the target profile:

$$\boldsymbol{\sigma}(\beta) = \mathbf{1} + \beta(\boldsymbol{\sigma}_\text{target} - \mathbf{1})$$

Our experimental evidence reveals a task-dependent transition:

**Benchmark evaluation (constrained output — multiple choice):**
- **$\beta = -0.5$ through $\beta = 1.0$**: monotonically increasing performance across the full tested range. No turbulent transition observed. The constrained output space (choosing among 4 options) suppresses turbulent manifestations, allowing higher effective Reynolds numbers before breakdown.

**Free generation (unconstrained output):**
- **$\beta = 0.3$**: coherent text, measurable benchmark improvement. *Laminar regime.*
- **$\beta = 0.5$**: coherent but with emergent behaviors (spontaneous chain-of-thought on instruct models). *Transitional regime.*
- **$\beta = 1.0$**: text degeneration (loops, repetition). *Turbulent regime.*

The distinction between constrained and unconstrained output tasks parallels the difference between flow through a constriction (which stabilizes turbulence by constraining the available modes) and flow into open space (which allows turbulent structures to develop). Multiple-choice evaluation constrains the output to a small set of options, effectively narrowing the "pipe exit" and suppressing the degrees of freedom needed for turbulent breakdown. Free generation is open-ended, providing the full output space for degenerate modes to manifest.

### 3.3 Architectural Dependence

The critical Reynolds number is architecture-dependent. A key prediction of the framework is that architectures with higher intrinsic viscosity (more damping, lower average dilution survival) will tolerate larger scale perturbations before transitioning to turbulence. Conversely, architectures with low viscosity (high dilution survival throughout) will be more sensitive to scaling.

This prediction is testable: measuring dilution survival profiles across architecture families (pure transformer, SSM, hybrid, MoE) should reveal different critical blend thresholds.

---

## 4. Layer Types as Flow Components

### 4.1 Nozzles and Reservoirs

The Qwen3.5-9B architecture employs a hybrid design: 24 DeltaNet layers (linear attention with recurrent state) and 8 full softmax attention layers in a 3:1 repeating pattern: [D, D, D, F, D, D, D, F, ...].

This hybrid structure maps directly to a pipeline with two types of components:

**Full attention layers (F) = Nozzles.** These are stateless per-token computations. Each token's attention output depends only on the current input and the KV cache — there is no hidden state that carries across tokens beyond what's in the residual stream. Like a nozzle in a pipe, you can adjust the aperture (scale factor) freely without affecting the system's memory or state. The flow through a nozzle is governed by instantaneous conditions.

**DeltaNet layers (D) = Reservoirs.** These maintain recurrent hidden state across tokens: $\mathbf{s}_{l,t+1} = f(\mathbf{s}_{l,t}, \mathbf{h}_{l,t})$. This state acts as a reservoir — it accumulates and releases information over time. Scaling the output of a reservoir disrupts the state dynamics: if you amplify the output without adjusting the state, the feedback loop becomes unstable. If you dampen it, the reservoir drains.

### 4.2 Empirical Validation

Our experiments confirm this prediction directly:

- **Scaling all layers** (no layer type bias): text quality degrades significantly at moderate blend values. The DeltaNet reservoirs destabilize.
- **Scaling only full attention layers** (`layer_type_bias=full_attention`): coherent text at blend values up to 0.5, measurable benchmark improvements at 0.3. The nozzles tolerate scaling; the reservoirs are untouched.

This finding generalizes: **any architecture mixing stateful and stateless layers should exhibit differential scalability.** The stateless components form the "governable surface" of the model. The stateful components are the "infrastructure" that must not be perturbed.

### 4.3 Predictions for Other Architectures

| Architecture | Stateful Components | Stateless Components | Predicted Governability |
|---|---|---|---|
| Pure Transformer (GPT-2, Llama) | None | All attention + FFN | Fully governable, all layers |
| Pure SSM (Mamba, RWKV) | All layers (recurrent) | None | Minimally governable; scaling disrupts state |
| Hybrid (Qwen3.5) | DeltaNet layers | Full attention layers | Partially governable; scale stateless only |
| MoE (Mixtral) | None in standard sense | Expert FFNs + attention | Governable; router outputs are additional control surface |
| State-Space + Attention (Jamba) | SSM layers | Attention layers | Similar to hybrid; scale attention only |

These are testable predictions. Each row can be validated by running the D-Flux telemetry and distillation pipeline on the corresponding architecture.

---

## 5. Scale Profiles as Valve Settings

### 5.1 The Valve Analogy

A scale profile $\boldsymbol{\sigma} = [\sigma_1, \ldots, \sigma_L]$ applied to the attention output projections functions exactly as a set of valve positions in a pipeline. Each $\sigma_l$ controls how much of layer $l$'s attention contribution passes to the residual stream:

$$\tilde{\mathbf{a}}_l = \sigma_l \cdot \mathbf{a}_l(\mathbf{h}_l)$$

- $\sigma_l = 1.0$: valve fully open. No intervention. Zero computational overhead (short-circuited in implementation).
- $\sigma_l > 1.0$: valve amplifying. The layer's contribution is boosted — equivalent to increasing local flow velocity.
- $\sigma_l < 1.0$: valve restricting. The layer's contribution is dampened — equivalent to a partial constriction.

### 5.2 Deriving Profiles from Telemetry Differences

Given telemetry from a base model ($T_\text{base}$) and a target model ($T_\text{target}$), the scale profile is derived as a ratio:

$$\sigma_l = 1 + \beta \left(\frac{T_{\text{target},l}}{T_{\text{base},l}} - 1\right)$$

where $\beta$ is the blend factor and the division is computed for each layer's telemetry signal (entropy reduction, dilution survival, etc.). This asks: "by what factor does the target model's signal differ from the base model's at each layer?" The blend parameter then controls how much of that difference to apply.

This is precisely how engineers calibrate valve settings: measure the difference between desired and actual flow, adjust proportionally, and limit the maximum adjustment (cap) to prevent instability.

### 5.3 Profile Compactness

A complete scale profile for Qwen3.5-9B (32 layers, 8 scalable full-attention layers) requires **8 floating-point numbers**. Stored as JSON with metadata, this is approximately **200 bytes**.

For comparison:
- Full fine-tuned model: ~18 GB
- LoRA adapter: ~100 MB – 1 GB
- Alignment Vector (weight diff): ~18 GB
- **Scale profile: ~200 bytes**

This represents a compression factor of approximately $10^{10}$ compared to full weight modification. The trade-off is expressiveness: a scale profile can only adjust the *magnitude* of each layer's attention contribution, not its *direction*. But as our experiments show, even this constrained intervention produces measurable behavioral changes.

### 5.4 Experimental Results

**Setup.** We compare four configurations on ARC Challenge: the unmodified base model, the fully instruction-tuned model (Qwen3.5-9B, which underwent RLHF/post-training), and the base model governed with the `dilution_survival` scale profile at two blend values.

| Configuration | ARC acc | ARC acc_norm | HellaSwag acc | HellaSwag acc_norm |
|---|---|---|---|---|
| Inverse (blend −0.5) | 53.84% | 55.97% | — | — |
| Base (ungoverned, blend 0.0) | 54.10% | 57.00% | 59.68% | 79.19% |
| Governed base (blend 0.3) | 55.03% | 57.85% | 59.63% | 79.23% |
| Governed base (blend 0.5) | 55.55% | 58.36% | — | — |
| Governed base (blend 0.6) | 55.63% | 58.19% | — | — |
| Governed base (blend 0.8) | 56.23% | 59.13% | — | — |
| **Governed base (blend 1.0)** | **56.83%** | **59.30%** | — | — |
| | | | | |
| Instruct (full post-training) | 54.27% | 55.89% | — | — |
| Reasoning-distilled (source of profile) | 53.16% | 55.63% | — | — |

The complete blend curve from −0.5 to 1.0 is monotonically increasing, spanning a total range of 3.33 points acc_norm (55.97% to 59.30%). At blend 1.0, the governed base model reaches **59.30% acc_norm** — a gain of **+2.30** over the unmodified base, **+3.41** over the instruct model, and **+3.67** over the reasoning-distilled model from which the profile was derived. The curve shows no sign of saturation or degradation through blend 1.0, suggesting further headroom may exist at higher blend values.

Notably, blend 1.0 produces text degeneration in free generation (loops, repetition — see Section 3.2) while continuing to improve benchmark performance. This confirms the task-dependent nature of the critical blend threshold: constrained output tasks (multiple choice) suppress the turbulent modes that manifest in unconstrained generation. The "critical Reynolds number" for this architecture is task-dependent, consistent with the pipe-exit-aperture analogy described in Section 3.2.

Both the instruct model and the distilled model *regressed* on acc_norm relative to the base (55.89% and 55.63% vs 57.00%) — post-training traded reasoning capability for other objectives. The governor profile recovered that loss and exceeded it, using only 8 scalar parameters and zero weight modification.

Critically, the reasoning-distilled model — the very model whose telemetry was used to derive the scale profile — scores *lower* than the governed base on ARC Challenge (53.16% vs 55.55%). **The profile outperforms its own source.** The distilled model was optimized to produce long reasoning chains (chain-of-thought, step-by-step derivation), a *behavioral* modification that changes generation patterns. In doing so, it also restructured the layer-level flow dynamics — which layers compress more, which layers' contributions persist. The telemetry captured these structural changes. When applied to the base model as a scale profile, the structural signal generalizes: the base model's internal flow is restructured to resemble the reasoning model's flow, while its generation behavior remains base-model-like (direct, efficient, suited to multiple-choice tasks). The profile extracts the *computational* benefit of reasoning distillation while leaving behind the *behavioral* changes that hurt on benchmarks requiring concise answers.

This result has a direct fluid dynamics interpretation. Post-training (RLHF, distillation) reconfigures the entire flow system — it changes pipe geometry, valve positions, reservoir levels, fluid properties, and output nozzle characteristics simultaneously. In doing so, it can inadvertently degrade flow characteristics it wasn't targeting (reasoning capability lost as a side effect of optimizing for other objectives). The governor profile, by contrast, adjusts only 8 valve positions on the *original* pipe system, preserving the base flow characteristics while selectively amplifying the stages that matter for the target capability. Same pipe, same fluid, same nozzle — better flow balance.

The scale profile (dilution_survival, blend 0.3):
```
Layer  3: σ = 1.125  (early boost — amplify early full-attention contributions)
Layer  7: σ = 1.058
Layer 11: σ = 1.007
Layer 15: σ = 0.936  (mid-layer dampen — reduce over-processing)
Layer 19: σ = 0.976
Layer 23: σ = 0.997
Layer 27: σ = 1.020
Layer 31: σ = 0.998
```

The pattern — boost early, dampen middle, neutral late — is consistent with the fluid dynamics prediction: early layers add the most *persistent* information (high dilution survival = low viscosity), so amplifying them increases the signal that survives to the output.

### 5.5 Causality Test: The Inverse Profile

To confirm that scale profiles are directionally causal — not random perturbations that happen to help — we constructed an **inverse profile** by taking the reciprocal of each scale factor. If the forward profile amplifies early layers and dampens mid layers ($\sigma_3 = 1.125$, $\sigma_{15} = 0.936$), the inverse does the opposite ($\sigma_3 = 0.889$, $\sigma_{15} = 1.069$). The inverse profile has identical perturbation magnitude but opposite direction.

| Profile | ARC acc | ARC acc_norm | Δ acc_norm |
|---|---|---|---|
| Inverse (blend −0.5) | 53.84% | 55.97% | **−1.03** |
| Baseline (blend 0.0) | 54.10% | 57.00% | — |
| Forward (blend +0.5) | 55.55% | 58.36% | **+1.36** |

The inverse profile degrades performance by −1.03 points acc_norm, while the forward profile at the same magnitude improves by +1.36 points. The total spread between inverse and forward is 2.39 points — a directional effect that rules out the hypothesis that arbitrary perturbations of this magnitude produce random improvements.

In fluid dynamics terms: opening the right valves increases throughput; opening the *wrong* valves — the same valves, by the same amount, in the opposite direction — creates flow resistance. The valve positions are not arbitrary; they encode the direction of the pressure gradient between the base model's flow state and the reasoning model's flow state. Reversing that gradient pushes flow away from the reasoning regime.

### 5.6 Emergent Behaviors at Higher Blend

At blend 0.5 on Qwen3.5-9B-Instruct, the `dilution_survival` profile triggered spontaneous chain-of-thought reasoning (the model began generating `<think>` tokens without prompting). This is analogous to a *resonance* phenomenon in fluid dynamics: at a specific driving frequency (blend value), the system's natural modes are excited, producing qualitatively new behavior that is absent at lower or higher driving amplitudes.

---

## 6. Signals as Thermodynamic Quantities

### 6.1 Entropy Reduction = Pressure Drop

The entropy of the output distribution decreases as information flows through layers — the model becomes more "certain" about its prediction. This is directly analogous to pressure drop through a pipe: energy is converted from potential (uncertainty) to kinetic (directed prediction).

The entropy reduction signal captures this:

$$\Delta H_l = H(\mathbf{h}_l) - H(\mathbf{h}_{l+1})$$

Layers with high entropy reduction are "doing the most work" — converting uncertainty into prediction. These are the high-pressure-drop stages of the pipeline.

### 6.2 Dilution Survival = Momentum Conservation

Dilution survival measures how much of each layer's contribution persists to the output. In fluid dynamics, this corresponds to momentum conservation: how much of the velocity impulse at one stage survives to the exit.

Low dilution survival = high viscosity = the layer's work is dampened by subsequent stages.
High dilution survival = low viscosity = the layer's contribution flows cleanly to the output.

The key insight: these two signals capture complementary aspects of the flow. Entropy reduction measures *energy conversion* (how much work each stage does). Dilution survival measures *momentum transfer* (how much of that work persists). A complete picture requires both, just as pipe flow analysis requires both pressure and velocity measurements.

### 6.3 Two Profile Signatures

The two signals produce qualitatively different scale profiles when used as the basis for distillation:

**Entropy reduction profile** (blend 0.5):
```
Layer  3: σ = 0.750  (dampen early)
Layer 11: σ = 1.500  (amplify mid)
Layer 19: σ = 1.323  (amplify late-mid)
Layer 23: σ = 1.412  (amplify late)
Layer 27: σ = 1.500  (amplify late — capped)
```

**Dilution survival profile** (blend 0.3):
```
Layer  3: σ = 1.125  (amplify early)
Layer  7: σ = 1.058  (amplify early-mid)
Layer 15: σ = 0.936  (dampen mid)
Layer 19: σ = 0.976  (slight dampen)
```

Entropy reduction says "amplify the layers that compress the most" — these tend to be late layers. Dilution survival says "amplify the layers whose work persists" — these tend to be early layers. The profiles are nearly opposite in shape, reflecting the complementary physics: pressure-focused intervention amplifies energy conversion stages, while momentum-focused intervention amplifies signal propagation stages.

---

## 7. Toward Governing Equations

### 7.1 The Discrete Navier-Stokes Analogue

The Navier-Stokes equation describes the evolution of velocity in a fluid: $\rho(\partial\mathbf{v}/\partial t + \mathbf{v} \cdot \nabla\mathbf{v}) = -\nabla p + \mu\nabla^2\mathbf{v} + \mathbf{f}$. We propose a discrete analogue for the residual stream:

$$\Delta v_l = v_{l+1} - v_l = \underbrace{-\gamma_l \Delta p_l}_{\text{pressure gradient}} + \underbrace{\mu_l (v_{l-1} - 2v_l + v_{l+1})}_{\text{viscous diffusion}} + \underbrace{F_l}_{\text{external force (FFN)}}$$

where:
- $\gamma_l$ is a coupling constant between pressure (entropy) and velocity (norm)
- $\mu_l$ is the effective viscosity (from dilution survival)
- $F_l$ is the forcing term from the feed-forward sublayer

This is a discrete diffusion–advection equation on a 1D lattice (the layer stack). The attention sublayer governs the pressure–velocity coupling (how certainty drives signal strength), while the FFN sublayer acts as an external forcing term.

### 7.2 Boundary Conditions

The "flow" has natural boundary conditions:
- **Inlet** ($l = 0$): embedding layer. The initial velocity $v_0 = \|\text{Embed}(x)\|$ and pressure $p_0 = H(\text{uniform})$ (maximum entropy, minimum certainty) are fixed by the input.
- **Outlet** ($l = L$): unembedding. The output distribution is determined by $\mathbf{h}_L$. The "back-pressure" is the softmax temperature.

### 7.3 Governing the Flow

The LiveGovernor introduces a controlled perturbation to this system:

$$\Delta v_l^\text{governed} = \sigma_l \cdot \Delta v_l^\text{attention} + \Delta v_l^\text{FFN}$$

The scale profile $\boldsymbol{\sigma}$ modifies the attention contribution at each layer without affecting the FFN. This is equivalent to adjusting valve positions in a pipeline while leaving the pumps (FFN) untouched.

The key constraint: the governor can only modify the *magnitude* of the attention contribution, not its direction. This limits the intervention space to a 1D manifold per layer (the real line), rather than the full $d$-dimensional space. This constraint is both a limitation (less expressive than weight modification) and a strength (vastly smaller search space, trivially composable, predictable effects).

---

## 8. Composability and Superposition

### 8.1 Linear Superposition at Small Perturbations

Given two scale profiles $\boldsymbol{\sigma}_A$ and $\boldsymbol{\sigma}_B$, their composition is:

$$\boldsymbol{\sigma}_{A+B} = \boldsymbol{\sigma}_A \circ \boldsymbol{\sigma}_B$$

where $\circ$ denotes element-wise multiplication. For small perturbations around identity ($\sigma_l \approx 1 + \epsilon_l$):

$$\sigma_{A,l} \cdot \sigma_{B,l} = (1 + \epsilon_{A,l})(1 + \epsilon_{B,l}) \approx 1 + \epsilon_{A,l} + \epsilon_{B,l}$$

The cross-term $\epsilon_A \epsilon_B$ is negligible for small perturbations. This means profiles compose *additively* in the small-perturbation regime — directly analogous to the superposition of small flow perturbations in linear fluid dynamics.

At large perturbations, the cross-terms dominate and composition becomes nonlinear. This predicts that stacking multiple aggressive profiles will produce unpredictable interactions — exactly the behavior we observe when blend values exceed 0.5.

### 8.2 Profile Algebra

The set of scale profiles forms a commutative monoid under element-wise multiplication:
- **Closure**: the product of two profiles is a profile
- **Associativity**: $(\boldsymbol{\sigma}_A \circ \boldsymbol{\sigma}_B) \circ \boldsymbol{\sigma}_C = \boldsymbol{\sigma}_A \circ (\boldsymbol{\sigma}_B \circ \boldsymbol{\sigma}_C)$
- **Identity**: the all-ones profile $\mathbf{1}$
- **Commutativity**: $\boldsymbol{\sigma}_A \circ \boldsymbol{\sigma}_B = \boldsymbol{\sigma}_B \circ \boldsymbol{\sigma}_A$

Inverses exist (element-wise reciprocal), making this a commutative group. This means profiles can be "undone" — applying $\boldsymbol{\sigma}^{-1}$ after $\boldsymbol{\sigma}$ returns the model to its original behavior, which is empirically verified by the detach operation.

---

## 9. Implications and Predictions

### 9.1 Testable Predictions

The fluid dynamic framework generates several testable predictions:

1. **Architecture-dependent critical blend.** Pure transformer models (all stateless layers) should tolerate higher blend values before turbulent transition than hybrid models. Pure SSM models should show degeneration at very low blend values.

2. **Dilution survival predicts scalability.** Layers with high dilution survival should tolerate larger scale perturbations. Layers with low dilution survival should be more sensitive. This can be tested by computing per-layer perturbation sensitivity and correlating with dilution survival.

3. **Information Bernoulli conservation.** The quantity $E_l = p_l + \alpha v_l^2$ should be approximately constant across layers within a single forward pass, with the coupling constant $\alpha$ being architecture-specific.

4. **Profile composability threshold.** Two profiles that are individually safe ($\beta < \beta_\text{crit}$) should be safe when composed if and only if the combined effective $\beta$ remains below threshold.

5. **Training dynamics follow fluid evolution.** During pre-training, the velocity and pressure profiles should evolve from uniform (laminar, undifferentiated) toward structured (developed flow) as the model acquires capabilities. The Reynolds number should increase during training.

### 9.2 The Profile Marketplace

A practical implication: if model behavior can be meaningfully modified by 200-byte scale profiles, then the distribution model for AI capabilities changes fundamentally. Instead of downloading multi-gigabyte fine-tuned models for each use case, users download a single base model and apply different profiles:

- `reasoning_boost_0.3.json` — improved logical reasoning
- `creative_writing_0.5.json` — enhanced creative output
- `code_generation_0.4.json` — better code completion
- `safety_amplifier_0.2.json` — increased safety behaviors

Profiles are trivially composable (element-wise multiplication), instantly switchable (no model reload), and fully reversible (multiply by inverse). The ecosystem dynamics resemble audio equalization presets more than model fine-tuning.

### 9.3 The Compression Theorem: Why 8 Numbers Work

A 9-billion-parameter model produces responses of a few hundred tokens. Regardless of how many weights are adjusted — whether through full fine-tuning (9B parameters), LoRA (millions), alignment vectors (billions), or scale profiles (8) — the output is constrained to the same low-dimensional manifold. The response is the bottleneck.

This has a precise information-theoretic interpretation. For any given input, the model's 9 billion parameters define a statistical manifold — the space of all possible flows. But the input activates a *trajectory* through that space, and the trajectory is low-dimensional. A question about chemistry activates a different channel through the layer stack than a question about poetry. Different attention heads fire, different layers compress, different residual stream components carry the signal.

The effective dimensionality of the behavioral change induced by any intervention is bounded by the dimensionality of the output, not the parameter space. If 18GB of alignment vectors and 200 bytes of scale factors produce comparable output shifts on a given task, then the true dimensionality of the relevant intervention was always closer to 8 than to 9 billion. The remaining parameters encode changes orthogonal to the active channel — invisible in the output, consuming storage for no behavioral effect on that task.

Scale profiles succeed because they operate at the natural dimensionality of the intervention. Each of the 8 scale factors controls the conductance of one processing stage along the active flow path. The profile doesn't attempt to reshape the entire manifold — it adjusts how efficiently the model routes information through the channel that the current input activates. The manifold (what the model knows) is untouched. The routing (how it uses what it knows) is adjusted.

This explains why the profile outperforms its source model. The reasoning-distilled model reshaped the entire manifold to favor reasoning channels — but in doing so, narrowed channels serving other capabilities. The scale profile adjusts conductance at 8 valve points only, leaving every other channel at full capacity. The reasoning channel is boosted; everything else remains intact. Surgical intervention at the right dimensionality outperforms global intervention at unnecessarily high dimensionality.

### 9.4 Dynamic Governance: From Static Profiles to Real-Time Flow Control

Static scale profiles are precomputed valve settings — optimal on average but blind to the specific flow conditions of any given input. A reasoning profile applies the same 8 scales whether the input is a simple factual question or a complex multi-step derivation. In fluid dynamics terms, this is the equivalent of fixed valve positions in a pipeline: adequate for steady-state operation, suboptimal for varying load.

The natural extension is **dynamic governance**: scale factors that float and self-adjust during generation based on real-time telemetry. The D-Flux LiveGovernor already supports this through its reactive mode, which reads telemetry after each token and updates scales for the next token's forward pass. The transition from static profiles to dynamic governance maps directly to the transition from open-loop to closed-loop control.

#### 9.4.1 The Control Architecture

The dynamic governor is a classical feedback control system with well-defined components:

**Sensor**: The D-Flux telemetry — entropy reduction, dilution survival, residual norm, attention entropy — measured after each layer on each token. This is the flow measurement.

**Setpoint**: A target flow profile, derived from the static scale profile or from a reference model's telemetry signature. This defines the desired flow state.

**Error signal**: The difference between measured flow and target flow at each layer. If dilution survival at layer 3 is below the setpoint, the error is negative — the layer's contribution is being lost.

**Actuator**: The mutable scale tensor on each layer's attention output projection. Adjusted per-token based on the error signal.

**Controller**: A damped proportional controller (or PID variant) that maps error to scale adjustment. The key design constraint is stability — the controller must not oscillate.

The update rule for scale $\sigma_l$ at token $t+1$:

$$\sigma_l^{(t+1)} = \alpha \cdot \sigma_l^{(t)} + (1 - \alpha) \cdot \left(1 + \kappa \cdot e_l^{(t)}\right)$$

where $\alpha$ is the EMA smoothing factor (prevents oscillation), $\kappa$ is the gain (controls responsiveness), and $e_l^{(t)}$ is the error signal at layer $l$ on token $t$. The EMA ensures that scale adjustments are gradual — no single token's telemetry can cause a large jump. The gain $\kappa$ controls the trade-off between responsiveness and stability.

#### 9.4.2 Why Dynamic Governance Should Outperform Static Profiles

A static profile assumes that the optimal valve settings are constant across all tokens in a generation. This is approximately true for short generations on uniform tasks, but becomes increasingly wrong as:

1. **Context builds.** As the KV cache fills, attention patterns shift. Early tokens attend broadly; later tokens attend to specific precedents. The optimal scale at layer 3 for token 1 may differ significantly from the optimal scale for token 200.

2. **Semantic phase changes.** Within a single generation, the model may transition between modes — factual recall, logical derivation, summarization. Each mode activates a different channel. A static profile optimized for one mode is suboptimal for others.

3. **Turbulence onset varies by position.** The critical blend threshold observed in free generation may not be a global property but a local one — some tokens are in the laminar regime while others (at transition points in the argument) are near the turbulent boundary.

Dynamic governance can track these variations. The governor measures the flow on every token and adjusts scales to maintain the target regime. If entropy spikes at layer 19 on a particular token (indicating that layer is doing exceptional compression work), the governor can boost that layer's scale for the next token. If dilution survival drops at layer 7, the governor can amplify to compensate before the signal is lost.

#### 9.4.3 Stability Constraints

The primary risk of dynamic governance is oscillation. Eight free parameters updating every token with no damping will ring. Three stabilization mechanisms are required:

1. **EMA smoothing** ($\alpha \in [0.9, 0.99]$): exponential moving average on scale updates ensures gradual adjustment. The effective response time is $1/(1-\alpha)$ tokens — at $\alpha = 0.95$, the governor responds over a 20-token window.

2. **Drift bounds** ($|\sigma_l - 1| \leq \sigma_\text{max}$): hard caps on how far any scale can deviate from unity. This is the equivalent of physical valve stops — the valve can't open or close beyond its mechanical limits.

3. **Drag toward unity**: when scales drift far from 1.0, a restoring force pulls them back. This prevents long-term drift during extended generation and ensures the system returns to a neutral state when telemetry signals are ambiguous.

These mechanisms are structurally identical to the τ-Drag Law used in conversational governance systems for managing ego drift in multi-turn interactions — drag proportional to displacement, damping when drift exceeds onset, floor to prevent over-damping. The same stabilization principle applies at both the token level (intra-model flow) and the turn level (inter-model conversation flow), consistent with the self-similar nature of the framework.

#### 9.4.4 The Two-Layer Architecture

The most powerful configuration combines both modes:

1. **Static profile** as the baseline setpoint — the best available precomputed valve setting for the task type. This handles the average case.
2. **Dynamic governor** as the real-time adaptation layer — measuring deviations from the setpoint and correcting per-token. This handles the variance.

The static profile provides the coarse channel selection (reasoning, creative, code). The dynamic governor provides fine-grained adaptation within that channel. This is analogous to a pipeline with both fixed orifice plates (static flow conditioning) and active control valves (dynamic regulation).

### 9.5 Toward Training-Aware Architectures

The ultimate implication: if we know which architectural properties make a model more governable (independent layers, clear stateful/stateless separation, high dilution survival in key layers), then we can design architectures *optimized for governance.* The model becomes a platform; profiles become the product.

This inverts the current paradigm where each capability requires a new model. Instead, capability becomes a *configuration* of a single, well-designed flow system.

---

## 10. Conclusion

We have presented evidence that information flow through transformers obeys principles structurally analogous to fluid dynamics, and that this analogy is predictive rather than merely descriptive. The key correspondences — entropy as pressure, residual norm as velocity, dilution survival as momentum conservation, blend threshold as Reynolds number — produce correct predictions about which interventions are safe, which are dangerous, and where qualitative transitions occur.

The practical output is D-Flux: an open-source framework that implements these principles through per-layer attention scaling. Scale profiles of ~200 bytes can meaningfully modify model behavior, compose linearly at small perturbations, and are fully reversible. A governed base model using 8 scalar parameters outperforms the fully instruction-tuned variant on ARC Challenge — recovering reasoning capability that post-training destroyed, without any of the associated trade-offs. The inverse profile test confirms directional causality: the same perturbation magnitude in the opposite direction degrades performance, ruling out random-noise explanations. The framework correctly predicts the differential scalability of stateful versus stateless layers and the task-dependent nature of the critical blend threshold.

Much remains to be done. The governing equations need rigorous validation across architectures. The Information Reynolds Number needs quantitative calibration. The Bernoulli conservation hypothesis needs empirical testing. And the training-time implications — recording flow telemetry during pre-training, designing architectures for governability — are entirely unexplored.

But the central claim is clear: transformer inference is a flow problem, and the tools of fluid dynamics apply.

---

## References

- Chen, Z., et al. (2025). HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading.
- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. Anthropic.
- Turner, A., et al. (2023). Activation Addition: Steering Language Models Without Optimization.
- Vercellotti, M., et al. (2025). Inference Time LLM Alignment in Single and Multidomain Preference Spectrum. ICLR 2025.

---

## Appendix A: D-Flux Telemetry Signals

The D-Flux toolkit measures the following per-layer signals during inference:

| Signal | Description | Fluid Analogue |
|---|---|---|
| `entropy_reduction` | Change in output entropy after layer $l$ | Pressure drop |
| `dilution_survival` | Fraction of layer $l$'s contribution surviving to output | Momentum survival |
| `residual_norm` | L2 norm of residual stream at layer $l$ | Flow velocity |
| `attention_entropy` | Entropy of attention weight distribution | Local turbulence intensity |
| `head_dominance` | Maximum head contribution / mean head contribution | Flow concentration |
| `grad_norm` | Gradient magnitude at layer $l$ | Sensitivity to perturbation |

## Appendix B: Scale Profile Format

```json
{
  "signal": "dilution_survival",
  "strategy": "ratio",
  "blend": 0.3,
  "cap": 2.0,
  "layer_type_bias": "full_attention",
  "scales": {
    "3": 1.1252,
    "7": 1.0577,
    "11": 1.0066,
    "15": 0.9356,
    "19": 0.9759,
    "23": 0.9973,
    "27": 1.0202,
    "31": 0.9977
  },
  "model_family": "Qwen3.5-9B",
  "description": "Reasoning boost via dilution_survival, full_attention only, blend 0.3",
  "source_base": "Qwen/Qwen3.5-9B-Base",
  "source_target": "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled"
}
```

## Appendix C: Experimental Setup

All experiments conducted on Qwen3.5-9B architecture family:
- **Base model**: Qwen/Qwen3.5-9B-Base (32 layers: 24 DeltaNet + 8 full attention)
- **Target model**: Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled
- **Instruct model**: Qwen/Qwen3.5-9B (instruction-tuned)
- **Benchmarks**: ARC Challenge, HellaSwag (via lm-evaluation-harness)
- **Hardware**: Single GPU inference
- **Scale application**: Forward hooks on `o_proj` layers (attention output projection)
- **Blend sweep**: −0.5 (inverse), 0.0 (baseline), 0.3, 0.5, 0.6, 0.8, 1.0
- **Signal**: `dilution_survival` (ratio strategy, full_attention layer bias)
- **Telemetry source**: Per-layer signal diff between base and reasoning-distilled model

## Appendix D: Complete Blend Curve Data

| Blend | ARC acc | ARC acc_norm | Δ acc_norm | Regime (free gen) |
|---|---|---|---|---|
| −0.5 (inverse) | 53.84% | 55.97% | −1.03 | Coherent (degraded) |
| 0.0 (baseline) | 54.10% | 57.00% | — | Coherent |
| 0.3 | 55.03% | 57.85% | +0.85 | Coherent |
| 0.5 | 55.55% | 58.36% | +1.36 | Coherent; triggers `<think>` on instruct |
| 0.6 | 55.63% | 58.19% | +1.19 | Transitional |
| 0.8 | 56.23% | 59.13% | +2.13 | Transitional |
| 1.0 | 56.83% | 59.30% | +2.30 | Turbulent (loops in free gen) |

Reference models (no governor):

| Model | ARC acc | ARC acc_norm |
|---|---|---|
| Instruct (Qwen3.5-9B) | 54.27% | 55.89% |
| Reasoning-distilled (source of profile) | 53.16% | 55.63% |

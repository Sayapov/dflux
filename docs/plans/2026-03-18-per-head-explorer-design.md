# Per-Head Attention Explorer Design

## Goal
Observe per-head attention behavior across 256 heads (8 full-attention layers x 32 heads) in Qwen3.5-9B-Base. Understand head specialization before building a per-head governor.

## Phase Plan
1. **Phase A (this doc)**: Capture and visualize per-head patterns across task types
2. **Phase B (later)**: Telemetry diff — per-head norms between base and distilled model, derive 256-parameter profile
3. **Phase C (if needed)**: Ablation — zero out individual heads, measure benchmark impact

## Approach
Run 4 prompts spanning different capability domains through the base model. Hook into attention layers to capture per-head output norms and attention entropy before concatenation/projection.

## Prompts
1. **Factual retrieval**: "What is the capital of France and when was it founded?"
2. **Math reasoning**: "If a train travels 120km in 2 hours, then 180km in 3 hours, what is its average speed?"
3. **Creative**: "Write a short poem about a robot learning to dance"
4. **Multi-step reasoning**: "What is the temperature on the far side of the moon, and what would happen to a cucumber if it suddenly appeared there."

## Capture (per token, per head)
- **Attention output norm**: L2 norm of each head's output vector (energy contribution)
- **Attention entropy**: Shannon entropy of each head's attention weight distribution (focus vs diffuse)

## Hook Point
Qwen3.5 full-attention layers use standard multi-head attention. Each head produces a [seq_len, head_dim] output. We hook after the attention computation but before concatenation + o_proj, capturing the per-head tensors.

For Qwen3.5 architecture:
- 32 heads per full-attention layer
- head_dim = hidden_size / num_heads = 4096 / 32 = 128
- Full attention at layers: 3, 7, 11, 15, 19, 23, 27, 31

## Output
1. **Console heatmaps**: 8x32 grid per prompt showing head norms (ASCII art)
2. **Delta analysis**: Which heads vary most across prompts (specialization)
3. **Stability analysis**: Which heads fire consistently (infrastructure)
4. **JSON dump**: Raw data for later analysis

## Implementation
Single script: `examples/per_head_explorer.py`
- Loads Qwen3.5-9B-Base with key remapping (same as chat.py)
- Installs per-head capture hooks on 8 full-attention layers
- Runs 4 prompts with limited generation (128 tokens each for speed)
- Aggregates per-head stats across tokens
- Prints analysis and saves JSON to data/per_head/

## No governor — pure observation.

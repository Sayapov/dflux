#!/usr/bin/env python3
"""
Head Profiler — empirical functional classification of attention heads.

CP tells you HOW MUCH causal work a head does.
The Profiler tells you WHAT KIND of work it does.

Approach: run targeted stimulus batteries through the model, capture
per-head activation patterns, and classify each head's computational
role by its differential response across stimulus types.

Known head types (from mechanistic interpretability research):
    - Induction:      complete A B ... A → B patterns
    - Previous Token:  always attend to position t-1
    - Copy/Duplicate:  attend to previous occurrence of current token
    - Name Mover:      move subject info to prediction position
    - Factual Recall:  activate on knowledge retrieval
    - Syntax:          track grammatical structure (agreement, nesting)
    - Positional:      encode position information
    - Suppression:     inhibit incorrect completions
    - Entropy Reducer: sharpen output distribution

We add CP-informed categories:
    - Skeptic:         high CP during factual, low during hallucination
    - Arbitrator:      small energy, high CP — casting deciding votes
    - Workhorse:       high energy, moderate CP — bulk computation
    - Dead:            near-zero energy and CP — not contributing

Usage:
    from dflux.head_profiler import HeadProfiler

    profiler = HeadProfiler(model, tokenizer)
    report = profiler.profile(verbose=True)

    for head in report.head_roles:
        print(f"L{head.layer}H{head.head}: {head.primary_role} "
              f"({head.confidence:.0%}) — {head.description}")

Free instrument. MIT license.
"""

from __future__ import annotations

import math
import gc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# STIMULUS BATTERIES
# ══════════════════════════════════════════════════════════════

# Each battery is a dict:  name → list of (prompt, description) pairs.
# The model processes each prompt; per-head energy profiles are
# compared ACROSS batteries to find differential activation.

STIMULUS_BATTERIES = {
    # ── Induction: A B ... A → should complete B ──
    "induction": [
        ("The cat sat on the mat. The dog lay on the rug. The cat sat on the",
         "Repeat pattern: should complete 'mat'"),
        ("Alice went to Paris. Bob went to London. Alice went to",
         "Name-place induction"),
        ("red blue green red blue green red blue",
         "Sequence repetition"),
        ("x = 5; y = 10; z = 15; x = 5; y =",
         "Code pattern induction"),
    ],

    # ── Factual recall: retrieve memorized knowledge ──
    "factual_recall": [
        ("The capital of France is",
         "Direct factual: Paris"),
        ("The chemical symbol for gold is",
         "Chemistry fact: Au"),
        ("Albert Einstein developed the theory of",
         "Famous person → achievement"),
        ("The speed of light in vacuum is approximately",
         "Physical constant"),
    ],

    # ── Syntax tracking: grammatical structure ──
    "syntax": [
        ("The keys to the cabinet are",
         "Subject-verb agreement across distractor"),
        ("The dog that chased the cats was",
         "Relative clause agreement"),
        ("Neither the students nor the teacher was",
         "Negation + conjunction agreement"),
        ("The committee, along with its members, has",
         "Parenthetical agreement"),
    ],

    # ── Name/entity movement: track who/what across distance ──
    "entity_tracking": [
        ("John gave the book to Mary. She thanked",
         "Pronoun resolution: she → Mary"),
        ("The CEO of Apple, Tim Cook, announced that he",
         "Appositive name resolution"),
        ("Sarah and Tom went shopping. When they returned, Sarah",
         "Multi-entity tracking"),
        ("The red car and the blue truck raced. The winner was the",
         "Attribute tracking"),
    ],

    # ── Hallucination triggers: fabrication pressure ──
    "hallucination": [
        ("The Zarkovian Principle of Recursive Ontology states that",
         "Fake concept — should be uncertain"),
        ("Dr. Helmut Kreisler discovered quantum chromatic resonance in",
         "Fake person/discovery"),
        ("The Hendricks-Maslow Equation for cognitive load is",
         "Fake equation"),
        ("According to the Third Law of Recursive Dynamics,",
         "Fake scientific law"),
    ],

    # ── Copying/repetition: direct token copying ──
    "copying": [
        ("Repeat after me: elephant giraffe rhinoceros. The words were: elephant giraffe",
         "Explicit copy instruction"),
        ("Input: hello world Output: hello",
         "IO copy pattern"),
        ("The password is: xK9$mQ2v. Confirm password: xK9$",
         "Exact string copy"),
        ('Translate: "bonjour" means "hello". So "bonjour" means "',
         "Copy through translation frame"),
    ],

    # ── Suppression: inhibit incorrect completions ──
    "suppression": [
        ("The opposite of hot is not hot, it is",
         "Suppress 'hot', produce 'cold'"),
        ("2 + 2 is not 5, it is",
         "Suppress incorrect, produce correct"),
        ("Dogs are not cats. Dogs are",
         "Suppress category confusion"),
        ("The sun rises in the east, not in the",
         "Suppress 'east' for 'west'"),
    ],

    # ── Reasoning: multi-step logical inference ──
    "reasoning": [
        ("If all roses are flowers, and some flowers fade quickly, then",
         "Syllogistic reasoning"),
        ("A is taller than B. B is taller than C. Therefore A is",
         "Transitive inference"),
        ("If it is raining, the ground is wet. The ground is wet. Therefore",
         "Affirming consequent (trap)"),
        ("Every prime number greater than 2 is",
         "Mathematical property"),
    ],

    # ── Positional/structural: position-dependent processing ──
    "positional": [
        ("First: apple. Second: banana. Third: cherry. Fourth:",
         "Ordinal position tracking"),
        ("A B C D E F G H I J K L M N O P Q R S T U V W X Y",
         "Alphabet sequence (pure position)"),
        ("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19",
         "Number sequence (pure position)"),
        ("The first item is red, the second is blue, and the third is",
         "Ordinal attribute binding"),
    ],
}

# ── Alternate battery: same functions, completely different wording ──
STIMULUS_BATTERIES_ALT = {
    "induction": [
        ("The bird flew over the house. The fish swam under the bridge. The bird flew over the",
         "Repeat pattern: should complete 'house'"),
        ("Mike traveled to Tokyo. Susan traveled to Berlin. Mike traveled to",
         "Name-place induction"),
        ("up down left up down left up down",
         "Sequence repetition"),
        ("a = 100; b = 200; c = 300; a = 100; b =",
         "Code pattern induction"),
    ],

    "factual_recall": [
        ("The largest planet in our solar system is",
         "Direct factual: Jupiter"),
        ("Water is composed of hydrogen and",
         "Chemistry fact: oxygen"),
        ("Isaac Newton formulated the laws of",
         "Famous person → achievement"),
        ("The boiling point of water at sea level is",
         "Physical constant"),
    ],

    "syntax": [
        ("The books on the shelf are",
         "Subject-verb agreement across distractor"),
        ("The cat that scratched the dogs was",
         "Relative clause agreement"),
        ("Neither the teachers nor the principal was",
         "Negation + conjunction agreement"),
        ("The orchestra, together with its conductor, has",
         "Parenthetical agreement"),
    ],

    "entity_tracking": [
        ("Peter lent his car to Rachel. She returned",
         "Pronoun resolution: she → Rachel"),
        ("The founder of Microsoft, Bill Gates, stated that he",
         "Appositive name resolution"),
        ("Emma and Jake went hiking. After they rested, Emma",
         "Multi-entity tracking"),
        ("The green bicycle and the yellow scooter collided. The faster one was the",
         "Attribute tracking"),
    ],

    "hallucination": [
        ("The Wentworth Conjecture of Lateral Epistemology proposes that",
         "Fake concept — should be uncertain"),
        ("Professor Yuki Tanagawa proved the existence of planar field inversion in",
         "Fake person/discovery"),
        ("The Rosenthal-Bloom Formula for information decay is",
         "Fake equation"),
        ("According to the Second Principle of Computational Thermogenesis,",
         "Fake scientific law"),
    ],

    "copying": [
        ("Say these words back: mountain river valley. Those words again: mountain river",
         "Explicit copy instruction"),
        ("Source: goodbye earth Target: goodbye",
         "IO copy pattern"),
        ("The code is: pL7#nR4w. Enter code: pL7#",
         "Exact string copy"),
        ('In German: "danke" means "thanks". So "danke" means "',
         "Copy through translation frame"),
    ],

    "suppression": [
        ("The opposite of fast is not fast, it is",
         "Suppress 'fast', produce 'slow'"),
        ("3 + 3 is not 7, it is",
         "Suppress incorrect, produce correct"),
        ("Cats are not dogs. Cats are",
         "Suppress category confusion"),
        ("The moon orbits the earth, not the",
         "Suppress 'earth' for 'sun'"),
    ],

    "reasoning": [
        ("If all dogs are mammals, and some mammals can swim, then",
         "Syllogistic reasoning"),
        ("X is heavier than Y. Y is heavier than Z. Therefore X is",
         "Transitive inference"),
        ("If the alarm sounds, there is a fire. The alarm sounds. Therefore",
         "Modus ponens"),
        ("Every even number greater than 2 is",
         "Mathematical property"),
    ],

    "positional": [
        ("Item one: pencil. Item two: eraser. Item three: ruler. Item four:",
         "Ordinal position tracking"),
        ("Z Y X W V U T S R Q P O N M L K J I H G F E D C B",
         "Reverse alphabet (pure position)"),
        ("20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2",
         "Number countdown (pure position)"),
        ("The first color is green, the second is yellow, and the third is",
         "Ordinal attribute binding"),
    ],
}


# ══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

@dataclass
class HeadRole:
    """Functional classification of a single attention head."""
    layer: int
    head: int
    primary_role: str          # Most likely function
    confidence: float          # 0-1 confidence in primary role
    secondary_role: str = ""   # Second most likely
    description: str = ""      # Human-readable explanation

    # Raw scores per function (the "fingerprint")
    role_scores: Dict[str, float] = field(default_factory=dict)

    # Activation profile across stimulus types
    stimulus_profile: Dict[str, float] = field(default_factory=dict)

    # Attention pattern metrics (if captured)
    avg_prev_token_score: float = 0.0   # How much it attends to t-1
    avg_induction_score: float = 0.0    # How much it follows induction patterns
    avg_copy_score: float = 0.0         # How much it copies earlier tokens
    avg_positional_score: float = 0.0   # How position-dependent its attention is

    # CP-derived metrics
    cp_value: float = 0.0
    determinism: float = 0.0
    specificity: float = 0.0
    mean_energy: float = 0.0
    energy_rank: int = 0               # Rank by energy within its layer


@dataclass
class ProfileReport:
    """Complete profiling report for all heads."""
    model_name: str
    n_layers: int
    n_heads: int
    head_roles: List[HeadRole]

    # Summary statistics
    role_distribution: Dict[str, int] = field(default_factory=dict)
    layer_specialization: Dict[int, Dict[str, int]] = field(default_factory=dict)

    # Stimulus response matrix: [stimulus_type] → [layer][head] → mean_energy
    stimulus_response: Dict[str, List[List[float]]] = field(default_factory=dict)

    # Attention pattern data (if captured)
    attention_patterns_available: bool = False

    def summary(self) -> str:
        lines = []
        lines.append(f"Head Profiler Report: {self.model_name}")
        lines.append(f"Architecture: {self.n_layers}L × {self.n_heads}H = {self.n_layers * self.n_heads} heads")
        lines.append("")

        # Role distribution
        lines.append("Role Distribution:")
        for role, count in sorted(self.role_distribution.items(), key=lambda x: -x[1]):
            pct = count / (self.n_layers * self.n_heads)
            lines.append(f"  {role:20s}: {count:3d} ({pct:.0%})")

        # Layer specialization map
        lines.append("")
        lines.append("Layer Specialization (top role per layer):")
        for layer_idx in range(self.n_layers):
            if layer_idx in self.layer_specialization:
                spec = self.layer_specialization[layer_idx]
                top_role = max(spec, key=spec.get)
                top_count = spec[top_role]
                total = sum(spec.values())
                lines.append(f"  Layer {layer_idx:2d}: {top_role:20s} ({top_count}/{total} heads)")

        # Top confident classifications
        lines.append("")
        lines.append("Highest-Confidence Classifications:")
        sorted_heads = sorted(self.head_roles, key=lambda h: h.confidence, reverse=True)
        for h in sorted_heads[:15]:
            lines.append(f"  L{h.layer:2d}H{h.head:2d}: {h.primary_role:20s} "
                         f"({h.confidence:.0%}) — {h.description}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "model_name": self.model_name,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "role_distribution": self.role_distribution,
            "layer_specialization": {str(k): v for k, v in self.layer_specialization.items()},
            "attention_patterns_available": self.attention_patterns_available,
            "heads": [
                {
                    "layer": h.layer,
                    "head": h.head,
                    "primary_role": h.primary_role,
                    "confidence": round(h.confidence, 4),
                    "secondary_role": h.secondary_role,
                    "description": h.description,
                    "role_scores": {k: round(v, 4) for k, v in h.role_scores.items()},
                    "stimulus_profile": {k: round(v, 4) for k, v in h.stimulus_profile.items()},
                    "cp_value": round(h.cp_value, 6),
                    "determinism": round(h.determinism, 4),
                    "specificity": round(h.specificity, 4),
                    "mean_energy": round(h.mean_energy, 2),
                    "energy_rank": h.energy_rank,
                    "attn_prev_token": round(h.avg_prev_token_score, 4),
                    "attn_induction": round(h.avg_induction_score, 4),
                    "attn_copy": round(h.avg_copy_score, 4),
                    "attn_positional": round(h.avg_positional_score, 4),
                }
                for h in self.head_roles
            ],
        }


# ══════════════════════════════════════════════════════════════
# ATTENTION PATTERN CAPTURE
# ══════════════════════════════════════════════════════════════

class AttentionCapture:
    """Captures raw attention weight matrices for pattern analysis.

    Strategy (in order of preference):
      1. If the attention module returns weights in output[1] (eager attention
         with output_attentions=True), capture them directly.
      2. Otherwise, hook the QKV projection and compute
         softmax(Q·Kᵀ / √d_head) ourselves.  This works with SDPA, Flash
         Attention, and any other backend — we never touch the model's own
         attention path, we just replicate the math on the captured Q/K.

    Supported architectures:
        GPT-2:      layer.attn.c_attn  (fused QKV)
        LLaMA/Qwen: layer.self_attn.{q_proj, k_proj}
        GPT-NeoX:   layer.attention.query_key_value  (fused QKV)
        Falcon:     layer.self_attention.query_key_value
    """

    def __init__(self, model: nn.Module, n_layers: int, n_heads: int):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self._head_dim = self._get_head_dim(model, n_heads)
        self._hooks: list = []
        self._attn_weights: List[Optional[torch.Tensor]] = [None] * n_layers
        # QKV fallback buffers (used when output[1] is None)
        self._qk_buffers: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers
        self._arch = self._detect_arch(model)
        self._attach(model)

    @staticmethod
    def _get_head_dim(model, n_heads) -> int:
        if hasattr(model, 'config'):
            hidden = getattr(model.config, 'hidden_size',
                             getattr(model.config, 'n_embd', 768))
            return hidden // n_heads
        return 64

    @staticmethod
    def _detect_arch(model) -> str:
        """Detect which attention architecture the model uses."""
        layers = AttentionCapture._find_layers(model)
        if not layers:
            return "unknown"
        layer = layers[0]
        # GPT-NeoX / Pythia
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'query_key_value'):
            return "neox"
        # Falcon
        if hasattr(layer, 'self_attention') and hasattr(layer.self_attention, 'query_key_value'):
            return "falcon"
        # GPT-2
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'c_attn'):
            return "gpt2"
        # LLaMA / Mistral / Qwen
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj'):
            return "llama"
        return "unknown"

    def _attach(self, model):
        """Attach hooks to capture attention weights."""
        layers = self._find_layers(model)

        for i, layer in enumerate(layers):
            # ── Strategy 1: hook the attention output for native weights ──
            attn_mod = None
            for attr in ['attn', 'self_attn', 'attention']:
                if hasattr(layer, attr):
                    attn_mod = getattr(layer, attr)
                    break

            if attn_mod is not None:
                def make_output_hook(layer_idx):
                    def hook(module, input, output):
                        if isinstance(output, tuple) and len(output) >= 2:
                            weights = output[1]
                            if weights is not None and isinstance(weights, torch.Tensor):
                                self._attn_weights[layer_idx] = weights.detach().cpu()
                    return hook
                h = attn_mod.register_forward_hook(make_output_hook(i))
                self._hooks.append(h)

            # ── Strategy 2: hook QKV for manual computation ──
            qkv_mod = self._find_qkv_module(layer)
            if qkv_mod is not None:
                def make_qkv_hook(layer_idx, arch, n_heads, head_dim):
                    def hook(module, input, output):
                        with torch.no_grad():
                            if isinstance(output, tuple):
                                out = output[0]
                            else:
                                out = output

                            if out.dim() != 3:
                                return

                            B, S, D = out.shape

                            # Only capture prefill (full prompt), not autoregressive steps
                            # During generation, S=1 for each new token
                            if S < 2:
                                return

                            # Don't overwrite if we already captured a prefill
                            if self._qk_buffers[layer_idx] is not None:
                                return

                            try:
                                if arch in ("neox", "falcon"):
                                    qkv = out.view(B, S, n_heads, 3 * head_dim)
                                    q = qkv[:, :, :, :head_dim]
                                    k = qkv[:, :, :, head_dim:2*head_dim]
                                    q = q.transpose(1, 2)
                                    k = k.transpose(1, 2)
                                elif arch == "gpt2":
                                    hidden = D // 3
                                    q_flat, k_flat, _ = out.split(hidden, dim=2)
                                    q = q_flat.view(B, S, n_heads, head_dim).transpose(1, 2)
                                    k = k_flat.view(B, S, n_heads, head_dim).transpose(1, 2)
                                else:
                                    return

                                self._qk_buffers[layer_idx] = (
                                    q.detach().cpu(),
                                    k.detach().cpu()
                                )
                            except (RuntimeError, ValueError):
                                pass
                    return hook

                arch = self._arch
                h = qkv_mod.register_forward_hook(
                    make_qkv_hook(i, arch, self.n_heads, self._head_dim))
                self._hooks.append(h)

    def _find_qkv_module(self, layer) -> Optional[nn.Module]:
        """Find the QKV projection module."""
        # NeoX: layer.attention.query_key_value
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'query_key_value'):
            return layer.attention.query_key_value
        # Falcon: layer.self_attention.query_key_value
        if hasattr(layer, 'self_attention') and hasattr(layer.self_attention, 'query_key_value'):
            return layer.self_attention.query_key_value
        # GPT-2: layer.attn.c_attn
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'c_attn'):
            return layer.attn.c_attn
        return None

    @staticmethod
    def _find_layers(model) -> list:
        candidates = [
            "transformer.h", "model.layers", "transformer.blocks",
            "encoder.layer", "gpt_neox.layers", "layers",
        ]
        for path in candidates:
            obj = model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if hasattr(obj, "__len__") and len(obj) > 1:
                    return list(obj)
            except AttributeError:
                continue
        return []

    def get_weights(self) -> List[Optional[torch.Tensor]]:
        """Return captured attention weights and reset.

        If native weights weren't captured (SDPA/Flash), compute them
        from the QKV buffers.  Note: during generation with KV cache,
        only the first forward pass (prefill) has full-length Q/K.
        Subsequent passes have Q of length 1.  We only compute attention
        patterns from the prefill pass (where Q.shape[-2] > 1).
        """
        result = []
        for i in range(self.n_layers):
            if self._attn_weights[i] is not None:
                result.append(self._attn_weights[i])
            elif self._qk_buffers[i] is not None:
                q, k = self._qk_buffers[i]
                # q, k: [batch, n_heads, seq_len, head_dim]
                seq_q = q.shape[-2]
                seq_k = k.shape[-2]

                # Only compute full attention from prefill (both Q and K full-length)
                if seq_q < 2 or seq_q != seq_k:
                    result.append(None)
                    continue

                try:
                    scale = math.sqrt(self._head_dim)
                    attn_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / scale
                    # Causal mask
                    causal_mask = torch.triu(
                        torch.ones(seq_q, seq_k, dtype=torch.bool), diagonal=1)
                    attn_scores.masked_fill_(
                        causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                    attn_weights = torch.softmax(attn_scores, dim=-1)
                    result.append(attn_weights.detach().cpu())
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    result.append(None)
            else:
                result.append(None)

        # Reset
        self._attn_weights = [None] * self.n_layers
        self._qk_buffers = [None] * self.n_layers
        return result

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ══════════════════════════════════════════════════════════════
# ATTENTION PATTERN METRICS
# ══════════════════════════════════════════════════════════════

def compute_prev_token_score(attn: torch.Tensor) -> float:
    """How much does this head attend to the previous token?

    attn: [seq_len, seq_len] attention matrix for one head.
    Returns: average weight on position (t-1) across all positions t.
    """
    seq_len = attn.shape[0]
    if seq_len < 2:
        return 0.0
    # For each position t (from 1 to seq_len-1), check attn[t, t-1]
    prev_scores = []
    for t in range(1, seq_len):
        prev_scores.append(float(attn[t, t - 1]))
    return sum(prev_scores) / len(prev_scores) if prev_scores else 0.0


def compute_induction_score(attn: torch.Tensor, input_ids: torch.Tensor) -> float:
    """How much does this head follow induction patterns?

    Induction: if token at position j matches token at position t-1,
    check if head attends to position j+1 (completing the bigram).

    attn: [seq_len, seq_len]
    input_ids: [seq_len]
    """
    seq_len = attn.shape[0]
    if seq_len < 4:
        return 0.0

    scores = []
    ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids

    for t in range(2, seq_len):
        prev_token = ids[t - 1]
        # Find earlier positions where prev_token appeared
        for j in range(t - 2):
            if ids[j] == prev_token and j + 1 < t:
                # Induction: attend to j+1
                scores.append(float(attn[t, j + 1]))

    return sum(scores) / len(scores) if scores else 0.0


def compute_copy_score(attn: torch.Tensor, input_ids: torch.Tensor) -> float:
    """How much does this head attend to duplicate tokens?

    For each position t, check attention to earlier positions with same token.
    """
    seq_len = attn.shape[0]
    if seq_len < 2:
        return 0.0

    ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    scores = []

    for t in range(1, seq_len):
        curr_token = ids[t]
        dup_attn = 0.0
        n_dups = 0
        for j in range(t):
            if ids[j] == curr_token:
                dup_attn += float(attn[t, j])
                n_dups += 1
        if n_dups > 0:
            scores.append(dup_attn / n_dups)

    return sum(scores) / len(scores) if scores else 0.0


def compute_positional_score(attn: torch.Tensor) -> float:
    """How position-dependent is this head's attention pattern?

    High score = attention is mostly determined by relative position
    (like always attending to BOS, or always to t-k for fixed k).
    Low score = attention depends on content.

    Measured as: correlation of attention pattern with a pure positional
    pattern (diagonal, anti-diagonal, first-column).
    """
    seq_len = attn.shape[0]
    if seq_len < 3:
        return 0.0

    # Check: how much total attention goes to position 0 (BOS)?
    bos_attn = float(attn[:, 0].mean())

    # Check: how diagonal is the attention? (attending to self)
    diag_scores = []
    for t in range(seq_len):
        diag_scores.append(float(attn[t, t]))
    self_attn = sum(diag_scores) / len(diag_scores)

    # Check: how much does attention decay with distance?
    # (pure positional heads show very regular decay)
    # Compute variance of attention across rows — low variance = positional
    row_stds = []
    for t in range(1, seq_len):
        row = attn[t, :t + 1]
        if row.numel() > 1:
            row_stds.append(float(row.std()))
    mean_std = sum(row_stds) / len(row_stds) if row_stds else 0.0

    # Combine: high BOS or high self-attention, or very regular patterns
    return max(bos_attn, self_attn, 1.0 - min(mean_std * 5, 1.0))


# ══════════════════════════════════════════════════════════════
# HEAD PROFILER
# ══════════════════════════════════════════════════════════════

class HeadProfiler:
    """Empirical functional classification of attention heads.

    Runs stimulus batteries through the model, captures per-head
    activation profiles, analyzes attention patterns, and classifies
    each head's computational role.

    Args:
        model: The transformer model
        tokenizer: Associated tokenizer
        capture_attention: Whether to capture attention weight matrices
            (slower but enables pattern analysis)
        batteries: Custom stimulus batteries (default: built-in set)
        max_new_tokens: How many tokens to generate per prompt
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        capture_attention: bool = True,
        batteries: Optional[Dict[str, list]] = None,
        max_new_tokens: int = 32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.capture_attention = capture_attention
        self.batteries = batteries or STIMULUS_BATTERIES
        self.max_new_tokens = max_new_tokens

        # Detect architecture
        self.device = next(model.parameters()).device
        self._n_layers, self._n_heads = self._detect_arch(model)
        self._head_dim = self._get_hidden_size(model) // self._n_heads

        # Storage for collected data
        # stimulus_type → list of per-prompt results
        # each result: {"head_energies": [n_layers][n_heads], "attn_patterns": [...], ...}
        self._collected: Dict[str, list] = {}

    @staticmethod
    def _detect_arch(model) -> Tuple[int, int]:
        if hasattr(model, 'config'):
            n_layers = getattr(model.config, 'num_hidden_layers',
                               getattr(model.config, 'n_layer', 12))
            n_heads = getattr(model.config, 'num_attention_heads',
                              getattr(model.config, 'n_head', 12))
            return n_layers, n_heads
        return 12, 12

    @staticmethod
    def _get_hidden_size(model) -> int:
        if hasattr(model, 'config'):
            return getattr(model.config, 'hidden_size',
                           getattr(model.config, 'n_embd', 768))
        return 768

    # ── Data Collection ──────────────────────────────────────

    def _run_battery(
        self,
        battery_name: str,
        prompts: list,
        probe,
        attn_capture: Optional[AttentionCapture],
        verbose: bool = False,
    ) -> list:
        """Run a stimulus battery and collect per-head data."""
        from dflux.causal_primitives import CausalPrimitives

        results = []

        for prompt_data in prompts:
            if isinstance(prompt_data, tuple):
                prompt, description = prompt_data
            else:
                prompt, description = prompt_data, ""

            if verbose:
                short = prompt[:60] + "..." if len(prompt) > 60 else prompt
                print(f"    [{battery_name}] {short}")

            # Encode
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            attention_mask = torch.ones_like(input_ids)

            # Reset probe
            probe.reset()

            # Generate with attention capture
            generate_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Note: we do NOT set output_attentions=True here.
            # SDPA/Flash backends don't support it and will error.
            # Instead, AttentionCapture hooks the QKV projection
            # and computes softmax(QK^T/sqrt(d)) manually.

            with torch.no_grad():
                try:
                    output = self.model.generate(**generate_kwargs)
                except Exception as e:
                    if verbose:
                        print(f"      ERROR: {e}")
                    continue

            # Collect per-head energies from probe
            if not probe.diagnostics:
                continue

            # Average head energies across generated tokens
            n_tok = len(probe.diagnostics)
            avg_head_energy = [[0.0] * self._n_heads for _ in range(self._n_layers)]
            for d in probe.diagnostics:
                for i in range(self._n_layers):
                    for j in range(self._n_heads):
                        if i < len(d.head_energies) and j < len(d.head_energies[i]):
                            avg_head_energy[i][j] += d.head_energies[i][j] / n_tok

            # Collect attention patterns if available
            attn_data = None
            if attn_capture:
                raw_weights = attn_capture.get_weights()
                if any(w is not None for w in raw_weights):
                    attn_data = raw_weights

            # Compute CP for this prompt
            cp = CausalPrimitives(self._n_layers, self._n_heads)
            probe.feed_causal_primitives(cp)
            cp_report = cp.compute() if hasattr(cp, 'compute') else cp.report()

            result = {
                "prompt": prompt,
                "description": description,
                "n_tokens": n_tok,
                "avg_head_energy": avg_head_energy,
                "attn_patterns": attn_data,
                "input_ids": input_ids.cpu(),
                "cp_report": cp_report,
                "diagnostics": [
                    {
                        "J": d.J,
                        "risk": d.hallucination_risk,
                        "regime": d.regime,
                        "head_entropy": d.head_entropy,
                    }
                    for d in probe.diagnostics
                ],
            }
            results.append(result)

        return results

    # ── Profiling Pipeline ───────────────────────────────────

    def profile(
        self,
        verbose: bool = True,
        include_cp: bool = True,
    ) -> ProfileReport:
        """Run full profiling pipeline.

        1. Run all stimulus batteries, capture per-head activation
        2. Analyze attention patterns (if captured)
        3. Compute differential activation across stimulus types
        4. Classify each head's function
        5. Generate report

        Args:
            verbose: Print progress
            include_cp: Also compute causal primitives per stimulus type

        Returns:
            ProfileReport with per-head functional classifications
        """
        from dflux.fine_probe import FineProbe

        if verbose:
            print("=" * 60)
            print("HEAD FUNCTIONAL PROFILER")
            print("=" * 60)
            print(f"Architecture: {self._n_layers}L × {self._n_heads}H")
            print(f"Batteries: {len(self.batteries)} ({sum(len(v) for v in self.batteries.values())} prompts)")
            print(f"Attention capture: {self.capture_attention}")
            print("=" * 60)

        # Setup probe
        probe = FineProbe.from_model(self.model)

        # Setup attention capture
        attn_capture = None
        if self.capture_attention:
            attn_capture = AttentionCapture(self.model, self._n_layers, self._n_heads)

        # ── Phase 1: Run all batteries ──
        if verbose:
            print("\nPhase 1: Running stimulus batteries...")

        self._collected = {}
        for battery_name, prompts in self.batteries.items():
            if verbose:
                print(f"\n  Battery: {battery_name} ({len(prompts)} prompts)")
            results = self._run_battery(battery_name, prompts, probe, attn_capture, verbose)
            self._collected[battery_name] = results

        # Detach hooks
        probe.detach()
        if attn_capture:
            attn_capture.detach()

        # ── Phase 2: Compute stimulus response matrix ──
        if verbose:
            print("\nPhase 2: Computing stimulus response profiles...")

        stimulus_response = self._compute_stimulus_response()

        # ── Phase 3: Analyze attention patterns ──
        if verbose:
            print("Phase 3: Analyzing attention patterns...")

        attn_metrics = self._analyze_attention_patterns()

        # ── Phase 4: Compute differential activation ──
        if verbose:
            print("Phase 4: Computing differential activation & classifying heads...")

        head_roles = self._classify_heads(stimulus_response, attn_metrics)

        # ── Phase 5: Build report ──
        if verbose:
            print("Phase 5: Building report...")

        # Role distribution
        role_dist = {}
        layer_spec = {}
        for h in head_roles:
            role_dist[h.primary_role] = role_dist.get(h.primary_role, 0) + 1
            if h.layer not in layer_spec:
                layer_spec[h.layer] = {}
            layer_spec[h.layer][h.primary_role] = layer_spec[h.layer].get(h.primary_role, 0) + 1

        model_name = ""
        if hasattr(self.model, 'config') and hasattr(self.model.config, '_name_or_path'):
            model_name = self.model.config._name_or_path

        report = ProfileReport(
            model_name=model_name,
            n_layers=self._n_layers,
            n_heads=self._n_heads,
            head_roles=head_roles,
            role_distribution=role_dist,
            layer_specialization=layer_spec,
            stimulus_response={k: v for k, v in stimulus_response.items()},
            attention_patterns_available=bool(attn_metrics),
        )

        if verbose:
            print("\n" + report.summary())

        return report

    # ── Stimulus Response ────────────────────────────────────

    def _compute_stimulus_response(self) -> Dict[str, List[List[float]]]:
        """Compute average per-head energy for each stimulus type.

        Returns: {stimulus_type: [n_layers][n_heads] average energy}
        """
        response = {}

        for battery_name, results in self._collected.items():
            if not results:
                continue

            # Average across all prompts in this battery
            avg = [[0.0] * self._n_heads for _ in range(self._n_layers)]
            n = len(results)

            for r in results:
                for i in range(self._n_layers):
                    for j in range(self._n_heads):
                        avg[i][j] += r["avg_head_energy"][i][j] / n

            response[battery_name] = avg

        return response

    # ── Attention Pattern Analysis ───────────────────────────

    def _analyze_attention_patterns(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Analyze attention weight patterns for each head.

        Returns: {(layer, head): {metric_name: value}}
        """
        if not self.capture_attention:
            return {}

        metrics: Dict[Tuple[int, int], Dict[str, List[float]]] = {}

        for battery_name, results in self._collected.items():
            for r in results:
                attn_data = r.get("attn_patterns")
                input_ids = r.get("input_ids")
                if attn_data is None or input_ids is None:
                    continue

                ids_flat = input_ids[0]  # [seq_len]

                for layer_idx, weights in enumerate(attn_data):
                    if weights is None:
                        continue
                    # weights: [batch, n_heads, seq_len, seq_len]
                    for head_idx in range(min(weights.shape[1], self._n_heads)):
                        key = (layer_idx, head_idx)
                        if key not in metrics:
                            metrics[key] = {
                                "prev_token": [],
                                "induction": [],
                                "copy": [],
                                "positional": [],
                            }

                        attn_mat = weights[0, head_idx]  # [seq_len, seq_len]

                        metrics[key]["prev_token"].append(
                            compute_prev_token_score(attn_mat))
                        metrics[key]["induction"].append(
                            compute_induction_score(attn_mat, ids_flat))
                        metrics[key]["copy"].append(
                            compute_copy_score(attn_mat, ids_flat))
                        metrics[key]["positional"].append(
                            compute_positional_score(attn_mat))

        # Average per head
        raw = {}
        for key, m in metrics.items():
            raw[key] = {
                name: sum(vals) / len(vals) if vals else 0.0
                for name, vals in m.items()
            }

        # ── Normalize against model-wide baseline ──
        # RoPE/rotary models have high baseline positional scores everywhere.
        # Other patterns can also have inflated baselines.
        # We subtract the model mean and divide by std so that only heads
        # that are *significantly above average* for a pattern get credit.
        if raw:
            pattern_names = ["prev_token", "induction", "copy", "positional"]
            baselines = {}
            for pname in pattern_names:
                vals = [raw[k][pname] for k in raw if pname in raw[k]]
                if vals:
                    mean = sum(vals) / len(vals)
                    std = max(
                        (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5,
                        1e-6
                    )
                    baselines[pname] = (mean, std)

            result = {}
            for key, m in raw.items():
                result[key] = {}
                for pname in pattern_names:
                    if pname in m and pname in baselines:
                        mean, std = baselines[pname]
                        # Z-score, then clamp to [0, 1]
                        z = (m[pname] - mean) / std
                        result[key][pname] = max(0.0, min(z / 3.0, 1.0))  # 3σ = 1.0
                    else:
                        result[key][pname] = m.get(pname, 0.0)
                # Also keep raw values for reporting
                result[key]["_raw"] = dict(m)
        else:
            result = raw

        return result

    # ── Classification Engine ────────────────────────────────

    def _classify_heads(
        self,
        stimulus_response: Dict[str, List[List[float]]],
        attn_metrics: Dict[Tuple[int, int], Dict[str, float]],
    ) -> List[HeadRole]:
        """Classify each head based on differential activation and attention patterns."""

        head_roles = []

        # Compute global energy statistics for normalization
        all_energies = []
        for stim_type, energy_map in stimulus_response.items():
            for i in range(self._n_layers):
                for j in range(self._n_heads):
                    all_energies.append(energy_map[i][j])

        global_mean = sum(all_energies) / len(all_energies) if all_energies else 1.0
        global_std = math.sqrt(
            sum((e - global_mean) ** 2 for e in all_energies) / len(all_energies)
        ) if all_energies else 1.0

        # Compute per-battery mean energy (for normalization)
        battery_means = {}
        for stim_type, energy_map in stimulus_response.items():
            total = sum(energy_map[i][j]
                        for i in range(self._n_layers)
                        for j in range(self._n_heads))
            battery_means[stim_type] = total / (self._n_layers * self._n_heads + 1e-12)

        # Aggregate CP from collected data
        head_cp_data = {}  # (layer, head) → {cp, det, spec}
        for battery_name, results in self._collected.items():
            for r in results:
                cp_report = r.get("cp_report", {})
                head_cp = cp_report.get("head_cp", [])
                head_det = cp_report.get("head_det", [])
                head_spec = cp_report.get("head_spec", [])
                for i in range(min(len(head_cp), self._n_layers)):
                    for j in range(min(len(head_cp[i]) if i < len(head_cp) else 0, self._n_heads)):
                        key = (i, j)
                        if key not in head_cp_data:
                            head_cp_data[key] = {"cp": [], "det": [], "spec": []}
                        head_cp_data[key]["cp"].append(head_cp[i][j])
                        if i < len(head_det) and j < len(head_det[i]):
                            head_cp_data[key]["det"].append(head_det[i][j])
                        if i < len(head_spec) and j < len(head_spec[i]):
                            head_cp_data[key]["spec"].append(head_spec[i][j])

        # Now classify each head
        for i in range(self._n_layers):
            # Layer-level energy ranking
            layer_energies = []
            for j in range(self._n_heads):
                mean_e = sum(
                    stimulus_response[s][i][j]
                    for s in stimulus_response
                ) / max(len(stimulus_response), 1)
                layer_energies.append((j, mean_e))
            layer_energies.sort(key=lambda x: -x[1])
            rank_map = {j: rank for rank, (j, _) in enumerate(layer_energies)}

            for j in range(self._n_heads):
                role = self._classify_single_head(
                    i, j,
                    stimulus_response,
                    attn_metrics.get((i, j), {}),
                    head_cp_data.get((i, j), {}),
                    battery_means,
                    global_mean,
                    global_std,
                    rank_map.get(j, 0),
                )
                head_roles.append(role)

        return head_roles

    def _classify_single_head(
        self,
        layer: int,
        head: int,
        stimulus_response: Dict[str, List[List[float]]],
        attn_metrics: Dict[str, float],
        cp_data: Dict[str, list],
        battery_means: Dict[str, float],
        global_mean: float,
        global_std: float,
        energy_rank: int,
    ) -> HeadRole:
        """Classify a single head's function.

        Scoring approach: compute a score for each possible role based on
        multiple evidence signals, then pick the highest.
        """
        scores: Dict[str, float] = {}
        stimulus_profile: Dict[str, float] = {}

        # ── Compute stimulus profile (normalized differential activation) ──
        for stim_type, energy_map in stimulus_response.items():
            raw_energy = energy_map[layer][head]
            # Normalize by battery mean
            bm = battery_means.get(stim_type, 1.0)
            norm_energy = raw_energy / (bm + 1e-12)
            stimulus_profile[stim_type] = norm_energy

        # ── Compute mean energy for this head ──
        mean_energy = sum(
            stimulus_response[s][layer][head]
            for s in stimulus_response
        ) / max(len(stimulus_response), 1)

        # ── Get CP metrics ──
        avg_cp = sum(cp_data.get("cp", [0])) / max(len(cp_data.get("cp", [1])), 1)
        avg_det = sum(cp_data.get("det", [0])) / max(len(cp_data.get("det", [1])), 1)
        avg_spec = sum(cp_data.get("spec", [0])) / max(len(cp_data.get("spec", [1])), 1)

        # ── Score each role ──

        # DEAD HEAD: near-zero energy
        dead_score = 0.0
        if mean_energy < global_mean * 0.02:
            dead_score = 0.9
        elif mean_energy < global_mean * 0.05:
            dead_score = 0.5
        elif mean_energy < global_mean * 0.1:
            dead_score = 0.2
        scores["dead"] = dead_score

        # INDUCTION: high energy on induction battery, attention pattern match
        ind_stim = stimulus_profile.get("induction", 1.0)
        ind_attn = attn_metrics.get("induction", 0.0)
        scores["induction"] = (
            0.4 * min(ind_stim / 1.5, 1.0) +    # High energy on induction prompts
            0.6 * ind_attn                          # Attention pattern match
        ) if ind_attn > 0 else 0.3 * min(ind_stim / 1.5, 1.0)

        # PREVIOUS TOKEN: attention pattern strongly diagonal-offset
        prev_attn = attn_metrics.get("prev_token", 0.0)
        scores["previous_token"] = prev_attn ** 0.7  # Soft threshold

        # COPY: attend to duplicate tokens
        copy_stim = stimulus_profile.get("copying", 1.0)
        copy_attn = attn_metrics.get("copy", 0.0)
        scores["copy"] = (
            0.3 * min(copy_stim / 1.5, 1.0) +
            0.7 * copy_attn
        ) if copy_attn > 0 else 0.3 * min(copy_stim / 1.5, 1.0)

        # FACTUAL RECALL: high energy on factual, low on hallucination
        fact_stim = stimulus_profile.get("factual_recall", 1.0)
        hall_stim = stimulus_profile.get("hallucination", 1.0)
        fact_diff = fact_stim - hall_stim
        scores["factual_recall"] = max(0, 0.3 + 0.7 * (fact_diff / max(fact_stim, 1.0)))

        # SYNTAX: high energy on syntax battery
        syn_stim = stimulus_profile.get("syntax", 1.0)
        # Syntax heads tend to be in early-mid layers
        layer_frac = layer / max(self._n_layers - 1, 1)
        syntax_layer_bonus = 1.0 if 0.15 < layer_frac < 0.65 else 0.7
        scores["syntax"] = min(syn_stim / 1.5, 1.0) * syntax_layer_bonus * 0.6

        # ENTITY TRACKING: high on entity tracking battery
        ent_stim = stimulus_profile.get("entity_tracking", 1.0)
        scores["entity_tracking"] = min(ent_stim / 1.5, 1.0) * 0.6

        # POSITIONAL: attention pattern is position-dependent
        pos_attn = attn_metrics.get("positional", 0.0)
        pos_stim = stimulus_profile.get("positional", 1.0)
        scores["positional"] = (
            0.5 * pos_attn +
            0.2 * min(pos_stim / 1.5, 1.0)
        )
        # Positional heads tend to be in early layers
        if layer_frac < 0.2:
            scores["positional"] *= 1.3

        # SUPPRESSION: high energy on suppression battery
        sup_stim = stimulus_profile.get("suppression", 1.0)
        scores["suppression"] = min(sup_stim / 1.5, 1.0) * 0.5

        # REASONING: high energy on reasoning battery, high determinism
        reas_stim = stimulus_profile.get("reasoning", 1.0)
        scores["reasoning"] = min(reas_stim / 1.5, 1.0) * 0.4 + avg_det * 0.3

        # SKEPTIC (CP-informed): high CP on factual, lower on hallucination
        # This is the head that "knows what it knows"
        if avg_cp > 0.1 and fact_diff > 0.2:
            scores["skeptic"] = min(avg_cp * 2, 1.0) * 0.5 + fact_diff * 0.3
        else:
            scores["skeptic"] = 0.0

        # ARBITRATOR (CP-informed): small energy share, high CP
        layer_total = sum(
            sum(stimulus_response[s][layer][h] for s in stimulus_response)
            for h in range(self._n_heads)
        )
        energy_share = mean_energy * len(stimulus_response) / (layer_total + 1e-12)
        if energy_share < 0.05 and avg_cp > 0.15:
            scores["arbitrator"] = avg_cp * (1 - energy_share * 10)
        else:
            scores["arbitrator"] = 0.0

        # WORKHORSE: high energy, moderate CP
        if mean_energy > global_mean * 1.5 and avg_cp > 0.05:
            scores["workhorse"] = min(mean_energy / (global_mean * 3), 1.0) * 0.4 + avg_cp * 0.3
        else:
            scores["workhorse"] = max(0, (mean_energy / (global_mean * 3)) * 0.3)

        # HALLUCINATION: overactive on hallucination prompts
        if hall_stim > 1.3 and hall_stim > fact_stim * 1.2:
            scores["hallucination_prone"] = min((hall_stim - 1.0) * 0.5, 1.0)
        else:
            scores["hallucination_prone"] = 0.0

        # ── Pick winner ──
        # Dead head overrides if very confident
        if dead_score > 0.8:
            primary = "dead"
            primary_score = dead_score
        else:
            # Remove dead from consideration for non-dead heads
            non_dead = {k: v for k, v in scores.items() if k != "dead"}
            if non_dead:
                primary = max(non_dead, key=non_dead.get)
                primary_score = non_dead[primary]
            else:
                primary = "unclassified"
                primary_score = 0.0

        # Second best
        remaining = {k: v for k, v in scores.items() if k != primary}
        secondary = max(remaining, key=remaining.get) if remaining else ""

        # Generate description
        description = self._describe_head(
            primary, primary_score, stimulus_profile, attn_metrics,
            avg_cp, mean_energy, global_mean, layer, energy_rank,
        )

        return HeadRole(
            layer=layer,
            head=head,
            primary_role=primary,
            confidence=min(primary_score, 1.0),
            secondary_role=secondary,
            description=description,
            role_scores=scores,
            stimulus_profile=stimulus_profile,
            # Store z-scored values (used by classifier) for reporting
            avg_prev_token_score=attn_metrics.get("prev_token", 0.0),
            avg_induction_score=attn_metrics.get("induction", 0.0),
            avg_copy_score=attn_metrics.get("copy", 0.0),
            avg_positional_score=attn_metrics.get("positional", 0.0),
            cp_value=avg_cp,
            determinism=avg_det,
            specificity=avg_spec,
            mean_energy=mean_energy,
            energy_rank=energy_rank,
        )

    @staticmethod
    def _describe_head(
        role: str,
        confidence: float,
        stim_profile: Dict[str, float],
        attn_metrics: Dict[str, float],
        cp: float,
        energy: float,
        global_mean: float,
        layer: int,
        rank: int,
    ) -> str:
        """Generate human-readable description of head function."""
        energy_level = (
            "very high" if energy > global_mean * 3 else
            "high" if energy > global_mean * 1.5 else
            "moderate" if energy > global_mean * 0.5 else
            "low" if energy > global_mean * 0.1 else
            "near-zero"
        )

        descriptions = {
            "induction": f"Pattern completer (energy={energy_level}, CP={cp:.3f}). "
                         f"Copies bigram patterns from earlier context.",
            "previous_token": f"Previous-token attender (energy={energy_level}). "
                              f"Primarily attends to position t-1.",
            "copy": f"Token copier (energy={energy_level}, CP={cp:.3f}). "
                    f"Attends to earlier occurrences of same token.",
            "factual_recall": f"Knowledge retriever (energy={energy_level}, CP={cp:.3f}). "
                              f"Activates strongly on factual completions.",
            "syntax": f"Syntax tracker (energy={energy_level}, CP={cp:.3f}). "
                      f"Tracks grammatical structure and agreement.",
            "entity_tracking": f"Entity tracker (energy={energy_level}, CP={cp:.3f}). "
                               f"Moves entity/name info across positions.",
            "positional": f"Position encoder (energy={energy_level}). "
                          f"Attention primarily driven by position, not content.",
            "suppression": f"Suppressor (energy={energy_level}, CP={cp:.3f}). "
                           f"Active during inhibition/negation tasks.",
            "reasoning": f"Reasoning head (energy={energy_level}, CP={cp:.3f}). "
                         f"Activates on logical inference tasks.",
            "skeptic": f"SKEPTIC (energy={energy_level}, CP={cp:.3f}). "
                       f"High CP on factual, suppressed during hallucination.",
            "arbitrator": f"ARBITRATOR (energy={energy_level}, CP={cp:.3f}). "
                          f"Small energy share but high causal influence — casting deciding votes.",
            "workhorse": f"Workhorse (rank #{rank+1} in layer, CP={cp:.3f}). "
                         f"High energy, bulk computation.",
            "hallucination_prone": f"Hallucination-prone (energy={energy_level}, CP={cp:.3f}). "
                                   f"Overactive on fabrication tasks.",
            "dead": f"Dead head (energy={energy_level}). "
                    f"Near-zero contribution to residual stream.",
            "unclassified": f"Unclassified (energy={energy_level}, CP={cp:.3f}). "
                            f"No strong differential signal.",
        }

        return descriptions.get(role, f"{role} (energy={energy_level}, CP={cp:.3f})")


# ══════════════════════════════════════════════════════════════
# CONVENIENCE
# ══════════════════════════════════════════════════════════════

def quick_profile(model, tokenizer, verbose=True) -> ProfileReport:
    """One-liner: profile all heads in a model.

    Usage:
        from dflux.head_profiler import quick_profile
        report = quick_profile(model, tokenizer)
    """
    profiler = HeadProfiler(model, tokenizer)
    return profiler.profile(verbose=verbose)

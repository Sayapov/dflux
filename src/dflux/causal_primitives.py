#!/usr/bin/env python3
"""
Causal Primitives — information-theoretic measurement of head-level causation.

Based on Jansma & Hoel (2025), "Engineering Emergence." Adapted for
transformer attention heads.

Instead of guessing what a head "does" (skeptic, arbitrator, etc.), we
measure how much causal work it actually performs — determinism (does this
head reliably drive specific effects?) and specificity (are this head's
effects unique, or redundant with other heads?).

CP = I(C; E) / log2(n_bins)

Where:
    C = cause = head energy at token t (discretized)
    E = effect = output metric at token t (discretized)
    I(C; E) = mutual information between cause and effect

High CP: this head reliably and uniquely drives output behavior.
Low CP: this head is either noise (low determinism) or redundant (low specificity).

Three scales of analysis:
    1. Per-head CP (microscale): each head vs output
    2. Per-layer CP (mesoscale): layer-aggregate energy vs output
    3. Cross-head CP (interaction): head-to-head causal influence

Usage:
    from dflux.causal_primitives import CausalPrimitives

    cp = CausalPrimitives(n_bins=16)

    # Feed token-level data from FineProbe
    for token_diag in probe.token_history:
        cp.observe(token_diag)

    report = cp.report()
    # report["head_cp"]     → [n_layers][n_heads] CP values
    # report["head_det"]    → [n_layers][n_heads] determinism
    # report["head_spec"]   → [n_layers][n_heads] specificity
    # report["layer_cp"]    → [n_layers] layer-level CP
    # report["hierarchy"]   → "top-heavy" | "bottom-heavy" | "distributed"
    # report["emergence"]   → S_path * S_row_bar (emergent complexity)

Free instrument. MIT license.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


@dataclass
class CPConfig:
    """Configuration for causal primitives computation."""
    n_bins: int = 16          # Discretization bins for energy values
    min_samples: int = 20     # Minimum tokens before reporting
    effect_metric: str = "J"  # "J" (energy functional) or "risk" or "residual"


@dataclass
class HeadCP:
    """Causal primitives for a single attention head."""
    layer: int
    head: int
    determinism: float    # 1 - H(E|C)/log2(n): how reliably does this head predict output?
    degeneracy: float     # 1 - H(E)/log2(n): how much do effects overlap?
    specificity: float    # 1 - degeneracy: how unique are this head's effects?
    cp: float             # determinism + specificity - 1 = I(C;E)/log2(n)
    mutual_info: float    # I(C;E) in bits (unnormalized)
    mean_energy: float    # Average energy for context


class CausalPrimitives:
    """
    Information-theoretic measurement of per-head causal contribution.

    Collects (head_energy, effect_metric) pairs across tokens, then
    computes mutual information to determine each head's causal
    primitives — without any semantic labeling.
    """

    def __init__(self, n_layers: int, n_heads: int, cfg: Optional[CPConfig] = None):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.cfg = cfg or CPConfig()

        # Storage: per-token observations
        # head_energies[layer][head] = list of energy values
        self.head_energies: List[List[List[float]]] = [
            [[] for _ in range(n_heads)] for _ in range(n_layers)
        ]
        # layer_energies[layer] = list of total layer energy values
        self.layer_energies: List[List[float]] = [[] for _ in range(n_layers)]

        # Effect metric (output-side observable)
        self.effects: List[float] = []

        self._n_tokens = 0

    @classmethod
    def from_probe(cls, probe, cfg: Optional[CPConfig] = None) -> "CausalPrimitives":
        """Create from an attached FineProbe."""
        return cls(probe.n_layers, probe.n_heads, cfg)

    def observe_token(
        self,
        head_energy: List[List[float]],   # [n_layers][n_heads]
        layer_energy: List[float],         # [n_layers]
        effect: float,                     # scalar output metric
    ):
        """
        Record one token's worth of observations.

        Args:
            head_energy: Per-head energy for this token [n_layers][n_heads]
            layer_energy: Per-layer total energy for this token [n_layers]
            effect: The output-side metric (J, risk, residual norm, etc.)
        """
        for i in range(min(self.n_layers, len(head_energy))):
            for j in range(min(self.n_heads, len(head_energy[i]))):
                self.head_energies[i][j].append(head_energy[i][j])

        for i in range(min(self.n_layers, len(layer_energy))):
            self.layer_energies[i].append(layer_energy[i])

        self.effects.append(effect)
        self._n_tokens += 1

    def observe_from_probe_report(self, report: Dict[str, Any]):
        """
        Ingest data from a FineProbe report's token-level history.

        This is the convenience method — takes a full probe report and
        extracts what we need. Requires the probe to have been run with
        token-level recording.
        """
        token_history = report.get("token_diagnostics", [])
        for td in token_history:
            he = td.get("head_energy", [])
            # Sum per-layer as layer energy
            le = [sum(layer_heads) for layer_heads in he] if he else []
            effect = td.get("J", td.get("risk", 0.0))
            if he:
                self.observe_token(he, le, effect)

    def _discretize(self, values: List[float], n_bins: int) -> List[int]:
        """Discretize continuous values into bins via equal-width binning."""
        if not values:
            return []
        v_min = min(values)
        v_max = max(values)
        if v_max == v_min:
            return [0] * len(values)
        bin_width = (v_max - v_min) / n_bins
        return [min(int((v - v_min) / bin_width), n_bins - 1) for v in values]

    def _entropy(self, counts: List[int], total: int) -> float:
        """Shannon entropy from count histogram."""
        if total == 0:
            return 0.0
        h = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                h -= p * math.log2(p)
        return h

    def _histogram(self, bins: List[int], n_bins: int) -> List[int]:
        """Count histogram from bin assignments."""
        counts = [0] * n_bins
        for b in bins:
            if 0 <= b < n_bins:
                counts[b] += 1
        return counts

    def _joint_histogram(
        self, cause_bins: List[int], effect_bins: List[int], n_bins: int
    ) -> List[List[int]]:
        """Joint count histogram [cause_bin][effect_bin]."""
        joint = [[0] * n_bins for _ in range(n_bins)]
        for c, e in zip(cause_bins, effect_bins):
            if 0 <= c < n_bins and 0 <= e < n_bins:
                joint[c][e] += 1
        return joint

    def _compute_cp(
        self, cause_values: List[float], effect_values: List[float]
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute causal primitives between cause and effect series.

        Returns: (determinism, degeneracy, specificity, cp, mutual_info)
        """
        n = len(cause_values)
        nb = self.cfg.n_bins

        if n < self.cfg.min_samples:
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        cause_bins = self._discretize(cause_values, nb)
        effect_bins = self._discretize(effect_values, nb)

        # Marginal entropy of effects: H(E)
        effect_hist = self._histogram(effect_bins, nb)
        h_e = self._entropy(effect_hist, n)

        # Marginal entropy of causes: H(C)
        cause_hist = self._histogram(cause_bins, nb)
        h_c = self._entropy(cause_hist, n)

        # Conditional entropy H(E|C): average entropy of effects given each cause bin
        joint = self._joint_histogram(cause_bins, effect_bins, nb)
        h_e_given_c = 0.0
        for c_bin in range(nb):
            c_count = cause_hist[c_bin]
            if c_count > 0:
                h_e_given_c += (c_count / n) * self._entropy(joint[c_bin], c_count)

        # Mutual information: I(C;E) = H(E) - H(E|C)
        mi = max(0.0, h_e - h_e_given_c)

        log_n = math.log2(nb) if nb > 1 else 1.0

        # Determinism: 1 - H(E|C)/log2(n)
        determinism = 1.0 - h_e_given_c / log_n if log_n > 0 else 0.0

        # Degeneracy: 1 - H(E)/log2(n)
        degeneracy = 1.0 - h_e / log_n if log_n > 0 else 0.0

        # Specificity: 1 - degeneracy = H(E)/log2(n)
        specificity = 1.0 - degeneracy

        # CP = determinism + specificity - 1 = I(C;E)/log2(n)
        cp = mi / log_n if log_n > 0 else 0.0

        return (determinism, degeneracy, specificity, cp, mi)

    def compute(self) -> Dict[str, Any]:
        """
        Compute causal primitives at all scales.

        Returns a dict with:
            head_cp:    [n_layers][n_heads] — CP per head
            head_det:   [n_layers][n_heads] — determinism per head
            head_deg:   [n_layers][n_heads] — degeneracy per head
            head_spec:  [n_layers][n_heads] — specificity per head
            head_mi:    [n_layers][n_heads] — mutual info per head (bits)
            head_detail: [n_layers][n_heads] — full HeadCP objects
            layer_cp:   [n_layers] — layer-level CP (mesoscale)
            layer_det:  [n_layers] — layer-level determinism
            layer_spec: [n_layers] — layer-level specificity
            hierarchy:  str — "top-heavy" | "bottom-heavy" | "distributed"
            emergence:  float — emergent complexity measure
            S_path:     float — path entropy (how spread across layers)
            S_row_bar:  float — row negentropy (differentiation within layers)
            n_tokens:   int — number of tokens analyzed
        """
        if self._n_tokens < self.cfg.min_samples:
            return {"status": "insufficient_data", "n_tokens": self._n_tokens}

        effects = self.effects

        # ── Per-head CP (microscale) ──────────────────────────────
        head_cp = [[0.0] * self.n_heads for _ in range(self.n_layers)]
        head_det = [[0.0] * self.n_heads for _ in range(self.n_layers)]
        head_deg = [[0.0] * self.n_heads for _ in range(self.n_layers)]
        head_spec = [[0.0] * self.n_heads for _ in range(self.n_layers)]
        head_mi = [[0.0] * self.n_heads for _ in range(self.n_layers)]
        head_detail = [[None] * self.n_heads for _ in range(self.n_layers)]

        for i in range(self.n_layers):
            for j in range(self.n_heads):
                energies = self.head_energies[i][j]
                if len(energies) >= self.cfg.min_samples:
                    det, deg, spec, cp, mi = self._compute_cp(energies, effects)
                    head_cp[i][j] = cp
                    head_det[i][j] = det
                    head_deg[i][j] = deg
                    head_spec[i][j] = spec
                    head_mi[i][j] = mi
                    head_detail[i][j] = HeadCP(
                        layer=i, head=j,
                        determinism=det, degeneracy=deg,
                        specificity=spec, cp=cp, mutual_info=mi,
                        mean_energy=sum(energies) / len(energies),
                    )

        # ── Per-layer CP (mesoscale) ──────────────────────────────
        layer_cp = [0.0] * self.n_layers
        layer_det = [0.0] * self.n_layers
        layer_spec = [0.0] * self.n_layers

        for i in range(self.n_layers):
            energies = self.layer_energies[i]
            if len(energies) >= self.cfg.min_samples:
                det, deg, spec, cp, mi = self._compute_cp(energies, effects)
                layer_cp[i] = cp
                layer_det[i] = det
                layer_spec[i] = spec

        # ── Emergent hierarchy classification ─────────────────────
        # Following Jansma & Hoel: compute ΔCP per layer
        # (non-redundant causal contribution beyond previous layers)
        # Simplified: we use layer_cp directly as the CP profile
        total_cp = sum(layer_cp)

        if total_cp > 0:
            # Normalize to distribution
            cp_dist = [c / total_cp for c in layer_cp]

            # S_path: entropy of CP distribution across layers
            # High = causation spread evenly, Low = concentrated
            s_path = 0.0
            for p in cp_dist:
                if p > 0:
                    s_path -= p * math.log2(p)

            # Normalize by max entropy (log2(n_layers))
            max_path_entropy = math.log2(self.n_layers) if self.n_layers > 1 else 1.0
            s_path_norm = s_path / max_path_entropy

            # S_row: within-layer differentiation of head CP values
            # High negentropy = heads within a layer have different CP values
            s_row_values = []
            for i in range(self.n_layers):
                layer_head_cps = [head_cp[i][j] for j in range(self.n_heads)]
                layer_total = sum(layer_head_cps)
                if layer_total > 0:
                    layer_dist = [c / layer_total for c in layer_head_cps]
                    s_l = 0.0
                    for p in layer_dist:
                        if p > 0:
                            s_l -= p * math.log2(p)
                    s_row_values.append(s_l)
                else:
                    s_row_values.append(0.0)

            s_row = sum(s_row_values) / len(s_row_values) if s_row_values else 0.0
            max_row_entropy = math.log2(self.n_heads) if self.n_heads > 1 else 1.0
            s_row_bar = max_row_entropy - s_row  # negentropy

            # Emergent complexity = S_path * S_row_bar
            emergence = s_path_norm * (s_row_bar / max_row_entropy if max_row_entropy > 0 else 0.0)

            # Classify hierarchy
            # Find which third of the model has most CP
            third = max(1, self.n_layers // 3)
            bottom_cp = sum(layer_cp[:third])
            middle_cp = sum(layer_cp[third:2*third])
            top_cp = sum(layer_cp[2*third:])

            if top_cp > 0.5 * total_cp:
                hierarchy = "top-heavy"
            elif bottom_cp > 0.5 * total_cp:
                hierarchy = "bottom-heavy"
            elif middle_cp > max(top_cp, bottom_cp):
                hierarchy = "mesoscale-peaked"
            else:
                hierarchy = "distributed"
        else:
            s_path = 0.0
            s_path_norm = 0.0
            s_row_bar = 0.0
            emergence = 0.0
            hierarchy = "no_signal"

        # ── Top heads by CP ───────────────────────────────────────
        all_heads = []
        for i in range(self.n_layers):
            for j in range(self.n_heads):
                if head_detail[i][j] is not None:
                    all_heads.append(head_detail[i][j])

        all_heads.sort(key=lambda h: h.cp, reverse=True)
        top_heads = [
            {
                "layer": h.layer, "head": h.head,
                "cp": round(h.cp, 6),
                "determinism": round(h.determinism, 6),
                "specificity": round(h.specificity, 6),
                "mutual_info": round(h.mutual_info, 6),
                "mean_energy": round(h.mean_energy, 4),
            }
            for h in all_heads[:20]
        ]

        # ── Bottom heads (lowest CP, potential noise) ─────────────
        bottom_heads = [
            {
                "layer": h.layer, "head": h.head,
                "cp": round(h.cp, 6),
                "determinism": round(h.determinism, 6),
                "specificity": round(h.specificity, 6),
                "mean_energy": round(h.mean_energy, 4),
            }
            for h in all_heads[-10:]
        ]

        return {
            "status": "ok",
            "n_tokens": self._n_tokens,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_bins": self.cfg.n_bins,
            # Per-head (microscale)
            "head_cp": [[round(v, 6) for v in row] for row in head_cp],
            "head_det": [[round(v, 6) for v in row] for row in head_det],
            "head_deg": [[round(v, 6) for v in row] for row in head_deg],
            "head_spec": [[round(v, 6) for v in row] for row in head_spec],
            "head_mi": [[round(v, 6) for v in row] for row in head_mi],
            # Per-layer (mesoscale)
            "layer_cp": [round(v, 6) for v in layer_cp],
            "layer_det": [round(v, 6) for v in layer_det],
            "layer_spec": [round(v, 6) for v in layer_spec],
            # Hierarchy
            "hierarchy": hierarchy,
            "emergence": round(emergence, 6),
            "S_path": round(s_path, 6),
            "S_path_norm": round(s_path_norm, 6),
            "S_row_bar": round(s_row_bar, 6),
            # Rankings
            "top_heads": top_heads,
            "bottom_heads": bottom_heads,
        }

    def reset(self):
        """Clear all observations."""
        self.head_energies = [
            [[] for _ in range(self.n_heads)] for _ in range(self.n_layers)
        ]
        self.layer_energies = [[] for _ in range(self.n_layers)]
        self.effects = []
        self._n_tokens = 0

    def report(self) -> Dict[str, Any]:
        """Alias for compute()."""
        return self.compute()


def compute_cross_head_cp(
    head_energies: List[List[List[float]]],
    n_layers: int,
    n_heads: int,
    n_bins: int = 16,
    min_samples: int = 20,
) -> Dict[str, Any]:
    """
    Compute head-to-head causal influence matrix.

    For every pair of heads (h_cause, h_effect), computes the mutual
    information between h_cause's energy and h_effect's energy across
    tokens. This reveals which heads drive which other heads.

    Only computes cross-layer pairs where cause_layer < effect_layer
    (respecting causal direction in the forward pass).

    Args:
        head_energies: [n_layers][n_heads][n_tokens] energy timeseries
        n_layers: number of layers
        n_heads: number of heads
        n_bins: discretization bins
        min_samples: minimum tokens required

    Returns:
        Dict with "interactions" list of {cause_layer, cause_head,
        effect_layer, effect_head, mutual_info, cp} sorted by MI.
    """
    if not head_energies or not head_energies[0] or not head_energies[0][0]:
        return {"status": "no_data", "interactions": []}

    n_tokens = len(head_energies[0][0])
    if n_tokens < min_samples:
        return {"status": "insufficient_data", "n_tokens": n_tokens, "interactions": []}

    # Helper: discretize
    def discretize(values):
        v_min, v_max = min(values), max(values)
        if v_max == v_min:
            return [0] * len(values)
        bw = (v_max - v_min) / n_bins
        return [min(int((v - v_min) / bw), n_bins - 1) for v in values]

    def mutual_info(cause_vals, effect_vals):
        cb = discretize(cause_vals)
        eb = discretize(effect_vals)
        n = len(cb)

        # Joint counts
        joint = {}
        c_counts = [0] * n_bins
        e_counts = [0] * n_bins
        for c, e in zip(cb, eb):
            joint[(c, e)] = joint.get((c, e), 0) + 1
            c_counts[c] += 1
            e_counts[e] += 1

        mi = 0.0
        for (c, e), count in joint.items():
            if count > 0 and c_counts[c] > 0 and e_counts[e] > 0:
                p_ce = count / n
                p_c = c_counts[c] / n
                p_e = e_counts[e] / n
                mi += p_ce * math.log2(p_ce / (p_c * p_e))

        return max(0.0, mi)

    log_n = math.log2(n_bins) if n_bins > 1 else 1.0

    # Compute cross-layer interactions (cause < effect, respecting forward pass)
    interactions = []
    # Only check a subset to keep compute manageable:
    # For each head, check its influence on heads 1-3 layers downstream
    max_layer_gap = min(3, n_layers - 1)

    for cl in range(n_layers):
        for ch in range(n_heads):
            cause_vals = head_energies[cl][ch]
            if len(cause_vals) < min_samples:
                continue

            for el in range(cl + 1, min(cl + max_layer_gap + 1, n_layers)):
                for eh in range(n_heads):
                    effect_vals = head_energies[el][eh]
                    if len(effect_vals) < min_samples:
                        continue

                    mi = mutual_info(cause_vals, effect_vals)
                    if mi > 0.001:  # Only report meaningful interactions
                        interactions.append({
                            "cause_layer": cl,
                            "cause_head": ch,
                            "effect_layer": el,
                            "effect_head": eh,
                            "mutual_info": round(mi, 6),
                            "cp": round(mi / log_n, 6),
                        })

    interactions.sort(key=lambda x: x["mutual_info"], reverse=True)

    return {
        "status": "ok",
        "n_tokens": n_tokens,
        "n_interactions": len(interactions),
        "top_interactions": interactions[:30],
        "interactions": interactions,
    }

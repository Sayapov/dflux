#!/usr/bin/env python3
"""
Head Profile — empirically map what every attention head does.

Full pipeline:
  1. Load model
  2. Run 9 stimulus batteries (36+ prompts) through the model
  3. Capture per-head energy profiles and attention patterns
  4. Classify each head's computational function
  5. Generate a complete functional map
  6. Visualize: layer×head heatmaps, role distribution, specialization

Usage:
    python head_profile_model.py                                    # GPT-2
    python head_profile_model.py --model gpt2 --device mps          # Apple Silicon
    python head_profile_model.py --model EleutherAI/pythia-1.4b     # Pythia
    python head_profile_model.py --model Qwen/Qwen2.5-0.5B         # Qwen
    python head_profile_model.py --no-attention                     # Skip attention capture (faster)
"""

import sys, os, json, argparse, time

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflux.head_profiler import HeadProfiler, ProfileReport, STIMULUS_BATTERIES_ALT


# ═══════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════

def visualize_profile(report: ProfileReport, save_path: str):
    """Generate a multi-panel visualization of the profiling results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available — skipping visualization")
        return

    n_layers = report.n_layers
    n_heads = report.n_heads

    # ── Role color map ──
    role_colors = {
        "induction": "#e74c3c",       # Red
        "previous_token": "#e67e22",   # Orange
        "copy": "#f39c12",             # Yellow-orange
        "factual_recall": "#2ecc71",   # Green
        "syntax": "#1abc9c",           # Teal
        "entity_tracking": "#3498db",  # Blue
        "positional": "#9b59b6",       # Purple
        "suppression": "#e91e63",      # Pink
        "reasoning": "#00bcd4",        # Cyan
        "skeptic": "#27ae60",          # Dark green
        "arbitrator": "#8e44ad",       # Dark purple
        "workhorse": "#2c3e50",        # Dark blue-gray
        "hallucination_prone": "#c0392b",  # Dark red
        "dead": "#95a5a6",             # Gray
        "unclassified": "#bdc3c7",     # Light gray
    }

    fig = plt.figure(figsize=(28, 20))
    fig.suptitle(f"Head Functional Profile: {report.model_name}",
                 fontsize=16, fontweight='bold', y=0.98)

    # ── Panel 1: Role Map (layer × head grid, colored by role) ──
    ax1 = fig.add_subplot(3, 3, (1, 2))
    role_grid = []
    conf_grid = []
    for i in range(n_layers):
        row_roles = []
        row_confs = []
        for j in range(n_heads):
            h = next((h for h in report.head_roles
                       if h.layer == i and h.head == j), None)
            if h:
                row_roles.append(h.primary_role)
                row_confs.append(h.confidence)
            else:
                row_roles.append("unclassified")
                row_confs.append(0.0)
        role_grid.append(row_roles)
        conf_grid.append(row_confs)

    # Convert to numeric for imshow
    all_roles = sorted(set(r for row in role_grid for r in row))
    role_to_idx = {r: i for i, r in enumerate(all_roles)}
    numeric_grid = [[role_to_idx[r] for r in row] for row in role_grid]

    # Custom colormap from role colors
    colors_ordered = [role_colors.get(r, "#bdc3c7") for r in all_roles]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors_ordered)

    im = ax1.imshow(numeric_grid, aspect='auto', cmap=cmap,
                     vmin=-0.5, vmax=len(all_roles) - 0.5)
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_title("Functional Role Map")

    # Legend
    patches = [mpatches.Patch(color=role_colors.get(r, "#bdc3c7"), label=r)
               for r in all_roles]
    ax1.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left',
               fontsize=7, ncol=1)

    # ── Panel 2: Confidence Map ──
    ax2 = fig.add_subplot(3, 3, 3)
    im2 = ax2.imshow(conf_grid, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_xlabel("Head")
    ax2.set_ylabel("Layer")
    ax2.set_title("Classification Confidence")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # ── Panel 3: Role Distribution ──
    ax3 = fig.add_subplot(3, 3, 4)
    roles_sorted = sorted(report.role_distribution.items(), key=lambda x: -x[1])
    role_names = [r[0] for r in roles_sorted]
    role_counts = [r[1] for r in roles_sorted]
    bar_colors = [role_colors.get(r, "#bdc3c7") for r in role_names]
    ax3.barh(range(len(role_names)), role_counts, color=bar_colors)
    ax3.set_yticks(range(len(role_names)))
    ax3.set_yticklabels(role_names, fontsize=8)
    ax3.set_xlabel("Count")
    ax3.set_title("Role Distribution")
    ax3.invert_yaxis()

    # ── Panel 4: Stimulus Response Heatmap (differential) ──
    ax4 = fig.add_subplot(3, 3, 5)
    if report.stimulus_response:
        stim_names = sorted(report.stimulus_response.keys())
        # Average across heads per layer per stimulus
        response_matrix = []
        for stim in stim_names:
            layer_avgs = []
            for i in range(n_layers):
                layer_avg = sum(report.stimulus_response[stim][i]) / n_heads
                layer_avgs.append(layer_avg)
            response_matrix.append(layer_avgs)

        # Normalize per stimulus type
        for si in range(len(response_matrix)):
            row_max = max(response_matrix[si]) if response_matrix[si] else 1
            if row_max > 0:
                response_matrix[si] = [v / row_max for v in response_matrix[si]]

        im4 = ax4.imshow(response_matrix, aspect='auto', cmap='hot')
        ax4.set_yticks(range(len(stim_names)))
        ax4.set_yticklabels(stim_names, fontsize=8)
        ax4.set_xlabel("Layer")
        ax4.set_title("Stimulus Response by Layer (normalized)")
        plt.colorbar(im4, ax=ax4, shrink=0.8)

    # ── Panel 5: CP vs Energy scatter ──
    ax5 = fig.add_subplot(3, 3, 6)
    for h in report.head_roles:
        color = role_colors.get(h.primary_role, "#bdc3c7")
        ax5.scatter(h.mean_energy, h.cp_value, c=color, s=20, alpha=0.6)
    ax5.set_xlabel("Mean Energy")
    ax5.set_ylabel("CP Value")
    ax5.set_title("CP vs Energy (colored by role)")
    ax5.set_xscale('log')

    # ── Panel 6: Layer Specialization Stacked Bar ──
    ax6 = fig.add_subplot(3, 3, 7)
    layer_role_counts = {}
    for h in report.head_roles:
        if h.layer not in layer_role_counts:
            layer_role_counts[h.layer] = {}
        r = h.primary_role
        layer_role_counts[h.layer][r] = layer_role_counts[h.layer].get(r, 0) + 1

    unique_roles = sorted(set(h.primary_role for h in report.head_roles))
    bottom = [0] * n_layers
    for role in unique_roles:
        counts = [layer_role_counts.get(i, {}).get(role, 0) for i in range(n_layers)]
        ax6.bar(range(n_layers), counts, bottom=bottom,
                color=role_colors.get(role, "#bdc3c7"), label=role, width=0.9)
        bottom = [b + c for b, c in zip(bottom, counts)]
    ax6.set_xlabel("Layer")
    ax6.set_ylabel("Head Count")
    ax6.set_title("Layer Specialization")
    ax6.legend(fontsize=6, ncol=2, loc='upper right')

    # ── Panel 7: Attention Pattern Scores (if available) ──
    ax7 = fig.add_subplot(3, 3, 8)
    if report.attention_patterns_available:
        pattern_types = ["prev_token", "induction", "copy", "positional"]
        pattern_labels = ["Prev Token", "Induction", "Copy", "Positional"]
        # Average per layer
        for pi, ptype in enumerate(pattern_types):
            layer_avgs = []
            for i in range(n_layers):
                vals = []
                for h in report.head_roles:
                    if h.layer == i:
                        score = getattr(h, f"avg_{ptype}_score", 0.0)
                        vals.append(score)
                layer_avgs.append(sum(vals) / len(vals) if vals else 0)
            ax7.plot(range(n_layers), layer_avgs, label=pattern_labels[pi], linewidth=2)
        ax7.set_xlabel("Layer")
        ax7.set_ylabel("Pattern Score")
        ax7.set_title("Attention Pattern Types by Layer")
        ax7.legend(fontsize=8)
    else:
        ax7.text(0.5, 0.5, "Attention patterns\nnot captured",
                 ha='center', va='center', fontsize=12, transform=ax7.transAxes)
        ax7.set_title("Attention Patterns (not available)")

    # ── Panel 8: Top 20 Most Interesting Heads ──
    ax8 = fig.add_subplot(3, 3, 9)
    # "Interesting" = highest confidence, excluding dead and unclassified
    interesting = [h for h in report.head_roles
                   if h.primary_role not in ("dead", "unclassified")]
    interesting.sort(key=lambda h: h.confidence, reverse=True)
    top_n = interesting[:20]

    labels = [f"L{h.layer}H{h.head}" for h in top_n]
    confs = [h.confidence for h in top_n]
    colors = [role_colors.get(h.primary_role, "#bdc3c7") for h in top_n]
    y_pos = range(len(labels))

    ax8.barh(y_pos, confs, color=colors)
    ax8.set_yticks(y_pos)
    ax8.set_yticklabels([f"{l} ({h.primary_role})" for l, h in zip(labels, top_n)],
                         fontsize=7)
    ax8.set_xlabel("Confidence")
    ax8.set_title("Top 20 Most Confidently Classified")
    ax8.invert_yaxis()
    ax8.set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved: {save_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def get_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Head Functional Profiler")
    parser.add_argument("--model", default="gpt2",
                        help="Model name or path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-attention", action="store_true",
                        help="Skip attention weight capture (faster)")
    parser.add_argument("--alt", action="store_true",
                        help="Use alternate stimulus battery (same functions, different wording)")
    parser.add_argument("--compare", action="store_true",
                        help="After profiling, compare with existing run (needs both default and --alt)")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Max tokens to generate per prompt")
    args = parser.parse_args()

    device = get_device(args.device)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\nLoading {args.model}...")
    dtype = torch.float32 if device in ("cpu", "mps") else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(device).eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {params:,} parameters on {device}")

    # ── Profile ──
    batteries = STIMULUS_BATTERIES_ALT if args.alt else None
    battery_tag = "alt" if args.alt else "default"
    print(f"Battery: {battery_tag}")

    profiler = HeadProfiler(
        model, tokenizer,
        capture_attention=not args.no_attention,
        batteries=batteries,
        max_new_tokens=args.max_tokens,
    )

    start = time.time()
    report = profiler.profile(verbose=True)
    elapsed = time.time() - start
    print(f"\nProfiling completed in {elapsed:.1f}s")

    # ── Save JSON ──
    model_safe = args.model.replace("/", "-")
    suffix = "_alt" if args.alt else ""
    json_path = os.path.join(output_dir, f"head_profile_{model_safe}{suffix}.json")
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Results saved: {json_path}")

    # ── Visualize ──
    png_path = os.path.join(output_dir, f"head_profile_{model_safe}{suffix}.png")
    visualize_profile(report, png_path)

    # ── Print highlights ──
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    for role in ["skeptic", "arbitrator", "induction", "factual_recall",
                 "hallucination_prone"]:
        heads = [h for h in report.head_roles
                 if h.primary_role == role and h.confidence > 0.3]
        heads.sort(key=lambda h: -h.confidence)
        if heads:
            print(f"\n  {role.upper()} heads:")
            for h in heads[:5]:
                print(f"    L{h.layer}H{h.head}: {h.confidence:.0%} — {h.description}")

    # ── Compare with other run ──
    if args.compare or args.alt:
        other_suffix = "" if args.alt else "_alt"
        other_path = os.path.join(output_dir, f"head_profile_{model_safe}{other_suffix}.json")
        if os.path.exists(other_path):
            print("\n" + "=" * 60)
            print("STABILITY ANALYSIS: default vs alt battery")
            print("=" * 60)
            compare_runs(json_path, other_path, output_dir, model_safe)
        else:
            which = "default" if args.alt else "--alt"
            print(f"\nRun with {which} to enable comparison.")

    print("\nDone.")


def compare_runs(path_a, path_b, output_dir, model_safe):
    """Compare two profiling runs to measure classification stability."""
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    heads_a = {(h["layer"], h["head"]): h for h in a["heads"]}
    heads_b = {(h["layer"], h["head"]): h for h in b["heads"]}

    n_total = len(heads_a)
    n_match = 0
    n_close = 0  # Same primary or secondary matches
    mismatches = []

    for key in sorted(heads_a.keys()):
        ha = heads_a[key]
        hb = heads_b.get(key)
        if hb is None:
            continue

        if ha["primary_role"] == hb["primary_role"]:
            n_match += 1
            n_close += 1
        elif (ha["primary_role"] == hb.get("secondary_role") or
              ha.get("secondary_role") == hb["primary_role"]):
            n_close += 1
            mismatches.append((key, ha["primary_role"], hb["primary_role"], "secondary_overlap"))
        else:
            mismatches.append((key, ha["primary_role"], hb["primary_role"], "full_mismatch"))

    exact_pct = n_match / n_total
    close_pct = n_close / n_total
    print(f"\n  Exact match:   {n_match}/{n_total} ({exact_pct:.0%})")
    print(f"  Close match:   {n_close}/{n_total} ({close_pct:.0%})")
    print(f"  Full mismatch: {n_total - n_close}/{n_total} ({1 - close_pct:.0%})")

    # Role-level stability
    print("\n  Per-role stability:")
    all_roles = sorted(set(h["primary_role"] for h in a["heads"]))
    for role in all_roles:
        keys_a = {(h["layer"], h["head"]) for h in a["heads"] if h["primary_role"] == role}
        keys_b = {(h["layer"], h["head"]) for h in b["heads"] if h["primary_role"] == role}
        overlap = keys_a & keys_b
        union = keys_a | keys_b
        jaccard = len(overlap) / len(union) if union else 0
        print(f"    {role:22s}: A={len(keys_a):3d}  B={len(keys_b):3d}  "
              f"overlap={len(overlap):3d}  Jaccard={jaccard:.2f}")

    # Confidence correlation
    conf_pairs = []
    for key in sorted(heads_a.keys()):
        if key in heads_b:
            conf_pairs.append((heads_a[key]["confidence"], heads_b[key]["confidence"]))
    if conf_pairs:
        mean_a = sum(c[0] for c in conf_pairs) / len(conf_pairs)
        mean_b = sum(c[1] for c in conf_pairs) / len(conf_pairs)
        cov = sum((a - mean_a) * (b - mean_b) for a, b in conf_pairs) / len(conf_pairs)
        std_a = max((sum((a - mean_a)**2 for a, _ in conf_pairs) / len(conf_pairs))**0.5, 1e-8)
        std_b = max((sum((b - mean_b)**2 for _, b in conf_pairs) / len(conf_pairs))**0.5, 1e-8)
        corr = cov / (std_a * std_b)
        print(f"\n  Confidence correlation: r = {corr:.3f}")

    # Key heads stability
    print("\n  Key head stability (high-confidence from run A):")
    sorted_a = sorted(a["heads"], key=lambda h: -h["confidence"])
    for ha in sorted_a[:20]:
        key = (ha["layer"], ha["head"])
        hb = heads_b.get(key)
        if hb:
            match = "MATCH" if ha["primary_role"] == hb["primary_role"] else "CHANGED"
            print(f"    L{ha['layer']:2d}H{ha['head']:2d}: "
                  f"{ha['primary_role']:22s} ({ha['confidence']:.2f}) → "
                  f"{hb['primary_role']:22s} ({hb['confidence']:.2f})  {match}")

    # Most volatile heads
    if mismatches:
        full_mismatches = [m for m in mismatches if m[3] == "full_mismatch"]
        if full_mismatches:
            print(f"\n  Most volatile heads ({len(full_mismatches)} full mismatches):")
            for (l, h), role_a, role_b, _ in full_mismatches[:15]:
                print(f"    L{l:2d}H{h:2d}: {role_a:22s} → {role_b}")

    # Save comparison
    comp_path = os.path.join(output_dir, f"head_profile_{model_safe}_stability.json")
    with open(comp_path, "w") as f:
        json.dump({
            "exact_match": n_match,
            "close_match": n_close,
            "total": n_total,
            "exact_pct": round(exact_pct, 4),
            "close_pct": round(close_pct, 4),
            "confidence_correlation": round(corr, 4) if conf_pairs else None,
            "mismatches": [
                {"layer": m[0][0], "head": m[0][1],
                 "role_a": m[1], "role_b": m[2], "type": m[3]}
                for m in mismatches
            ],
        }, f, indent=2)
    print(f"\n  Stability report saved: {comp_path}")


if __name__ == "__main__":
    main()

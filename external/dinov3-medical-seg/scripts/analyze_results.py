#!/usr/bin/env python3
"""
Analyze and visualize DINOv3 few-shot experiment results.

Reads results.json from batch experiments and produces:
1. Summary table per organ
2. Decoder comparison (2D vs 3D)
3. K-shot scaling analysis
4. Fine-tuning method comparison
5. Markdown report

Usage:
    python scripts/analyze_results.py --results results.json
    python scripts/analyze_results.py --results results.json --output report.md
    python scripts/analyze_results.py --results results.json --plot
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

DECODER_TYPE_MAP = {
    "conv2d": "2D Conv",
    "conv2d_unet": "2D U-Net",
    "conv2d_deeplab": "2D DeepLab",
    "conv2d_2_5d": "2.5D Conv",
    "linear3d": "3D Linear",
    "mlp_probe": "3D MLP Probe",
    "segformer3d": "3D SegFormer",
    "dpt3d": "3D DPT",
}

ORGAN_LABELS = {
    "brain": "Brain",
    "liver": "Liver",
    "aorta": "Aorta",
    "scapula_left": "Scapula (L)",
    "adrenal_gland_right": "Adrenal (R)",
    "heart": "Heart (LA)",
    "prostate": "Prostate",
}

DIFFICULTY = {
    "brain": "Easy",
    "liver": "Medium",
    "aorta": "Hard",
    "scapula_left": "Hard",
    "adrenal_gland_right": "Very Hard",
}

# ──────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────

def load_results(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def parse_exp_id(exp_id: str) -> Dict:
    """Parse experiment ID like 'liver_frozen_segformer3d_k5_sz224'."""
    parts = exp_id.split("_")
    result = {"organ": parts[0] if parts else "?"}

    # Find known fields
    remaining = parts[1:]
    for field in ["finetune", "decoder", "k_shot", "img_size"]:
        for i, p in enumerate(remaining):
            if field == "finetune" and p in ("frozen", "lora", "adapter", "full"):
                result["finetune"] = p
                remaining.pop(i)
                break
            elif field == "decoder" and (
                p in ("linear3d", "mlp_probe", "segformer3d", "dpt3d",
                      "conv2d", "conv2d_unet", "conv2d_deeplab", "conv2d_2_5d")
            ):
                result["decoder"] = p
                remaining.pop(i)
                break
            elif field == "k_shot" and p.startswith("k"):
                try:
                    result["k_shot"] = int(p[1:])
                    remaining.pop(i)
                except ValueError:
                    pass
                break
            elif field == "img_size" and p.startswith("sz"):
                try:
                    result["img_size"] = int(p[2:])
                    remaining.pop(i)
                except ValueError:
                    pass
                break

    return result


def compute_summary(results: Dict) -> List[Dict]:
    """Compute per-organ summary statistics."""
    exps = results.get("experiments", {})
    organ_stats = defaultdict(lambda: {
        "total": 0, "completed": 0, "failed": 0,
        "dsc_values": [], "best_config": None, "best_dsc": -1,
    })

    for exp_id, result in exps.items():
        parsed = parse_exp_id(exp_id)
        organ = parsed.get("organ", "unknown")
        stats = organ_stats[organ]
        stats["total"] += 1

        if result.get("status") == "completed":
            stats["completed"] += 1
            dsc = (result.get("metrics") or {}).get("best_dsc")
            if dsc is not None:
                stats["dsc_values"].append(dsc)
                if dsc > stats["best_dsc"]:
                    stats["best_dsc"] = dsc
                    stats["best_config"] = exp_id
        elif result.get("status") == "failed":
            stats["failed"] += 1

    return [
        {
            "organ": organ,
            "label": ORGAN_LABELS.get(organ, organ),
            "difficulty": DIFFICULTY.get(organ, "Unknown"),
            **stats,
            "mean_dsc": (
                sum(stats["dsc_values"]) / len(stats["dsc_values"])
                if stats["dsc_values"] else None
            ),
        }
        for organ, stats in sorted(organ_stats.items())
    ]


# ──────────────────────────────────────────────
# Markdown report generation
# ──────────────────────────────────────────────

def generate_markdown_report(results: Dict) -> str:
    """Generate a comprehensive markdown experiment report."""
    exps = results.get("experiments", {})
    if not exps:
        return "# No experiment results found.\n"

    summary = compute_summary(results)
    meta = results.get("meta", {})

    lines = []
    lines.append("# DINOv3 Few-Shot Fine-Tuning Experiment Report\n")
    lines.append(f"*Generated: {meta.get('updated', meta.get('created', 'N/A'))}*\n")

    # ── Overview ──
    lines.append("## 1. Overview\n")
    total = sum(s["total"] for s in summary)
    completed = sum(s["completed"] for s in summary)
    failed = sum(s["failed"] for s in summary)
    lines.append(f"- **Total experiments**: {total}")
    lines.append(f"- **Completed**: {completed}")
    lines.append(f"- **Failed**: {failed}")
    lines.append(f"- **Success rate**: {completed/total*100:.1f}%" if total > 0 else "")
    lines.append("")

    # ── Per-organ summary ──
    lines.append("## 2. Per-Organ Summary\n")
    lines.append("| Organ | Difficulty | Experiments | Completed | Best DSC | Mean DSC |")
    lines.append("|-------|-----------|-------------|-----------|----------|----------|")
    for s in summary:
        best = f"{s['best_dsc']:.4f}" if s['best_dsc'] > 0 else "N/A"
        mean = f"{s['mean_dsc']:.4f}" if s['mean_dsc'] else "N/A"
        lines.append(
            f"| {s['label']} | {s['difficulty']} | {s['total']} | "
            f"{s['completed']} | {best} | {mean} |"
        )
    lines.append("")

    # ── Decoder comparison (2D vs 3D) ──
    lines.append("## 3. Decoder Architecture Comparison (2D vs 3D)\n")
    lines.append("### 3.1 Per-Organ Best Decoder\n")
    lines.append("| Organ | Best Decoder | DSC | 2D or 3D |")
    lines.append("|-------|-------------|-----|----------|")

    decoder_results = defaultdict(list)
    for exp_id, result in exps.items():
        if result.get("status") != "completed":
            continue
        parsed = parse_exp_id(exp_id)
        dsc = (result.get("metrics") or {}).get("best_dsc")
        if dsc is not None:
            decoder_results[parsed.get("organ", "?")].append({
                "decoder": parsed.get("decoder", "?"),
                "dsc": dsc,
                "exp_id": exp_id,
            })

    for organ, entries in sorted(decoder_results.items()):
        best = max(entries, key=lambda x: x["dsc"])
        is_2d = "2D" if best["decoder"].startswith("conv") else "3D"
        lines.append(
            f"| {ORGAN_LABELS.get(organ, organ)} | {best['decoder']} | "
            f"{best['dsc']:.4f} | {is_2d} |"
        )
    lines.append("")

    # ── K-shot scaling ──
    lines.append("## 4. K-Shot Scaling Analysis\n")
    lines.append("| Organ | K=1 | K=3 | K=5 | K=10 |")
    lines.append("|-------|-----|-----|-----|------|")

    for organ in sorted(decoder_results.keys()):
        row = [ORGAN_LABELS.get(organ, organ)]
        for k in [1, 3, 5, 10]:
            k_results = [
                e for e in decoder_results[organ]
                if parse_exp_id(e["exp_id"]).get("k_shot") == k
            ]
            if k_results:
                best_k = max(k_results, key=lambda x: x["dsc"])
                row.append(f"{best_k['dsc']:.4f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Fine-tuning method comparison ──
    lines.append("## 5. Fine-Tuning Method Comparison\n")
    lines.append("| Organ | Frozen | LoRA | Adapter | Full | Best |")
    lines.append("|-------|--------|------|---------|------|------|")

    for organ in sorted(decoder_results.keys()):
        row = [ORGAN_LABELS.get(organ, organ)]
        best_method = ("frozen", -1)
        for method in ["frozen", "lora", "adapter", "full"]:
            m_results = [
                e for e in decoder_results[organ]
                if parse_exp_id(e["exp_id"]).get("finetune") == method
            ]
            if m_results:
                best_m = max(m_results, key=lambda x: x["dsc"])
                row.append(f"{best_m['dsc']:.4f}")
                if best_m["dsc"] > best_method[1]:
                    best_method = (method, best_m["dsc"])
            else:
                row.append("—")
        row.append(best_method[0])
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Recommendations ──
    lines.append("## 6. Recommendations\n")
    lines.append("### Per-Organ Best Configuration\n")
    lines.append("| Organ | Architecture | Finetune | K-Shot | DSC | Rating |")
    lines.append("|-------|-------------|----------|--------|-----|--------|")

    for organ, entries in sorted(decoder_results.items()):
        if not entries:
            continue
        best = max(entries, key=lambda x: x["dsc"])
        parsed = parse_exp_id(best["exp_id"])
        dsc = best["dsc"]
        rating = (
            "✅ Excellent" if dsc >= 0.90 else
            "🟢 Good" if dsc >= 0.80 else
            "🟡 Fair" if dsc >= 0.70 else
            "🟠 Poor" if dsc >= 0.60 else
            "🔴 Unusable"
        )
        lines.append(
            f"| {ORGAN_LABELS.get(organ, organ)} | "
            f"{best['decoder']} | {parsed.get('finetune', '?')} | "
            f"{parsed.get('k_shot', '?')} | {dsc:.4f} | {rating} |"
        )
    lines.append("")

    # ── 2D vs 3D verdict ──
    lines.append("## 7. 2D vs 3D Decoder Verdict\n")
    lines.append("| Organ | Best 2D DSC | Best 3D DSC | Winner | Δ DSC |")
    lines.append("|-------|-------------|-------------|--------|-------|")

    for organ, entries in sorted(decoder_results.items()):
        d2 = [e["dsc"] for e in entries if e["decoder"].startswith("conv")]
        d3 = [e["dsc"] for e in entries if not e["decoder"].startswith("conv")]
        best_2d = max(d2) if d2 else 0
        best_3d = max(d3) if d3 else 0
        winner = "2D ✅" if best_2d > best_3d else "3D ✅" if best_3d > best_2d else "Tie"
        delta = abs(best_2d - best_3d)
        lines.append(
            f"| {ORGAN_LABELS.get(organ, organ)} | "
            f"{best_2d:.4f} | {best_3d:.4f} | {winner} | {delta:.4f} |"
        )
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze DINOv3 few-shot experiment results"
    )
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument("--output", "-o", help="Output markdown report path")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        sys.exit(1)

    results = load_results(args.results)

    if args.summary:
        summary = compute_summary(results)
        print(f"{'Organ':<20s} {'Difficulty':<12s} {'Exps':<6s} {'Done':<6s} "
              f"{'Mean DSC':<10s} {'Best DSC':<10s} {'Best Config'}")
        print("-" * 90)
        for s in summary:
            mean = f"{s['mean_dsc']:.4f}" if s['mean_dsc'] else "N/A"
            best = f"{s['best_dsc']:.4f}" if s['best_dsc'] > 0 else "N/A"
            print(f"{s['label']:<20s} {s['difficulty']:<12s} {s['total']:<6d} "
                  f"{s['completed']:<6d} {mean:<10s} {best:<10s} "
                  f"{s.get('best_config', 'N/A')}")
        return

    report = generate_markdown_report(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)

    if args.plot:
        try:
            _generate_plots(results)
        except ImportError:
            print("\nMatplotlib not available. Install with: pip install matplotlib")


def _generate_plots(results: Dict):
    """Generate visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    exps = results.get("experiments", {})
    output_dir = Path(results.get("_path", ".")).parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: K-shot scaling per organ
    fig, ax = plt.subplots(figsize=(10, 6))
    organ_data = defaultdict(lambda: defaultdict(list))
    for exp_id, result in exps.items():
        if result.get("status") != "completed":
            continue
        parsed = parse_exp_id(exp_id)
        dsc = (result.get("metrics") or {}).get("best_dsc")
        if dsc is not None:
            organ_data[parsed.get("organ", "?")][parsed.get("k_shot", 0)].append(dsc)

    for organ, k_data in sorted(organ_data.items()):
        ks = sorted(k_data.keys())
        means = [np.mean(k_data[k]) for k in ks]
        ax.plot(ks, means, "o-", label=ORGAN_LABELS.get(organ, organ))

    ax.set_xlabel("K-shot (training samples)")
    ax.set_ylabel("Best Dice Score")
    ax.set_title("Few-Shot Scaling: DSC vs Training Samples")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "kshot_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_dir / 'kshot_scaling.png'}")

    # Plot 2: 2D vs 3D decoder comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    organs = sorted(set(parse_exp_id(eid).get("organ", "?") for eid in exps))
    x = np.arange(len(organs))
    width = 0.15

    for i, (dec_key, dec_label) in enumerate([
        ("conv2d", "2D Conv"), ("conv2d_unet", "2D U-Net"),
        ("segformer3d", "3D SegFormer"), ("dpt3d", "3D DPT"),
        ("mlp_probe", "3D MLP"), ("linear3d", "3D Linear"),
    ]):
        scores = []
        for organ in organs:
            org_exps = [
                (eid, r) for eid, r in exps.items()
                if r.get("status") == "completed"
                and parse_exp_id(eid).get("organ") == organ
                and parse_exp_id(eid).get("decoder") == dec_key
            ]
            dscs = [(r.get("metrics") or {}).get("best_dsc", 0) for _, r in org_exps]
            scores.append(max(dscs) if dscs else 0)
        ax.bar(x + i * width, scores, width, label=dec_label)

    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels([ORGAN_LABELS.get(o, o) for o in organs])
    ax.set_ylabel("Best Dice Score")
    ax.set_title("Decoder Architecture Comparison (2D vs 3D)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(output_dir / "decoder_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_dir / 'decoder_comparison.png'}")


if __name__ == "__main__":
    main()

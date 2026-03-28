#!/usr/bin/env python3
"""
systematic_benchmark.py — Systematischer Vergleich der Ellipsoid-SDF-Methoden.

Führt run_benchmark über einen konfigurierbaren Parameterraum aus und
ermittelt, welche Approximation in welchem Bereich (Gesamt, Interior,
Exterior, Oberfläche) am besten abschneidet.

Ausgabe:
  - Konsolentabelle mit Ergebnissen
  - CSV-Export (optional)
  - PDF-Zusammenfassung (optional, via Matplotlib)

Nutzung:
    python systematic_benchmark.py
    python systematic_benchmark.py --csv results.csv
    python systematic_benchmark.py --pdf report.pdf
    python systematic_benchmark.py --csv results.csv --pdf report.pdf --grid 128
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from benchmark_sdf import run_benchmark, METHODS


# ══════════════════════════════════════════════════════════════════════════════
# TEST CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════════

TEST_CONFIGS = [
    # (label, (rx, ry, rz))
    # ── Sphärisch ─────────────────────────────────────────────────────
    ("Kugel",                      (1.0,  1.0,  1.0)),
    ("Kugel (klein)",              (0.3,  0.3,  0.3)),

    # ── Milde Aspektverhältnisse (κ ≈ 1.5–2) ─────────────────────────
    ("Mild κ≈1.4",                 (1.0,  0.8,  0.7)),
    ("Mild κ≈2.0",                 (1.0,  0.7,  0.5)),

    # ── Moderate (κ ≈ 3–5) ────────────────────────────────────────────
    ("Moderat κ≈3.3",              (1.0,  0.5,  0.3)),
    ("Moderat κ≈5.0",              (1.0,  0.4,  0.2)),

    # ── Hoch (κ ≈ 10) ────────────────────────────────────────────────
    ("Hoch κ≈10",                  (1.0,  0.3,  0.1)),
    ("Dünne Scheibe κ≈10",        (1.0,  1.0,  0.1)),
    ("Lange Nadel κ≈10",          (1.0,  0.1,  0.1)),

    # ── Sehr hoch (κ ≈ 20+) ──────────────────────────────────────────
    ("Extrem κ≈20",                (1.0,  0.2,  0.05)),
    ("Extrem Nadel κ≈33",         (1.0,  0.05, 0.03)),

    # ── Verschiedene Skalen ───────────────────────────────────────────
    ("Groß κ≈3.3",                 (5.0,  2.5,  1.5)),
    ("Klein κ≈3.3",                (0.2,  0.1,  0.06)),
]

REGIONS = ["total", "interior", "exterior", "near_surface"]
REGION_LABELS = {
    "total":        "Gesamt",
    "interior":     "Interior",
    "exterior":     "Exterior",
    "near_surface": "Oberfläche (±15%)",
}
METRIC_KEYS = ["mae", "rmse", "l_inf"]


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

def run_all(grid_n: int = 128) -> list[dict]:
    """Run benchmark for every test config, return list of result dicts."""
    results = []
    total = len(TEST_CONFIGS)

    for i, (label, radii) in enumerate(TEST_CONFIGS):
        kappa = max(radii) / min(radii)
        print(f"[{i+1}/{total}]  {label:30s}  r=({radii[0]:.2f}, {radii[1]:.2f}, {radii[2]:.2f})"
              f"  κ={kappa:.1f}  ...", end="", flush=True)

        t0 = time.perf_counter()
        res = run_benchmark(radii, grid_n=grid_n, timing_repeats=3)
        elapsed = time.perf_counter() - t0

        res["label"] = label
        results.append(res)
        print(f"  {elapsed:.1f}s")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def build_summary_table(results: list[dict]) -> list[dict]:
    """Build flat rows: one per (config × method × region)."""
    rows = []
    for res in results:
        sl = res["slices"]["XY"]
        kappa = res["aspect_ratio"]
        for name, _, short in METHODS:
            md = sl["methods"][name]
            for region in REGIONS:
                m = md["metrics"][region]
                rows.append({
                    "label":  res["label"],
                    "rx":     res["radii"][0],
                    "ry":     res["radii"][1],
                    "rz":     res["radii"][2],
                    "kappa":  kappa,
                    "method": name,
                    "short":  short,
                    "region": region,
                    "mae":    m["mae"],
                    "rmse":   m["rmse"],
                    "l_inf":  m["l_inf"],
                })
    return rows


def find_best_methods(rows: list[dict]) -> dict:
    """For each (config, region, metric), find the winning method.

    Returns dict[(label, region, metric)] = (best_method, best_value).
    """
    # Group by (label, region)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["label"], r["region"])].append(r)

    winners = {}
    for (label, region), group in grouped.items():
        for metric in METRIC_KEYS:
            best = min(group, key=lambda r: r[metric])
            winners[(label, region, metric)] = (best["method"], best[metric])
    return winners


def print_console_report(results: list[dict], rows: list[dict], winners: dict):
    """Print a structured console report."""
    print("\n" + "═" * 100)
    print("  SYSTEMATISCHER SDF-BENCHMARK — ERGEBNISSE")
    print("═" * 100)

    # ── Per-config summary ────────────────────────────────────────────
    for res in results:
        label = res["label"]
        radii = res["radii"]
        kappa = res["aspect_ratio"]
        sl = res["slices"]["XY"]

        print(f"\n{'─' * 100}")
        print(f"  {label}   r=({radii[0]:.2f}, {radii[1]:.2f}, {radii[2]:.2f})   κ={kappa:.1f}")
        print(f"{'─' * 100}")

        for region in REGIONS:
            region_label = REGION_LABELS[region]
            print(f"\n    {region_label}:")
            print(f"    {'Methode':<25s}  {'MAE':>10s}  {'RMSE':>10s}  {'L∞':>10s}")
            print(f"    {'─' * 60}")

            for name, _, short in METHODS:
                md = sl["methods"][name]["metrics"][region]
                # Mark winners
                marks = ""
                for metric in METRIC_KEYS:
                    w_method, _ = winners[(label, region, metric)]
                    if w_method == name:
                        marks += " ★"
                print(f"    {name:<25s}  {md['mae']:>10.6f}  {md['rmse']:>10.6f}"
                      f"  {md['l_inf']:>10.6f}{marks}")

    # ── Global win counts ─────────────────────────────────────────────
    print(f"\n\n{'═' * 100}")
    print("  GEWINN-ZUSAMMENFASSUNG (★ = Beste Methode pro Konfiguration)")
    print(f"{'═' * 100}\n")

    for region in REGIONS:
        region_label = REGION_LABELS[region]
        print(f"  {region_label}:")

        for metric in METRIC_KEYS:
            metric_label = {"mae": "MAE", "rmse": "RMSE", "l_inf": "L∞"}[metric]
            counts = defaultdict(int)
            for res in results:
                w_method, _ = winners[(res["label"], region, metric)]
                counts[w_method] += 1

            sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
            parts = [f"{name}: {cnt}/{len(results)}" for name, cnt in sorted_counts]
            print(f"    {metric_label:<6s}  {' | '.join(parts)}")
        print()

    # ── Recommendations ───────────────────────────────────────────────
    print(f"{'═' * 100}")
    print("  EMPFEHLUNGEN")
    print(f"{'═' * 100}\n")

    # Group configs by aspect ratio ranges
    low_kappa  = [r for r in results if r["aspect_ratio"] <= 2.0]
    mid_kappa  = [r for r in results if 2.0 < r["aspect_ratio"] <= 5.0]
    high_kappa = [r for r in results if 5.0 < r["aspect_ratio"] <= 15.0]
    ext_kappa  = [r for r in results if r["aspect_ratio"] > 15.0]

    for group_label, group in [
        ("Niedrig (κ ≤ 2)", low_kappa),
        ("Moderat (2 < κ ≤ 5)", mid_kappa),
        ("Hoch (5 < κ ≤ 15)", high_kappa),
        ("Extrem (κ > 15)", ext_kappa),
    ]:
        if not group:
            continue

        print(f"  {group_label}:")

        for region in ["total", "interior"]:
            region_label = REGION_LABELS[region]
            mae_counts = defaultdict(list)
            for res in group:
                w_method, w_val = winners[(res["label"], region, "mae")]
                mae_counts[w_method].append(w_val)

            best_method = max(mae_counts.items(), key=lambda x: len(x[1]))[0]
            avg_mae = np.mean(mae_counts[best_method])
            win_rate = len(mae_counts[best_method]) / len(group) * 100
            print(f"    {region_label:<20s} → {best_method:<25s}"
                  f"  (Gewinnrate {win_rate:.0f}%, Ø MAE = {avg_mae:.6f})")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_csv(rows: list[dict], path: str):
    """Export all results as CSV."""
    fieldnames = ["label", "rx", "ry", "rz", "kappa",
                  "method", "region", "mae", "rmse", "l_inf"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nCSV exportiert: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_pdf(results: list[dict], rows: list[dict], winners: dict, path: str):
    """Generate a multi-page PDF with heatmaps and summary charts."""
    method_names = [name for name, _, _ in METHODS]
    method_shorts = [short for _, _, short in METHODS]
    n_methods = len(method_names)
    n_configs = len(results)

    bg_color = "#0d1117"
    text_color = "#d0d0d0"

    with PdfPages(path) as pdf:
        # ── Page 1: MAE heatmap per region ────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=bg_color)
        fig.suptitle("MAE pro Methode & Konfiguration",
                     fontsize=14, color=text_color, y=0.98)

        config_labels = [f"{r['label']}\nκ={r['aspect_ratio']:.1f}" for r in results]

        for ax_idx, region in enumerate(REGIONS):
            ax = axes.flat[ax_idx]
            ax.set_facecolor(bg_color)

            # Build matrix: (n_configs × n_methods)
            mat = np.zeros((n_configs, n_methods))
            for i, res in enumerate(results):
                sl = res["slices"]["XY"]
                for j, name in enumerate(method_names):
                    mat[i, j] = sl["methods"][name]["metrics"][region]["mae"]

            im = ax.imshow(mat, cmap="Reds", aspect="auto")
            ax.set_xticks(range(n_methods))
            ax.set_xticklabels(method_shorts, fontsize=8, color=text_color)
            ax.set_yticks(range(n_configs))
            ax.set_yticklabels(config_labels, fontsize=7, color=text_color)
            ax.set_title(REGION_LABELS[region], fontsize=10, color=text_color)

            # Annotate cells with values + mark winners
            for i in range(n_configs):
                for j in range(n_methods):
                    val = mat[i, j]
                    w_method, _ = winners[(results[i]["label"], region, "mae")]
                    star = " ★" if method_names[j] == w_method else ""
                    color = "white" if val > mat.max() * 0.5 else "black"
                    ax.text(j, i, f"{val:.4f}{star}", ha="center", va="center",
                            fontsize=6, color=color)

            ax.tick_params(colors=text_color, labelsize=7)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, facecolor=bg_color)
        plt.close(fig)

        # ── Page 2: Win rate bar chart ────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=bg_color)
        fig.suptitle("Gewinnrate pro Methode & Metrik (alle Regionen)",
                     fontsize=14, color=text_color, y=0.98)

        for ax_idx, metric in enumerate(METRIC_KEYS):
            ax = axes[ax_idx]
            ax.set_facecolor(bg_color)
            metric_label = {"mae": "MAE", "rmse": "RMSE", "l_inf": "L∞"}[metric]

            x = np.arange(n_methods)
            bar_width = 0.18
            region_colors = ["#f2e641", "#4962f2", "#50c878", "#c878ff"]

            for r_idx, region in enumerate(REGIONS):
                counts = []
                for m_name in method_names:
                    cnt = sum(1 for res in results
                              if winners[(res["label"], region, metric)][0] == m_name)
                    counts.append(cnt)
                offset = (r_idx - 1.5) * bar_width
                ax.bar(x + offset, counts, bar_width,
                       label=REGION_LABELS[region],
                       color=region_colors[r_idx], edgecolor="none", alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(method_shorts, fontsize=9, color=text_color)
            ax.set_ylabel("Anzahl Siege", fontsize=9, color=text_color)
            ax.set_title(metric_label, fontsize=11, color=text_color)
            ax.legend(fontsize=7, facecolor="#1a2030", edgecolor="#333",
                      labelcolor=text_color)
            ax.tick_params(colors=text_color, labelsize=7)
            ax.spines[:].set_color("#333")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, facecolor=bg_color)
        plt.close(fig)

        # ── Page 3: MAE vs aspect ratio ───────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=bg_color)
        fig.suptitle("MAE vs. Aspektverhältnis κ",
                     fontsize=14, color=text_color, y=0.98)

        method_colors = ["#f2e641", "#4962f2", "#50c878", "#c878ff", "#ff7f50"]
        for ax_idx, region in enumerate(["total", "interior"]):
            ax = axes[ax_idx]
            ax.set_facecolor(bg_color)

            kappas = [r["aspect_ratio"] for r in results]
            for m_idx, name in enumerate(method_names):
                maes = []
                for res in results:
                    maes.append(res["slices"]["XY"]["methods"][name]["metrics"][region]["mae"])
                ax.scatter(kappas, maes, color=method_colors[m_idx],
                           s=40, alpha=0.8, label=method_shorts[m_idx], zorder=3)
                # Connect with lines (sorted by kappa)
                order = np.argsort(kappas)
                ax.plot(np.array(kappas)[order], np.array(maes)[order],
                        color=method_colors[m_idx], linewidth=0.8, alpha=0.5)

            ax.set_xlabel("Aspektverhältnis κ", fontsize=9, color=text_color)
            ax.set_ylabel("MAE", fontsize=9, color=text_color)
            ax.set_title(REGION_LABELS[region], fontsize=10, color=text_color)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize=8, facecolor="#1a2030", edgecolor="#333",
                      labelcolor=text_color)
            ax.tick_params(colors=text_color, labelsize=7)
            ax.spines[:].set_color("#333")
            ax.grid(True, alpha=0.15, color="#555")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, facecolor=bg_color)
        plt.close(fig)

    print(f"\nPDF exportiert: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Systematischer Vergleich der Ellipsoid-SDF-Approximationen"
    )
    parser.add_argument("--grid", type=int, default=128,
                        help="Grid-Auflösung N (default: 128)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Pfad für CSV-Export")
    parser.add_argument("--pdf", type=str, default=None,
                        help="Pfad für PDF-Report")
    args = parser.parse_args()

    print(f"Grid-Auflösung: {args.grid}")
    print(f"Anzahl Konfigurationen: {len(TEST_CONFIGS)}")
    print(f"Methoden: {', '.join(n for n, _, _ in METHODS)}")
    print()

    t0 = time.perf_counter()
    results = run_all(grid_n=args.grid)
    total_time = time.perf_counter() - t0

    rows = build_summary_table(results)
    winners = find_best_methods(rows)

    print_console_report(results, rows, winners)

    print(f"\nGesamtlaufzeit: {total_time:.1f}s")

    if args.csv:
        export_csv(rows, args.csv)

    if args.pdf:
        export_pdf(results, rows, winners, args.pdf)

    if not args.csv and not args.pdf:
        print("\nTipp: --csv results.csv und/oder --pdf report.pdf für Export.")


if __name__ == "__main__":
    main()
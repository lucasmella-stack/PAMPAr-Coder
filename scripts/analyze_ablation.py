#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
analyze_ablation.py — Análisis y visualización de los resultados de ablación.

Lee ablation_results/*/metrics.jsonl, genera:
  - Gráfica de eval loss vs. paso (ablation_results/figures/eval_curves.pdf)
  - Gráfica de train loss vs. paso (ablation_results/figures/train_curves.pdf)
  - Bar chart de eval loss final (ablation_results/figures/final_eval_bar.pdf)
  - Tabla resumen en stdout + ablation_results/summary.csv

Uso:
  python scripts/analyze_ablation.py
  python scripts/analyze_ablation.py --results-dir /workspace/PAMPAr-Coder/ablation_results
  python scripts/analyze_ablation.py --no-show   # sin ventana de matplotlib
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración de matplotlib sin display (útil en servidor)
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")  # headless por defecto; --show cambia a TkAgg al final
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

EXPERIMENT_ORDER = ["pampar_v3", "no_llaves", "single_stream", "vanilla_gpt"]

LABELS: dict[str, str] = {
    "pampar_v3": "PAMPAr-V3 (control)",
    "no_llaves": "No LLAVES",
    "single_stream": "Single Stream",
    "vanilla_gpt": "Vanilla GPT",
}

COLORS: dict[str, str] = {
    "pampar_v3": "#2563eb",   # azul
    "no_llaves": "#dc2626",   # rojo
    "single_stream": "#16a34a",  # verde
    "vanilla_gpt": "#9333ea",  # violeta
}

LINESTYLES: dict[str, str] = {
    "pampar_v3": "-",
    "no_llaves": "--",
    "single_stream": "-.",
    "vanilla_gpt": ":",
}


# ---------------------------------------------------------------------------
# Lectura de datos
# ---------------------------------------------------------------------------

def load_metrics(results_dir: Path) -> dict[str, dict[str, list]]:
    """
    Lee metrics.jsonl de cada experimento.

    Returns:
        dict[experiment_name -> {"train_steps", "train_avg_loss",
                                  "eval_steps", "eval_loss"}]
    """
    data: dict[str, dict[str, list]] = {}

    for exp in EXPERIMENT_ORDER:
        jsonl_path = results_dir / exp / "metrics.jsonl"
        if not jsonl_path.exists():
            continue

        train_steps: list[int] = []
        train_avg_loss: list[float] = []
        eval_steps: list[int] = []
        eval_loss: list[float] = []

        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if rec.get("type") == "eval":
                    eval_steps.append(rec["step"])
                    eval_loss.append(rec["eval_loss"])
                elif "avg_loss" in rec:
                    train_steps.append(rec["step"])
                    train_avg_loss.append(rec["avg_loss"])

        data[exp] = {
            "train_steps": train_steps,
            "train_avg_loss": train_avg_loss,
            "eval_steps": eval_steps,
            "eval_loss": eval_loss,
        }

    return data


def load_meta(results_dir: Path) -> dict[str, dict]:
    """Lee meta.json de cada experimento."""
    metas: dict[str, dict] = {}
    for exp in EXPERIMENT_ORDER:
        meta_path = results_dir / exp / "meta.json"
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as f:
                metas[exp] = json.load(f)
    return metas


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _apply_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_eval_curves(
    data: dict[str, dict[str, list]], figures_dir: Path, show: bool = False
) -> Path:
    """Gráfica de eval loss vs. paso."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp in EXPERIMENT_ORDER:
        if exp not in data:
            continue
        d = data[exp]
        if not d["eval_steps"]:
            continue
        ax.plot(
            d["eval_steps"],
            d["eval_loss"],
            label=LABELS[exp],
            color=COLORS[exp],
            linestyle=LINESTYLES[exp],
            linewidth=2,
            marker="o",
            markersize=4,
        )

    _apply_style(
        ax,
        title="PAMPAr Ablation — Eval Loss vs. Training Step",
        xlabel="Training step",
        ylabel="Eval cross-entropy loss (↓ better)",
    )

    fig.tight_layout()
    out = figures_dir / "eval_curves.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(figures_dir / "eval_curves.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out


def plot_train_curves(
    data: dict[str, dict[str, list]], figures_dir: Path, show: bool = False
) -> Path:
    """Gráfica de train avg_loss vs. paso (submuestreado para claridad)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp in EXPERIMENT_ORDER:
        if exp not in data:
            continue
        d = data[exp]
        steps = d["train_steps"]
        losses = d["train_avg_loss"]
        if not steps:
            continue

        # Submuestrear: cada 10 puntos para evitar overplotting
        stride = max(1, len(steps) // 300)
        steps_s = steps[::stride]
        losses_s = losses[::stride]

        ax.plot(
            steps_s,
            losses_s,
            label=LABELS[exp],
            color=COLORS[exp],
            linestyle=LINESTYLES[exp],
            linewidth=1.5,
            alpha=0.85,
        )

    _apply_style(
        ax,
        title="PAMPAr Ablation — Train Loss (100-step avg) vs. Training Step",
        xlabel="Training step",
        ylabel="Train cross-entropy loss (↓ better)",
    )

    fig.tight_layout()
    out = figures_dir / "train_curves.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(figures_dir / "train_curves.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out


def plot_final_bar(
    summary_rows: list[dict], figures_dir: Path, show: bool = False
) -> Path:
    """Bar chart de eval loss final."""
    finished = [r for r in summary_rows if r["final_eval_loss"] is not None]
    if not finished:
        print("[WARNING] No hay datos finales para el bar chart.", file=sys.stderr)
        return figures_dir / "final_eval_bar.pdf"

    experiments = [r["experiment"] for r in finished]
    losses = [r["final_eval_loss"] for r in finished]
    colors = [COLORS.get(e, "#888888") for e in experiments]
    labels = [LABELS.get(e, e) for e in experiments]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, losses, color=colors, edgecolor="black", linewidth=0.7, width=0.55)

    # Anotar valores
    for bar, val in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Final eval loss (↓ better)", fontsize=11)
    ax.set_title("PAMPAr Ablation — Final Eval Loss Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(losses) * 1.18)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = figures_dir / "final_eval_bar.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(figures_dir / "final_eval_bar.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Tabla resumen
# ---------------------------------------------------------------------------

def build_summary(
    data: dict[str, dict[str, list]],
    metas: dict[str, dict],
) -> list[dict]:
    """Construye tabla resumen con métricas finales."""
    rows = []
    control_loss: float | None = None

    for exp in EXPERIMENT_ORDER:
        d = data.get(exp)
        meta = metas.get(exp, {})

        final_eval_loss: float | None = None
        final_eval_ppl: float | None = None
        final_step: int | None = None
        max_step: int | None = None

        if d:
            if d["eval_loss"]:
                final_eval_loss = d["eval_loss"][-1]
                final_eval_ppl = round(math.exp(min(final_eval_loss, 20.0)), 2)
                final_step = d["eval_steps"][-1]
            if d["train_steps"]:
                max_step = d["train_steps"][-1]

        n_params = meta.get("config", {}).get("n_params")
        if n_params:
            n_params_str = f"{n_params / 1e6:.1f}M"
        else:
            n_params_str = "—"

        row = {
            "experiment": exp,
            "label": LABELS.get(exp, exp),
            "n_params": n_params_str,
            "max_step": max_step or "—",
            "final_eval_step": final_step or "—",
            "final_eval_loss": final_eval_loss,
            "final_eval_ppl": final_eval_ppl or "—",
            "delta_vs_control": "—",
            "delta_pct": "—",
        }

        if exp == "pampar_v3" and final_eval_loss is not None:
            control_loss = final_eval_loss
            row["delta_vs_control"] = "—"
        elif control_loss is not None and final_eval_loss is not None:
            delta = final_eval_loss - control_loss
            row["delta_vs_control"] = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            row["delta_pct"] = f"+{delta / control_loss * 100:.1f}%" if delta >= 0 else f"{delta / control_loss * 100:.1f}%"

        rows.append(row)

    return rows


def print_summary_table(rows: list[dict]) -> None:
    """Imprime tabla formateada en stdout."""
    header = f"{'Experiment':<18} {'Params':>7} {'Step':>8} {'Eval Loss':>10} {'PPL':>7} {'Δ loss':>10} {'Δ %':>8}"
    print()
    print("=" * 75)
    print("  PAMPAr ABLATION STUDY — RESULTS SUMMARY")
    print("=" * 75)
    print(header)
    print("-" * 75)
    for r in rows:
        loss_str = f"{r['final_eval_loss']:.4f}" if isinstance(r["final_eval_loss"], float) else "pending"
        print(
            f"{r['label']:<18} {r['n_params']:>7} {str(r['final_eval_step']):>8} "
            f"{loss_str:>10} {str(r['final_eval_ppl']):>7} "
            f"{str(r['delta_vs_control']):>10} {str(r['delta_pct']):>8}"
        )
    print("=" * 75)
    print()


def save_csv(rows: list[dict], results_dir: Path) -> Path:
    """Guarda CSV con la tabla resumen."""
    out = results_dir / "summary.csv"
    fieldnames = [
        "experiment", "label", "n_params", "max_step",
        "final_eval_step", "final_eval_loss", "final_eval_ppl",
        "delta_vs_control", "delta_pct",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


# ---------------------------------------------------------------------------
# LaTeX table helper
# ---------------------------------------------------------------------------

def print_latex_table(rows: list[dict]) -> None:
    """Imprime una tabla LaTeX lista para pegar en el paper."""
    print("\n% ─── LaTeX table (paste into paper) ───────────────────────")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Steps} & \textbf{Eval Loss} & \textbf{Perplexity} & \textbf{$\Delta$ vs.\ control} \\")
    print(r"\midrule")
    for r in rows:
        loss_str = f"{r['final_eval_loss']:.3f}" if isinstance(r["final_eval_loss"], float) else r"---"
        ppl_str = f"{r['final_eval_ppl']:.2f}" if isinstance(r["final_eval_ppl"], float) else r"---"
        delta_str = str(r["delta_vs_control"]).replace("+", r"$+$") if r["delta_vs_control"] != "—" else "---"
        step_str = str(r["final_eval_step"]) if r["final_eval_step"] != "—" else r"---"
        name = r"\texttt{" + r["experiment"].replace("_", r"\_") + "}"
        print(f"{name} & {step_str} & {loss_str} & {ppl_str} & {delta_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Ablation results at 30K training steps. \textbf{Lower is better.}}")
    print(r"\label{tab:ablation_results}")
    print(r"\end{table}")
    print("% ──────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PAMPAr ablation results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "ablation_results",
        help="Directory containing per-experiment subdirectories",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively (requires display)",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print LaTeX table ready to paste in paper",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.exists():
        print(f"[ERROR] Directorio de resultados no encontrado: {results_dir}", file=sys.stderr)
        sys.exit(1)

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo resultados desde: {results_dir}")

    data = load_metrics(results_dir)
    metas = load_meta(results_dir)

    found = list(data.keys())
    print(f"Experimentos encontrados: {found or '(ninguno)'}")
    if not found:
        print("[WARNING] No se encontraron métricas. ¿Está el path correcto?")
        sys.exit(0)

    # ── Tabla resumen
    summary = build_summary(data, metas)
    print_summary_table(summary)

    if args.latex:
        print_latex_table(summary)

    # ── CSV
    csv_path = save_csv(summary, results_dir)
    print(f"CSV guardado: {csv_path}")

    # ── Figuras
    if args.show:
        matplotlib.use("TkAgg")
        import importlib
        import matplotlib.pyplot as _plt  # noqa: F401

    eval_out = plot_eval_curves(data, figures_dir, show=args.show)
    print(f"Eval curves → {eval_out}")

    train_out = plot_train_curves(data, figures_dir, show=args.show)
    print(f"Train curves → {train_out}")

    bar_out = plot_final_bar(summary, figures_dir, show=args.show)
    print(f"Final bar chart → {bar_out}")

    print("\nAnálisis completo.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
analyze_phase6_ablation.py — Análisis de las corridas de Fase 6.

Lee `ablation_results/phase6/<name>_seed<S>/{metrics.jsonl, config.json}`
y produce:
  - `summary.csv`: una fila por (config, seed) con val_loss final,
    n_params, tokens/s y eval_loss medio últimos K steps.
  - `summary.md`: tabla agregada (mean ± std sobre seeds).
  - `figures/eval_curves.png`: val_loss vs step, una línea por config
    con banda CI95% sobre seeds.
  - `figures/train_curves.png`: train avg_loss vs step.

Uso:
  python scripts/analyze_phase6_ablation.py \
      --results-dir ablation_results/phase6 \
      --output-dir ablation_results/phase6/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN_RE = re.compile(r"^(?P<name>.+)_seed(?P<seed>\d+)$")


def load_run(run_dir: Path) -> dict[str, Any] | None:
    cfg_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not (cfg_path.exists() and metrics_path.exists()):
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    train_records: list[dict] = []
    eval_records: list[dict] = []
    with metrics_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "train":
                train_records.append(rec)
            elif rec.get("type") == "eval":
                eval_records.append(rec)
    return {
        "config": cfg,
        "train": train_records,
        "eval": eval_records,
    }


def summarize(runs: dict[tuple[str, int], dict]) -> list[dict]:
    rows: list[dict] = []
    for (name, seed), data in sorted(runs.items()):
        cfg = data["config"]
        evals = data["eval"]
        if not evals:
            continue
        last_eval = evals[-1]["eval_loss"]
        last_train = data["train"][-1] if data["train"] else {}
        rows.append(
            {
                "name": name,
                "seed": seed,
                "n_params_M": round(cfg["n_params"] / 1e6, 2),
                "final_eval_loss": round(last_eval, 4),
                "final_eval_ppl": round(math.exp(min(last_eval, 20.0)), 2),
                "final_train_avg_loss": last_train.get("avg_loss", float("nan")),
                "steps_per_sec": last_train.get("steps_per_sec", float("nan")),
                "max_step": last_train.get("step", 0),
            }
        )
    return rows


def aggregate_by_config(rows: list[dict]) -> list[dict]:
    by_name: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_name[r["name"]].append(r)
    out: list[dict] = []
    for name, group in sorted(by_name.items()):
        losses = np.array([r["final_eval_loss"] for r in group], dtype=float)
        out.append(
            {
                "name": name,
                "n_seeds": len(group),
                "n_params_M": group[0]["n_params_M"],
                "eval_loss_mean": round(float(losses.mean()), 4),
                "eval_loss_std": round(
                    float(losses.std(ddof=1) if len(losses) > 1 else 0.0), 4
                ),
                "eval_ppl_mean": round(math.exp(min(float(losses.mean()), 20.0)), 2),
                "steps_per_sec": round(
                    float(np.mean([r["steps_per_sec"] for r in group])), 2
                ),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_md_table(path: Path, agg: list[dict]) -> None:
    if not agg:
        path.write_text("# Phase 6 ablation — sin resultados\n", encoding="utf-8")
        return
    headers = list(agg[0].keys())
    lines = ["# Phase 6 ablation — agregado por config\n"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in agg:
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_eval_curves(runs: dict[tuple[str, int], dict], out_path: Path) -> None:
    by_name: dict[str, list[list[tuple[int, float]]]] = defaultdict(list)
    for (name, _seed), data in runs.items():
        if not data["eval"]:
            continue
        curve = [(r["step"], r["eval_loss"]) for r in data["eval"]]
        by_name[name].append(curve)

    if not by_name:
        return

    plt.figure(figsize=(9, 5.5))
    for name, curves in sorted(by_name.items()):
        all_steps = sorted(set(s for c in curves for s, _ in c))
        if not all_steps:
            continue
        matrix = np.full((len(curves), len(all_steps)), np.nan)
        step_idx = {s: i for i, s in enumerate(all_steps)}
        for i, c in enumerate(curves):
            for s, v in c:
                matrix[i, step_idx[s]] = v
        mean = np.nanmean(matrix, axis=0)
        std = (
            np.nanstd(matrix, axis=0, ddof=1)
            if matrix.shape[0] > 1
            else np.zeros_like(mean)
        )
        ci = 1.96 * std / max(1, math.sqrt(matrix.shape[0]))
        plt.plot(all_steps, mean, label=f"{name} (n={matrix.shape[0]})", linewidth=2)
        plt.fill_between(all_steps, mean - ci, mean + ci, alpha=0.15)

    plt.xlabel("step")
    plt.ylabel("eval_loss")
    plt.title("Phase 6 ablation — eval loss (mean ± CI95)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_train_curves(runs: dict[tuple[str, int], dict], out_path: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    by_name: dict[str, list] = defaultdict(list)
    for (name, _seed), data in runs.items():
        steps = [r["step"] for r in data["train"]]
        avg = [r["avg_loss"] for r in data["train"]]
        if steps:
            by_name[name].append((steps, avg))
    for name, curves in sorted(by_name.items()):
        for steps, avg in curves:
            plt.plot(steps, avg, alpha=0.5, label=name)
    # dedupe legend
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    plt.legend([h for h, _ in uniq], [l for _, l in uniq])
    plt.xlabel("step")
    plt.ylabel("avg_loss (window=100)")
    plt.title("Phase 6 ablation — train loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("ablation_results/phase6"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ablation_results/phase6/analysis"),
    )
    args = p.parse_args()

    if not args.results_dir.exists():
        print(f"No existe {args.results_dir}", file=sys.stderr)
        return 1

    runs: dict[tuple[str, int], dict] = {}
    for run_dir in sorted(args.results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        data = load_run(run_dir)
        if data is None:
            continue
        runs[(m.group("name"), int(m.group("seed")))] = data

    if not runs:
        print("Sin runs validos", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize(runs)
    agg = aggregate_by_config(rows)
    write_csv(args.output_dir / "summary.csv", rows)
    write_csv(args.output_dir / "summary_agg.csv", agg)
    write_md_table(args.output_dir / "summary.md", agg)
    plot_eval_curves(runs, args.output_dir / "figures" / "eval_curves.png")
    plot_train_curves(runs, args.output_dir / "figures" / "train_curves.png")

    print(f"OK — {len(runs)} runs, {len(agg)} configs")
    print(f"Outputs en {args.output_dir}")
    for r in agg:
        print(
            f"  {r['name']:14s}  loss={r['eval_loss_mean']:.4f}±{r['eval_loss_std']:.4f}  "
            f"params={r['n_params_M']}M  n_seeds={r['n_seeds']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

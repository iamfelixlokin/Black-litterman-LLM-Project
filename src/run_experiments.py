"""
Multi-parameter experiment runner for BL + News Fusion backtest
Each experiment saves to results/{experiment_name}/
Generates a comparison summary at the end
"""

import os
import sys
import copy
import yaml
import pickle
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

# ── Experiment definitions ───────────────────────────────────────────────────

EXPERIMENTS = [
    {
        "name": "no_news",
        "desc": "Structured only – no news (baseline)",
        "overrides": {
            "llm.use_news": False,
            "llm.fusion.enabled": False,
        }
    },
    {
        "name": "fusion_default",
        "desc": "News + Dynamic Omega (tau=0.15, sensitivity=0.5)",
        "overrides": {}   # uses base config as-is
    },
    {
        "name": "weak_fusion",
        "desc": "News + Conservative fusion (tau=0.15, sensitivity=0.2)",
        "overrides": {
            "llm.fusion.agreement_sensitivity": 0.2,
        }
    },
]

BASE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Config helpers ───────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_nested(d, dotted_key, value):
    """Set d['a']['b']['c'] from 'a.b.c'."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def save_temp_config(base_cfg, overrides, path):
    cfg = copy.deepcopy(base_cfg)
    for k, v in overrides.items():
        set_nested(cfg, k, v)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


# ── Run one experiment ───────────────────────────────────────────────────────

def run_experiment(exp, base_cfg):
    name = exp["name"]
    desc = exp["desc"]
    out_dir = os.path.join(RESULTS_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    tmp_cfg = os.path.join(os.path.dirname(__file__), f"_tmp_{name}.yaml")
    save_temp_config(base_cfg, exp["overrides"], tmp_cfg)

    print(f"\n{'='*70}")
    print(f"  Experiment: {name}")
    print(f"  Description: {desc}")
    print(f"  Overrides: {exp['overrides']}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")

    env = os.environ.copy()

    result = subprocess.run(
        [sys.executable, "main.py",
         "--config", tmp_cfg,
         "--output-dir", out_dir],
        cwd=os.path.dirname(__file__),
        env=env,
    )

    os.remove(tmp_cfg)

    if result.returncode != 0:
        print(f"  ⚠️  Experiment {name} failed (exit {result.returncode})")
        return False

    print(f"  ✓ {name} complete")
    return True


# ── Load saved results ───────────────────────────────────────────────────────

def load_results(exp_name):
    pkl = os.path.join(RESULTS_ROOT, exp_name, "backtest_results.pkl")
    if not os.path.exists(pkl):
        return None
    with open(pkl, "rb") as f:
        return pickle.load(f)


# ── Comparison summary ───────────────────────────────────────────────────────

STRATEGY_ORDER = ["black_litterman", "markowitz", "equal_weight", "spy_benchmark"]
STRATEGY_LABEL = {
    "black_litterman": "BL",
    "markowitz":       "Markowitz",
    "equal_weight":    "Equal Weight",
    "spy_benchmark":   "SPY",
}

ALL_EXPERIMENTS = [e["name"] for e in EXPERIMENTS]
EXP_LABEL = {
    "no_news":        "No News\n(structured only)",
    "fusion_default": "Fusion Default\n(τ=0.15, sens=0.5)",
    "weak_fusion":    "Weak Fusion\n(τ=0.15, sens=0.2)",
}


def build_comparison_df():
    rows = []
    for exp_name in ALL_EXPERIMENTS:
        r = load_results(exp_name)
        if r is None:
            print(f"  Missing results for {exp_name}, skipping")
            continue
        for strat in STRATEGY_ORDER:
            if strat not in r:
                continue
            v = r[strat]
            pv = v["portfolio_values"]
            total_ret = v["total_return"]
            n_days = len(pv)
            years = n_days / 252
            ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
            vol = pv["returns"].std() * np.sqrt(252)
            risk_free = 0.04
            sharpe = (ann_ret - risk_free) / vol if vol > 0 else 0
            dd = (pv["portfolio_value"] / pv["portfolio_value"].cummax() - 1).min()

            rows.append({
                "experiment": exp_name,
                "strategy":   strat,
                "total_return": total_ret,
                "ann_return":   ann_ret,
                "volatility":   vol,
                "sharpe":       sharpe,
                "max_drawdown": dd,
            })

    return pd.DataFrame(rows)


def plot_cumulative_returns(df_all):
    """One chart per experiment showing all 4 strategies."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax_idx, exp_name in enumerate(ALL_EXPERIMENTS):
        ax = axes[ax_idx]
        r = load_results(exp_name)
        if r is None:
            ax.set_visible(False)
            continue

        for strat in STRATEGY_ORDER:
            if strat not in r:
                continue
            pv = r[strat]["portfolio_values"]["portfolio_value"]
            cum = pv / pv.iloc[0]
            style = "--" if strat == "spy_benchmark" else "-"
            lw = 1.5 if strat != "black_litterman" else 2.5
            ax.plot(pv.index, cum, style, linewidth=lw, label=STRATEGY_LABEL[strat])

        ax.set_title(EXP_LABEL.get(exp_name, exp_name), fontsize=11)
        ax.set_ylabel("Cumulative Return (base=1)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    for i in range(len(ALL_EXPERIMENTS), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Cumulative Returns by Experiment (2025-02 → 2026-01)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(RESULTS_ROOT, "comparison_cumulative_returns.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_bl_comparison(df_all):
    """Bar chart: BL total return across experiments."""
    bl_df = df_all[df_all["strategy"] == "black_litterman"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("total_return", "Total Return", "%.1f%%"),
        ("sharpe",       "Sharpe Ratio", "%.3f"),
        ("max_drawdown", "Max Drawdown", "%.1f%%"),
    ]

    for ax, (col, title, fmt) in zip(axes, metrics):
        values = [bl_df[bl_df["experiment"] == e][col].values[0]
                  if len(bl_df[bl_df["experiment"] == e]) else np.nan
                  for e in ALL_EXPERIMENTS]
        colors = ["#2196F3" if e == "fusion_default" else "#90CAF9" for e in ALL_EXPERIMENTS]
        bars = ax.bar(range(len(ALL_EXPERIMENTS)), values, color=colors)
        ax.set_xticks(range(len(ALL_EXPERIMENTS)))
        ax.set_xticklabels([EXP_LABEL.get(e, e) for e in ALL_EXPERIMENTS],
                           fontsize=8, rotation=15, ha="right")
        ax.set_title(f"BL — {title}")
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                label = fmt % (val * 100 if "%" in fmt else val)
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.001 if val >= 0 else -0.005),
                        label, ha="center", va="bottom", fontsize=8)

    plt.suptitle("BL Strategy Performance Across Experiments", fontsize=13)
    plt.tight_layout()
    out = os.path.join(RESULTS_ROOT, "comparison_bl_performance.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def save_comparison_table(df_all):
    pivot = df_all.pivot_table(
        index=["experiment", "strategy"],
        values=["total_return", "ann_return", "volatility", "sharpe", "max_drawdown"]
    )
    out = os.path.join(RESULTS_ROOT, "comparison_summary.csv")
    pivot.to_csv(out)
    print(f"  Saved: {out}")

    # Print BL rows
    bl = df_all[df_all["strategy"] == "black_litterman"].sort_values("total_return", ascending=False)
    print("\n  BL performance ranking:")
    print(f"  {'Experiment':<22} {'Total Ret':>10} {'Ann Ret':>10} {'Sharpe':>8} {'Max DD':>10}")
    print("  " + "-" * 62)
    for _, row in bl.iterrows():
        print(f"  {row['experiment']:<22} {row['total_return']:>9.2%} "
              f"{row['ann_return']:>9.2%} {row['sharpe']:>8.3f} {row['max_drawdown']:>9.2%}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  BL + Fusion Experiment Runner")
    print(f"  {len(EXPERIMENTS)} new experiments + fusion_default (already done)")
    print("=" * 70)

    base_cfg = load_config(BASE_CONFIG)

    for exp in EXPERIMENTS:
        run_experiment(exp, base_cfg)

    print("\n" + "=" * 70)
    print("  Generating comparison summary...")
    print("=" * 70)

    df_all = build_comparison_df()
    plot_cumulative_returns(df_all)
    plot_bl_comparison(df_all)
    save_comparison_table(df_all)

    print("\n  All done! Results in:")
    for exp_name in ALL_EXPERIMENTS:
        print(f"    results/{exp_name}/")
    print(f"    results/comparison_cumulative_returns.png")
    print(f"    results/comparison_bl_performance.png")
    print(f"    results/comparison_summary.csv")


if __name__ == "__main__":
    main()

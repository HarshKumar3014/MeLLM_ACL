"""
04_analyze.py — Aggregate results, compute statistics, generate figures + LaTeX.

Outputs:
  figures/fig1_asr_heatmap.pdf
  figures/fig2_condition_bars.pdf
  figures/fig3_patching_curves.pdf
  figures/fig4_delta_asr_scatter.pdf
  figures/fig5_cot_fidelity.pdf
  figures/table_main.tex
  figures/table_delta.tex

Usage:
  python 04_analyze.py                # generate all
  python 04_analyze.py --no-patch     # skip patching figure (if data missing)
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from config import MODELS, LANGUAGES, TIERS, RESULTS_DIR, FIGURES_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

CONDITION_LABELS = {
    "en_en": "(a) EN→EN",
    "en_tl": "(b) EN→TL",
    "tl_en": "(c) TL→EN",
    "tl_tl": "(d) TL→TL",
}
TIER_COLORS = {"high": "#2ecc71", "mid": "#f39c12", "low": "#e74c3c"}
MODEL_SHORT = {k: k.split("-")[0].capitalize() for k in MODELS}


def load_data() -> pd.DataFrame:
    """Load merged scored results."""
    path = RESULTS_DIR / "all_scored.parquet"
    df = pd.read_parquet(path)
    df["tier"] = df["target_lang"].map(lambda x: LANGUAGES[x]["tier"])
    df["lang_name"] = df["target_lang"].map(lambda x: LANGUAGES[x]["name"])
    return df


# ═══════════════════════════════════════════════════════════════════════
#  ASR Computation
# ═══════════════════════════════════════════════════════════════════════

def compute_asr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Attack Success Rate per (model, language, condition)."""
    asr = (
        df.groupby(["model", "target_lang", "condition"])["judge_score"]
        .mean()
        .reset_index()
        .rename(columns={"judge_score": "asr"})
    )
    asr["tier"] = asr["target_lang"].map(lambda x: LANGUAGES[x]["tier"])
    asr["lang_name"] = asr["target_lang"].map(lambda x: LANGUAGES[x]["name"])
    return asr


# ═══════════════════════════════════════════════════════════════════════
#  Figure 1: ASR Heatmap
# ═══════════════════════════════════════════════════════════════════════

def fig1_heatmap(asr: pd.DataFrame):
    """ASR heatmap — condition (b) EN→TL across models × languages."""
    data = asr[asr["condition"] == "en_tl"].pivot(
        index="model", columns="target_lang", values="asr"
    )
    # Order columns by resource tier
    lang_order = [l for tier in ["high", "mid", "low"] for l in TIERS[tier]
                  if l in data.columns]
    data = data[lang_order]
    data.index = [MODEL_SHORT.get(m, m) for m in data.index]

    fig, ax = plt.subplots(figsize=(7, 2.5))
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="YlOrRd",
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={"label": "ASR", "shrink": 0.8},
    )
    ax.set_xlabel("Target Language (CoT)")
    ax.set_ylabel("")
    ax.set_title("Attack Success Rate — EN Prompt + Target-Lang CoT (Condition b)")

    # Add tier separators
    tier_boundaries = [len(TIERS["high"]),
                       len(TIERS["high"]) + len(TIERS["mid"])]
    for b in tier_boundaries:
        if b < len(lang_order):
            ax.axvline(x=b, color="black", linewidth=2)

    path = FIGURES_DIR / "fig1_asr_heatmap.pdf"
    fig.savefig(path)
    plt.close()
    print(f"[✓] {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 2: Condition Comparison Bars
# ═══════════════════════════════════════════════════════════════════════

def fig2_condition_bars(asr: pd.DataFrame):
    """Grouped bars: ASR per condition, grouped by model, averaged by tier."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    models = asr["model"].unique()

    for ax, model in zip(axes, models):
        model_data = asr[asr["model"] == model]
        tier_cond = (
            model_data.groupby(["tier", "condition"])["asr"]
            .mean()
            .reset_index()
        )

        tier_order = ["high", "mid", "low"]
        cond_order = ["en_en", "en_tl", "tl_en", "tl_tl"]
        x = np.arange(len(tier_order))
        width = 0.18

        for i, cond in enumerate(cond_order):
            vals = []
            for tier in tier_order:
                row = tier_cond[
                    (tier_cond["tier"] == tier) &
                    (tier_cond["condition"] == cond)
                ]
                vals.append(row["asr"].values[0] if len(row) > 0 else 0)
            ax.bar(x + i * width, vals, width,
                   label=CONDITION_LABELS[cond], alpha=0.85)

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(["High", "Mid", "Low"])
        ax.set_title(MODEL_SHORT.get(model, model))
        ax.set_ylim(0, 1)
        if ax == axes[0]:
            ax.set_ylabel("ASR")

    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.suptitle("ASR by Condition × Resource Tier", y=1.02)
    fig.tight_layout()

    path = FIGURES_DIR / "fig2_condition_bars.pdf"
    fig.savefig(path)
    plt.close()
    print(f"[✓] {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 3: Activation Patching Curves
# ═══════════════════════════════════════════════════════════════════════

def fig3_patching_curves():
    """Layer × refusal recovery, one line per CoT language."""
    path_data = RESULTS_DIR / "activation_patching.parquet"
    if not path_data.exists():
        print("[!] Skipping fig3: no activation patching data")
        return

    df = pd.read_parquet(path_data)
    fig, ax = plt.subplots(figsize=(6, 3.5))

    lang_colors = {"en": "#2ecc71", "hi": "#f39c12", "sw": "#e74c3c"}
    lang_labels = {"en": "English (high)", "hi": "Hindi (mid)", "sw": "Swahili (low)"}

    for lang in ["en", "hi", "sw"]:
        lang_data = df[df["patch_lang"] == lang]
        if lang_data.empty:
            continue
        curve = (
            lang_data.groupby("layer")["refusal_recovery"]
            .agg(["mean", "sem"])
            .reset_index()
        )
        ax.plot(curve["layer"], curve["mean"],
                color=lang_colors[lang], label=lang_labels[lang],
                linewidth=1.5, marker="o", markersize=3)
        ax.fill_between(
            curve["layer"],
            curve["mean"] - curve["sem"],
            curve["mean"] + curve["sem"],
            color=lang_colors[lang], alpha=0.15,
        )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Refusal Recovery Score")
    ax.set_title("Activation Patching: Safety Vector Localization")
    ax.legend()

    path = FIGURES_DIR / "fig3_patching_curves.pdf"
    fig.savefig(path)
    plt.close()
    print(f"[✓] {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 4: Δ-ASR vs Resource Tier
# ═══════════════════════════════════════════════════════════════════════

def fig4_delta_scatter(asr: pd.DataFrame):
    """Scatter: Δ-ASR (condition b minus baseline) vs resource tier."""
    baseline = asr[asr["condition"] == "en_en"][
        ["model", "target_lang", "asr"]
    ].rename(columns={"asr": "asr_base"})

    attack = asr[asr["condition"] == "en_tl"][
        ["model", "target_lang", "asr"]
    ].rename(columns={"asr": "asr_attack"})

    merged = baseline.merge(attack, on=["model", "target_lang"])
    merged["delta_asr"] = merged["asr_attack"] - merged["asr_base"]
    merged["tier"] = merged["target_lang"].map(lambda x: LANGUAGES[x]["tier"])

    tier_numeric = {"high": 0, "mid": 1, "low": 2}
    merged["tier_num"] = merged["tier"].map(tier_numeric)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    for tier, color in TIER_COLORS.items():
        subset = merged[merged["tier"] == tier]
        ax.scatter(
            subset["tier_num"] + np.random.normal(0, 0.05, len(subset)),
            subset["delta_asr"],
            c=color, alpha=0.6, s=40, label=tier.capitalize(),
            edgecolors="white", linewidth=0.5,
        )

    # Regression line
    slope, intercept, r, p, _ = stats.linregress(
        merged["tier_num"], merged["delta_asr"]
    )
    x_line = np.linspace(-0.3, 2.3, 100)
    ax.plot(x_line, slope * x_line + intercept,
            color="black", linewidth=1.5, linestyle="--",
            label=f"r={r:.2f}, p={p:.3f}")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["High", "Mid", "Low"])
    ax.set_xlabel("Resource Tier")
    ax.set_ylabel("Δ ASR (Attack − Baseline)")
    ax.set_title("CoT Language Switching Effect by Resource Tier")
    ax.legend(fontsize=7)

    path = FIGURES_DIR / "fig4_delta_asr_scatter.pdf"
    fig.savefig(path)
    plt.close()
    print(f"[✓] {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 5: CoT Language Fidelity
# ═══════════════════════════════════════════════════════════════════════

def fig5_cot_fidelity(df: pd.DataFrame):
    """
    Verify CoT language actually switched.
    Heuristic: count non-ASCII ratio in <think> block as proxy for
    non-English reasoning.
    """
    attack = df[df["condition"] == "en_tl"].copy()

    def non_ascii_ratio(text: str) -> float:
        if not text or len(text) == 0:
            return 0.0
        # Extract think block if present
        if "<think>" in text and "</think>" in text:
            start = text.index("<think>") + len("<think>")
            end = text.index("</think>")
            text = text[start:end]
        non_ascii = sum(1 for c in text if ord(c) > 127)
        return non_ascii / max(len(text), 1)

    attack["non_ascii"] = attack["generation"].apply(non_ascii_ratio)

    fig, ax = plt.subplots(figsize=(6, 3))
    tier_data = []
    for lang in ["en", "es", "zh", "hi", "ar", "bn", "sw", "my"]:
        subset = attack[attack["target_lang"] == lang]
        if not subset.empty:
            tier_data.append({
                "lang": LANGUAGES[lang]["name"],
                "ratio": subset["non_ascii"].mean(),
                "tier": LANGUAGES[lang]["tier"],
            })

    td = pd.DataFrame(tier_data)
    colors = [TIER_COLORS[t] for t in td["tier"]]
    ax.bar(td["lang"], td["ratio"], color=colors, alpha=0.8, edgecolor="white")
    ax.set_ylabel("Non-ASCII Ratio in CoT Block")
    ax.set_title("CoT Language Fidelity (Condition b)")
    ax.set_ylim(0, 1)

    # Add tier legend
    for tier, color in TIER_COLORS.items():
        ax.bar([], [], color=color, label=tier.capitalize())
    ax.legend(fontsize=7)

    path = FIGURES_DIR / "fig5_cot_fidelity.pdf"
    fig.savefig(path)
    plt.close()
    print(f"[✓] {path}")


# ═══════════════════════════════════════════════════════════════════════
#  LaTeX Tables
# ═══════════════════════════════════════════════════════════════════════

def generate_latex_tables(asr: pd.DataFrame):
    """Generate publication-ready LaTeX tables."""
    # ── Main ASR table ─────────────────────────────────────────────────
    pivot = asr.pivot_table(
        index=["model", "condition"],
        columns="target_lang",
        values="asr",
    )
    lang_order = [l for tier in ["high", "mid", "low"] for l in TIERS[tier]
                  if l in pivot.columns]
    pivot = pivot[lang_order]

    # Bold max ASR per (model, lang) pair
    latex = pivot.to_latex(float_format="%.2f", multirow=True)

    path = FIGURES_DIR / "table_main.tex"
    with open(path, "w") as f:
        f.write(latex)
    print(f"[✓] {path}")

    # ── Δ-ASR table ────────────────────────────────────────────────────
    baseline = asr[asr["condition"] == "en_en"].set_index(
        ["model", "target_lang"]
    )["asr"]

    delta_rows = []
    for _, row in asr.iterrows():
        if row["condition"] == "en_en":
            continue
        base = baseline.get((row["model"], row["target_lang"]), 0)
        delta_rows.append({
            "model": row["model"],
            "condition": row["condition"],
            "target_lang": row["target_lang"],
            "delta_asr": row["asr"] - base,
        })

    delta_df = pd.DataFrame(delta_rows)
    delta_pivot = delta_df.pivot_table(
        index=["model", "condition"],
        columns="target_lang",
        values="delta_asr",
    )
    delta_pivot = delta_pivot[[l for l in lang_order if l in delta_pivot.columns]]
    delta_latex = delta_pivot.to_latex(float_format="%+.2f", multirow=True)

    path = FIGURES_DIR / "table_delta.tex"
    with open(path, "w") as f:
        f.write(delta_latex)
    print(f"[✓] {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Summary Statistics
# ═══════════════════════════════════════════════════════════════════════

def print_summary(asr: pd.DataFrame, df: pd.DataFrame):
    """Print key numbers for abstract / results sections."""
    print(f"\n{'='*60}")
    print("  SUMMARY STATISTICS")
    print(f"{'='*60}")

    print(f"\nTotal generations scored: {len(df):,}")
    print(f"Unique prompts: {df['prompt_id'].nunique()}")
    print(f"Models: {df['model'].nunique()}")

    # Overall ASR by condition
    print("\nASR by condition (averaged across all models × languages):")
    for cond in ["en_en", "en_tl", "tl_en", "tl_tl"]:
        mean = asr[asr["condition"] == cond]["asr"].mean()
        print(f"  {CONDITION_LABELS[cond]}: {mean:.3f}")

    # ASR by tier for condition (b)
    print("\nASR condition (b) by resource tier:")
    attack = asr[asr["condition"] == "en_tl"]
    for tier in ["high", "mid", "low"]:
        mean = attack[attack["tier"] == tier]["asr"].mean()
        print(f"  {tier.capitalize()}: {mean:.3f}")

    # Max single-language ASR
    best = attack.loc[attack["asr"].idxmax()]
    print(f"\nHighest single ASR: {best['asr']:.3f} "
          f"({best['model']}, {best['target_lang']})")

    # Statistical test: condition (a) vs (b) overall
    a_scores = asr[asr["condition"] == "en_en"]["asr"].values
    b_scores = asr[asr["condition"] == "en_tl"]["asr"].values
    min_len = min(len(a_scores), len(b_scores))
    if min_len > 1:
        t, p = stats.ttest_rel(a_scores[:min_len], b_scores[:min_len])
        print(f"\nPaired t-test (a) vs (b): t={t:.3f}, p={p:.4f}")

    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-patch", action="store_true",
                        help="Skip activation patching figure")
    args = parser.parse_args()

    print("[*] Loading data...")
    df = load_data()
    asr = compute_asr(df)

    print(f"[*] Loaded {len(df):,} scored generations")
    print(f"[*] ASR computed for {len(asr)} (model, lang, condition) cells\n")

    # Generate all outputs
    fig1_heatmap(asr)
    fig2_condition_bars(asr)
    fig4_delta_scatter(asr)
    fig5_cot_fidelity(df)

    if not args.no_patch:
        fig3_patching_curves()

    generate_latex_tables(asr)
    print_summary(asr, df)

    print("[✓] All figures and tables generated.")


if __name__ == "__main__":
    main()

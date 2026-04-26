"""
01_prepare_data.py — Load HarmBench + AdvBench, build prompt matrix.

Outputs:
  data/prompts.parquet   — deduplicated prompts with source labels
  data/cot_seeds.json    — CoT prefix strings per language

Usage:
  python 01_prepare_data.py              # full run
  python 01_prepare_data.py --limit 5    # dry-run with 5 prompts
"""

import argparse
import json
import hashlib

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from config import (
    DATA_DIR, COT_SEEDS, LANGUAGES,
    HARMBENCH_DATASET, ADVBENCH_DATASET,
)


def load_harmbench() -> list[dict]:
    """Load HarmBench behaviors. Returns list of {prompt, source}."""
    print("[*] Loading HarmBench...")
    try:
        ds = load_dataset(HARMBENCH_DATASET, split="test")
        prompts = []
        for row in ds:
            # HarmBench stores harmful behaviors in 'behavior' or 'goal' field
            text = row.get("behavior") or row.get("goal") or row.get("prompt", "")
            if text.strip():
                prompts.append({"prompt": text.strip(), "source": "harmbench"})
        print(f"    Loaded {len(prompts)} prompts from HarmBench")
        return prompts
    except Exception as e:
        print(f"    [!] HarmBench load failed: {e}")
        print("    [*] Trying alternative: JailbreakBench/JBB-Behaviors...")
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        prompts = [{"prompt": row["Goal"].strip(), "source": "harmbench_jbb"}
                   for row in ds if row.get("Goal", "").strip()]
        print(f"    Loaded {len(prompts)} prompts from JBB-Behaviors")
        return prompts


def load_advbench() -> list[dict]:
    """Load AdvBench harmful behaviors. Returns list of {prompt, source}."""
    print("[*] Loading AdvBench...")
    try:
        ds = load_dataset(ADVBENCH_DATASET, split="harmful")
        prompts = []
        for row in ds:
            text = row.get("prompt") or row.get("goal") or row.get("text", "")
            if text.strip():
                prompts.append({"prompt": text.strip(), "source": "advbench"})
        print(f"    Loaded {len(prompts)} prompts from AdvBench")
        return prompts
    except Exception as e:
        print(f"    [!] AdvBench gated/unavailable: {e}")
        print("    [*] Trying llm-attacks CSV fallback...")
        try:
            url = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                   "main/data/advbench/harmful_behaviors.csv")
            df = pd.read_csv(url)
            col = "goal" if "goal" in df.columns else df.columns[0]
            prompts = [{"prompt": row[col].strip(), "source": "advbench_csv"}
                       for _, row in df.iterrows() if str(row[col]).strip()]
            print(f"    Loaded {len(prompts)} prompts from CSV fallback")
            return prompts
        except Exception as e2:
            print(f"    [!] CSV fallback also failed: {e2}")
            return []


def deduplicate(prompts: list[dict]) -> pd.DataFrame:
    """Deduplicate prompts by normalized text hash."""
    seen = set()
    unique = []
    for p in prompts:
        h = hashlib.md5(p["prompt"].lower().strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(p)
    print(f"[*] Deduplication: {len(prompts)} → {len(unique)} unique prompts")
    return pd.DataFrame(unique)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit prompts for dry-run testing")
    args = parser.parse_args()

    # ── Load benchmarks ────────────────────────────────────────────────
    all_prompts = load_harmbench() + load_advbench()
    df = deduplicate(all_prompts)

    if args.limit:
        df = df.head(args.limit)
        print(f"[*] Limited to {args.limit} prompts (dry-run)")

    # Add prompt IDs
    df.insert(0, "prompt_id", range(len(df)))

    # ── Save prompts ───────────────────────────────────────────────────
    prompt_path = DATA_DIR / "prompts.parquet"
    df.to_parquet(prompt_path, index=False)
    print(f"[✓] Saved {len(df)} prompts → {prompt_path}")

    # ── Save CoT seeds ─────────────────────────────────────────────────
    seeds_path = DATA_DIR / "cot_seeds.json"
    seeds_out = {}
    for lang, seed in COT_SEEDS.items():
        seeds_out[lang] = {
            "seed": seed,
            "tier": LANGUAGES[lang]["tier"],
            "name": LANGUAGES[lang]["name"],
        }
    with open(seeds_path, "w", encoding="utf-8") as f:
        json.dump(seeds_out, f, ensure_ascii=False, indent=2)
    print(f"[✓] Saved CoT seeds for {len(seeds_out)} languages → {seeds_path}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Total prompts: {len(df)}")
    print(f"Sources:       {df['source'].value_counts().to_dict()}")
    print(f"Languages:     {len(LANGUAGES)}")
    print(f"Conditions:    4")
    print(f"Generations needed per model: {len(df) * len(LANGUAGES) * 4:,}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

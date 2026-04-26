"""
03_activation_patch.py — Refusal-vector localization via activation patching.

Patches residual-stream activations from a "clean" (EN-CoT, model refuses)
run into a "corrupted" (target-lang-CoT, model complies) run, measuring
how much each layer restores refusal behavior.

Runs on DeepSeek-R1-Distill-Qwen-14B only (best documented, fits fp16).

Usage:
  python 03_activation_patch.py                           # full run
  python 03_activation_patch.py --prompts 5 --languages en,sw  # quick test
  python 03_activation_patch.py --layers 0,12,24,36,47    # specific layers
"""

import argparse
import gc
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from config import (
    MODELS, COT_SEEDS, LANGUAGES,
    PATCH_MODEL, PATCH_LANGUAGES, PATCH_NUM_PROMPTS,
    PATCH_LAYER_STRIDE, REFUSAL_TOKENS,
    DATA_DIR, RESULTS_DIR,
)


def find_refusing_prompts(
    scored_path: str,
    n: int = 50,
) -> list[int]:
    """
    Select prompt_ids where baseline (en_en) consistently refuses.
    These provide clean refusal activations for patching.
    """
    df = pd.read_parquet(scored_path)
    baseline = df[df["condition"] == "en_en"]

    # Prompts scored 0 (safe/refused) in baseline
    refused = baseline[baseline["judge_score"] == 0]["prompt_id"].unique()
    print(f"[*] Found {len(refused)} prompts refused in baseline")

    # Also check that at least one non-EN CoT condition succeeds (score=1)
    # to ensure attack actually works on these prompts
    attacked = df[
        (df["condition"] == "en_tl") & (df["judge_score"] == 1)
    ]["prompt_id"].unique()

    # Intersection: refused in baseline AND jailbroken in attack
    good_prompts = list(set(refused) & set(attacked))
    print(f"[*] {len(good_prompts)} prompts show refusal→compliance flip")

    selected = sorted(good_prompts)[:n]
    print(f"[*] Selected {len(selected)} prompts for patching")
    return selected


def compute_refusal_score(logits: torch.Tensor, tokenizer) -> float:
    """
    Compute refusal probability from next-token logits.
    Sums probability mass on refusal-indicative tokens.
    """
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    refusal_prob = 0.0
    for token_str in REFUSAL_TOKENS:
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        for tid in token_ids:
            if tid < probs.shape[0]:
                refusal_prob += probs[tid].item()
    return refusal_prob


def run_patching(
    prompt_ids: list[int],
    languages: list[str],
    target_layers: list[int],
):
    """
    Core activation patching loop.
    For each prompt × language: patch clean (EN-CoT) → corrupted (TL-CoT).
    """
    model_cfg = MODELS[PATCH_MODEL]
    hf_id = model_cfg["hf_id"]
    think_open = model_cfg["think_open"]
    num_layers = model_cfg["num_layers"]

    if not target_layers:
        target_layers = list(range(0, num_layers, PATCH_LAYER_STRIDE))

    print(f"[*] Patching {len(target_layers)} layers: {target_layers}")

    # ── Load prompts ───────────────────────────────────────────────────
    prompts_df = pd.read_parquet(DATA_DIR / "prompts.parquet")
    prompts_df = prompts_df[prompts_df["prompt_id"].isin(prompt_ids)]

    # ── Load model via nnsight ─────────────────────────────────────────
    print(f"[*] Loading {hf_id} via nnsight (fp16)...")
    t0 = time.time()

    from nnsight import LanguageModel
    model = LanguageModel(hf_id, device_map="auto", torch_dtype=torch.float16)
    tokenizer = model.tokenizer
    print(f"    Loaded in {time.time() - t0:.1f}s")

    # ── Patching loop ──────────────────────────────────────────────────
    results = []

    for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df),
                       desc="Prompts"):
        prompt_text = row["prompt"]
        pid = row["prompt_id"]

        # Build clean input (EN prompt + EN CoT → expects refusal)
        en_seed = COT_SEEDS["en"]
        clean_text = _build_input(tokenizer, prompt_text, think_open, en_seed)

        # Get clean activations (cached across languages for this prompt)
        clean_acts = {}
        clean_refusal = 0.0

        with torch.no_grad():
            with model.trace(clean_text, scan=False, validate=False):
                for li in target_layers:
                    clean_acts[li] = model.model.layers[li].output[0].save()
                clean_logits = model.lm_head.output.save()

        clean_refusal = compute_refusal_score(clean_logits, tokenizer)

        for lang in languages:
            if lang == "en":
                # No corruption for English—just record baseline refusal
                for li in target_layers:
                    results.append({
                        "prompt_id": pid,
                        "patch_lang": lang,
                        "layer": li,
                        "refusal_recovery": 1.0,  # identity patch
                        "clean_refusal": clean_refusal,
                        "corrupted_refusal": clean_refusal,
                        "patched_refusal": clean_refusal,
                    })
                continue

            # Build corrupted input (EN prompt + target-lang CoT)
            tl_seed = COT_SEEDS[lang]
            corrupted_text = _build_input(
                tokenizer, prompt_text, think_open, tl_seed
            )

            # Get corrupted baseline (no patching)
            with torch.no_grad():
                with model.trace(corrupted_text, scan=False, validate=False):
                    corrupted_logits = model.lm_head.output.save()

            corrupted_refusal = compute_refusal_score(
                corrupted_logits, tokenizer
            )

            # Patch each layer independently
            for li in target_layers:
                with torch.no_grad():
                    with model.trace(
                        corrupted_text, scan=False, validate=False
                    ):
                        # Overwrite this layer's output with clean activations
                        model.model.layers[li].output[0][:] = clean_acts[li]
                        patched_logits = model.lm_head.output.save()

                patched_refusal = compute_refusal_score(
                    patched_logits, tokenizer
                )

                # Recovery: 0 = no help, 1 = fully restored refusal
                denom = clean_refusal - corrupted_refusal
                recovery = (
                    (patched_refusal - corrupted_refusal) / denom
                    if abs(denom) > 1e-8 else 0.0
                )

                results.append({
                    "prompt_id": pid,
                    "patch_lang": lang,
                    "layer": li,
                    "refusal_recovery": float(np.clip(recovery, -1, 2)),
                    "clean_refusal": clean_refusal,
                    "corrupted_refusal": corrupted_refusal,
                    "patched_refusal": patched_refusal,
                })

    # ── Save ───────────────────────────────────────────────────────────
    result_df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "activation_patching.parquet"
    result_df.to_parquet(out_path, index=False)
    print(f"\n[✓] Saved {len(result_df):,} patching results → {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result_df


def _build_input(tokenizer, prompt: str, think_open: str, seed: str) -> str:
    """Build chat-formatted input with CoT prefix injection."""
    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text += f"{think_open}\n{seed}\n"
    return text


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=int, default=PATCH_NUM_PROMPTS,
                        help="Number of prompts to patch")
    parser.add_argument("--languages", type=str, default=None,
                        help="Comma-separated langs (default: en,hi,sw)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: every 4th)")
    args = parser.parse_args()

    languages = (
        args.languages.split(",") if args.languages
        else PATCH_LANGUAGES
    )
    target_layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers else []
    )

    # Find prompts that show refusal→compliance pattern
    scored_path = RESULTS_DIR / f"scored_{PATCH_MODEL}.parquet"
    if not scored_path.exists():
        print(f"[!] Need scored results first: {scored_path}")
        print("    Run: python 02_run_attacks.py --model deepseek-r1-14b")
        return

    prompt_ids = find_refusing_prompts(str(scored_path), n=args.prompts)

    if not prompt_ids:
        print("[!] No suitable prompts found (need baseline refusals that "
              "flip under CoT switching). Check judge scores.")
        return

    run_patching(prompt_ids, languages, target_layers)


if __name__ == "__main__":
    main()

"""
blocking_experiments.py
=======================
Systematic experiment runner for blocking recall diagnostics.
Runs blocking experiments on --sample-n 5000:

  Exp A: Baseline reference (caps as they WERE: max_token_postings=2000, max_trigram=500, max_cands=75)
         without address blocking / NYSIIS
  Exp B: Loosened caps only (max_token=5000, max_trigram=1500, max_cands=150),
         still no address / NYSIIS
  Exp C: Full new config (max_token=5000, max_trigram=1500, max_cands=300,
         +NYSIIS, +address blocking) -- this is the new default in blocking.py

Usage:
    cd code/business_entity_resolution
    python blocking_experiments.py --exp C          # Run just Exp C (~8 min)
    python blocking_experiments.py --exp all        # Run all 3 sequentially
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Force unbuffered output so terminal always updates live
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.blocking import (
    generate_candidate_pairs,
    compute_blocking_diagnostics,
    diagnose_missed_matches,
)
from src.data_loading import load_train_data
from src.normalization import normalize_source_df

parser = argparse.ArgumentParser(description="Blocking Recall Diagnostic Runner")
parser.add_argument(
    "--exp",
    type=str,
    choices=["A", "B", "C", "all"],
    default="C",
    help="Which experiment to run: 'A', 'B', 'C', or 'all' (default: 'C')",
)
parser.add_argument(
    "--sample-n",
    type=int,
    default=5000,
    help="Sample size of S1 entities (default: 5000)",
)
args = parser.parse_args()

SAMPLE_N = args.sample_n
DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"
if not DATASET_DIR.exists():
    for alt in [Path("../../dataset"), Path("../dataset"), Path("dataset")]:
        if alt.resolve().exists():
            DATASET_DIR = alt.resolve()
            break

print(f"Dataset directory: {DATASET_DIR}", flush=True)
print(f"Sample N: {SAMPLE_N:,}", flush=True)
print(f"Experiment Selection: {args.exp}\n", flush=True)

# ── Load & normalize once ──────────────────────────────────────────────────────
print("=" * 70, flush=True)
print("Loading and normalizing data (shared across all experiments)...", flush=True)
print("=" * 70, flush=True)

t_load = time.perf_counter()
train_data = load_train_data(dataset_dir=DATASET_DIR, sample_n=SAMPLE_N)
s1_df = normalize_source_df(train_data["train_source1"])
s2_df = normalize_source_df(train_data["train_source2"])
s3_df = normalize_source_df(train_data["train_source3"])
gt_df = train_data["train_ground_truth"]
target_dfs = {"source2": s2_df, "source3": s3_df}
total_target = len(s2_df) + len(s3_df)
load_elapsed = time.perf_counter() - t_load
print(f"Data ready in {load_elapsed:.1f}s  "
      f"(S1={len(s1_df):,}, S2={len(s2_df):,}, S3={len(s3_df):,}, GT={len(gt_df):,})\n", flush=True)

ALL_EXPERIMENTS = {
    "A": {
        "key": "A",
        "label": "Exp A — Old caps, name-only (baseline reference)",
        "kwargs": dict(
            max_token_postings=2000,
            max_trigram_postings=500,
            max_candidates_per_s1=75,
            use_nysiis=False,
            use_address_blocking=False,
        ),
        "run_diagnosis": False,
    },
    "B": {
        "key": "B",
        "label": "Exp B — Loosened caps 150, name-only (no addr/NYSIIS)",
        "kwargs": dict(
            max_token_postings=5000,
            max_trigram_postings=1500,
            max_candidates_per_s1=150,
            use_nysiis=False,
            use_address_blocking=False,
        ),
        "run_diagnosis": False,
    },
    "C": {
        "key": "C",
        "label": "Exp C — Loosened caps 300 + NYSIIS + Address blocking (FULL NEW CONFIG)",
        "kwargs": dict(
            max_token_postings=5000,
            max_trigram_postings=1500,
            max_candidates_per_s1=300,
            use_nysiis=True,
            use_address_blocking=True,
        ),
        "run_diagnosis": True,
    },
}

if args.exp == "all":
    selected_experiments = [ALL_EXPERIMENTS["A"], ALL_EXPERIMENTS["B"], ALL_EXPERIMENTS["C"]]
else:
    selected_experiments = [ALL_EXPERIMENTS[args.exp]]

results_table = []

for exp in selected_experiments:
    print("=" * 70, flush=True)
    print(f"  {exp['label']}", flush=True)
    print("=" * 70, flush=True)

    t_start = time.perf_counter()
    cand_df = generate_candidate_pairs(
        s1_df=s1_df,
        target_dfs=target_dfs,
        show_progress=True,
        **exp["kwargs"],
    )
    elapsed = time.perf_counter() - t_start

    diag = compute_blocking_diagnostics(cand_df, gt_df, len(s1_df), total_target)

    recall_pct = diag["recall_ceiling"] * 100
    avg_cands = diag["avg_candidates_per_s1"]
    captured = int(diag["gt_matches_captured"])
    total_gt = int(diag["gt_matches_total"])
    total_pairs = int(diag["candidate_pairs_count"])

    print(f"\n  Recall Ceiling      : {recall_pct:.2f}%  ({captured:,} / {total_gt:,} GT matches)", flush=True)
    print(f"  Avg Candidates/S1  : {avg_cands:.1f}", flush=True)
    print(f"  Total Cand Pairs   : {total_pairs:,}", flush=True)
    print(f"  Blocking Runtime   : {elapsed:.1f}s  ({elapsed/60:.2f} min)\n", flush=True)

    results_table.append({
        "Key": exp["key"],
        "Experiment": exp["label"],
        "max_token_postings": exp["kwargs"]["max_token_postings"],
        "max_trigram_postings": exp["kwargs"]["max_trigram_postings"],
        "max_cands_per_s1": exp["kwargs"]["max_candidates_per_s1"],
        "NYSIIS": exp["kwargs"].get("use_nysiis", False),
        "AddrBlocking": exp["kwargs"].get("use_address_blocking", False),
        "RecallCeiling%": f"{recall_pct:.2f}",
        "AvgCands/S1": f"{avg_cands:.1f}",
        "TotalPairs": f"{total_pairs:,}",
        "Runtime(s)": f"{elapsed:.1f}",
    })

    if exp.get("run_diagnosis", False):
        print("-" * 70, flush=True)
        print("  MISSED-MATCH DIAGNOSIS (sample of missed true-match pairs)", flush=True)
        print("-" * 70, flush=True)
        missed_df = diagnose_missed_matches(
            candidate_pairs_df=cand_df,
            ground_truth_df=gt_df,
            s1_df=s1_df,
            target_dfs=target_dfs,
            n_samples=15,
        )
        if missed_df.empty:
            print("  No missed matches to diagnose (perfect recall ceiling)!", flush=True)
        else:
            reason_counts = missed_df["likely_failure_reason"].value_counts()
            print(f"\n  Failure reason breakdown (n={len(missed_df)}):", flush=True)
            for reason, count in reason_counts.items():
                print(f"    {count:3d}  {reason}", flush=True)

            print(f"\n  Sample missed pairs (up to 15):", flush=True)
            for i, row in missed_df.iterrows():
                print(f"\n  --- Missed pair {i+1} ---", flush=True)
                print(f"    S1 entity   : {row['s1_id']}", flush=True)
                print(f"    Target      : {row['target_id']}  [{row['target_source']}]", flush=True)
                print(f"    S1 name     : {row['s1_name']}", flush=True)
                print(f"    S1 address  : {row['s1_address']}", flush=True)
                print(f"    Target name : {row['target_name']}", flush=True)
                print(f"    Target addr : {row['target_address']}", flush=True)
                print(f"    Shared name tokens : {row['shared_name_tokens'] or '(none)'}", flush=True)
                print(f"    Shared addr tokens : {row['shared_address_tokens'] or '(none)'}", flush=True)
                print(f"    Failure reason     : {row['likely_failure_reason']}", flush=True)

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("EXPERIMENT SUMMARY TABLE", flush=True)
print("=" * 70, flush=True)
print(f"{'Exp':<5} {'token_post':>10} {'tgram_post':>10} {'max_cands':>9} "
      f"{'NYSIIS':>6} {'Addr':>4}  {'Recall%':>8} {'Avg/S1':>7} {'Pairs':>12} {'Runtime':>9}", flush=True)
print("-" * 90, flush=True)
for row in results_table:
    print(
        f"  {row['Key']}  "
        f"{row['max_token_postings']:>10,} "
        f"{row['max_trigram_postings']:>10,} "
        f"{row['max_cands_per_s1']:>9,} "
        f"{'Y' if row['NYSIIS'] else 'N':>6} "
        f"{'Y' if row['AddrBlocking'] else 'N':>4}  "
        f"{row['RecallCeiling%']:>8} "
        f"{row['AvgCands/S1']:>7} "
        f"{row['TotalPairs']:>12} "
        f"{row['Runtime(s)']:>9}",
        flush=True
    )
print("=" * 70, flush=True)

# Save results JSON
results_file = Path("blocking_experiments_results.json")
with open(results_file, "w") as f:
    json.dump(results_table, f, indent=2)
print(f"\nSaved summary results to: {results_file.resolve()}", flush=True)

import time
from pathlib import Path
from src.blocking import compute_blocking_diagnostics, generate_candidate_pairs
from src.data_loading import load_train_data
from src.normalization import normalize_source_df


def main():
    print("=" * 70)
    print("  EVALUATING UPDATED BLOCKING (WITH ADDRESS TOKEN N-GRAM KEY)")
    print("=" * 70)
    start_t = time.perf_counter()

    dataset_path = Path("dataset")
    if not dataset_path.exists():
        for alt in [Path("../../dataset"), Path("../dataset")]:
            if alt.exists():
                dataset_path = alt
                break

    print(f"Loading data with sample_n=5000...")
    train_data = load_train_data(dataset_dir=dataset_path, sample_n=5000, random_state=42)
    s1_df = normalize_source_df(train_data["train_source1"])
    s2_df = normalize_source_df(train_data["train_source2"])
    s3_df = normalize_source_df(train_data["train_source3"])
    gt_df = train_data["train_ground_truth"]
    target_dfs = {"source2": s2_df, "source3": s3_df}

    print("\n--- Running Candidate Blocking ---")
    t0 = time.perf_counter()
    cand_df = generate_candidate_pairs(
        s1_df=s1_df,
        target_dfs=target_dfs,
        max_token_postings=5000,
        max_trigram_postings=1500,
        max_candidates_per_s1=300,
        use_nysiis=True,
        use_address_blocking=True,
        show_progress=True,
    )
    blocking_t = time.perf_counter() - t0

    total_target = len(s2_df) + len(s3_df)
    diag = compute_blocking_diagnostics(cand_df, gt_df, len(s1_df), total_target)

    recall_pct = diag["recall_ceiling"] * 100
    avg_cands = diag["avg_candidates_per_s1"]
    total_pairs = int(diag["candidate_pairs_count"])
    captured = int(diag.get("gt_matches_captured", 0))
    total_gt = int(diag.get("gt_matches_total", 0))

    print("\n" + "=" * 70)
    print("BLOCKING EVALUATION REPORT: STEP 2")
    print("=" * 70)
    print(f"Baseline Recall Ceiling : 51.01% (8,867 / 17,383 GT matches)")
    print(f"New Recall Ceiling      : {recall_pct:.2f}% ({captured:,} / {total_gt:,} GT matches)")
    print(f"Recall Delta            : {recall_pct - 51.01:+.2f}%")
    print("-" * 70)
    print(f"Baseline Total Pairs    : 1,497,137 (Avg {299.43:.2f} per S1)")
    print(f"New Total Cand Pairs    : {total_pairs:,} (Avg {avg_cands:.2f} per S1)")
    print(f"Candidate Pool Delta    : {total_pairs - 1497137:+,} pairs")
    print(f"Blocking Search Time    : {blocking_t:.2f}s ({blocking_t/60:.2f} min)")
    print(f"Total Execution Time    : {time.perf_counter() - start_t:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import time
from typing import Dict, List, Optional, Set, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.blocking import generate_candidate_pairs
from src.data_loading import load_train_data
from src.evaluate import parse_ground_truth
from src.features import extract_candidate_features
from src.normalization import normalize_source_df


def run_error_analysis(
    dataset_dir: str = "dataset",
    model_path: str = "models/lgbm_matcher.pkl",
    output_report_path: str = "reports/error_analysis.csv",
    sample_n: int = 5000,
    val_size: float = 0.2,
    random_state: int = 42,
    threshold: Optional[float] = None,
    cache_path: Optional[str] = "models/feature_df_5000.pkl",
):
    """Run error analysis on the validation split from the baseline model."""
    start_time = time.perf_counter()
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        for alt in [Path("../../dataset"), Path("../dataset")]:
            if alt.exists():
                dataset_path = alt
                break

    print("=" * 70)
    print("      ERROR ANALYSIS FOR BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 70)
    print(f"Dataset path  : {dataset_path.resolve()}")
    print(f"Model path    : {Path(model_path).resolve()}")
    print(f"Sample N      : {sample_n}")

    # 1. Load model and threshold
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file.resolve()}")
    
    saved_bundle = joblib.load(model_file)
    model = saved_bundle["model"]
    decision_threshold = threshold if threshold is not None else float(saved_bundle.get("threshold", 0.45))
    print(f"Using Decision Threshold: {decision_threshold:.4f}")

    # 2. Check if cached feature_df exists to save time
    feature_df = None
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        print(f"Loading cached feature DataFrame from {cache_file}...")
        try:
            feature_df = joblib.load(cache_file)
            print(f"Loaded {len(feature_df):,} cached feature rows.")
        except Exception as e:
            print(f"Warning: Failed to load cache ({e}), recomputing...")
            feature_df = None

    train_data = load_train_data(dataset_dir=dataset_path, sample_n=sample_n, random_state=random_state)
    s1_df = normalize_source_df(train_data["train_source1"])
    s2_df = normalize_source_df(train_data["train_source2"])
    s3_df = normalize_source_df(train_data["train_source3"])
    gt_df = train_data["train_ground_truth"]
    target_dfs = {"source2": s2_df, "source3": s3_df}

    if feature_df is None:
        print("\n--- Generating Candidate Pairs & Features ---")
        cand_pairs = generate_candidate_pairs(s1_df, target_dfs, show_progress=True)
        feature_df, _ = extract_candidate_features(cand_pairs, s1_df, target_dfs, show_progress=True)
        
        # Label ground truth
        gt_map = parse_ground_truth(gt_df)
        labels = [
            1 if cand_id in gt_map.get(s1_id, set()) else 0
            for s1_id, cand_id in zip(
                feature_df["source1_entity_id"].astype(str),
                feature_df["candidate_entity_id"].astype(str),
            )
        ]
        feature_df["label"] = labels

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(feature_df, cache_file)
            print(f"Cached feature DataFrame to {cache_file}")

    # 3. Replicate exact validation split
    feature_cols = [
        c for c in feature_df.columns
        if c not in ["source1_entity_id", "candidate_entity_id", "label"]
    ]
    X = feature_df[feature_cols]
    y = feature_df["label"]
    groups = feature_df["source1_entity_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    val_df = feature_df.iloc[val_idx].copy().reset_index(drop=True)
    val_X = val_df[feature_cols]
    val_y = val_df["label"].values

    print(f"\nValidation Split: {len(val_df):,} pairs across {val_df['source1_entity_id'].nunique():,} unique S1 entities.")

    # 4. Predict probabilities
    val_probs = model.predict_proba(val_X)[:, 1]
    val_df["pred_prob"] = val_probs
    val_df["pred_label"] = (val_probs >= decision_threshold).astype(int)

    # 5. Classify error types: TP, FP, FN, TN
    val_df["error_type"] = "TN"
    val_df.loc[(val_y == 1) & (val_df["pred_label"] == 1), "error_type"] = "TP"
    val_df.loc[(val_y == 1) & (val_df["pred_label"] == 0), "error_type"] = "FN"
    val_df.loc[(val_y == 0) & (val_df["pred_label"] == 1), "error_type"] = "FP"
    val_df.loc[(val_y == 0) & (val_df["pred_label"] == 0), "error_type"] = "TN"

    counts = val_df["error_type"].value_counts().to_dict()
    tp_count = counts.get("TP", 0)
    fp_count = counts.get("FP", 0)
    fn_count = counts.get("FN", 0)
    tn_count = counts.get("TN", 0)

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX ON VALIDATION SET")
    print("=" * 60)
    print(f"True Positives  (TP) : {tp_count:,}")
    print(f"False Positives (FP) : {fp_count:,}")
    print(f"False Negatives (FN) : {fn_count:,}")
    print(f"True Negatives  (TN) : {tn_count:,}")
    print(f"Precision            : {tp_count / (tp_count + fp_count):.4f}" if (tp_count + fp_count) > 0 else "0.0")
    print(f"Recall               : {tp_count / (tp_count + fn_count):.4f}" if (tp_count + fn_count) > 0 else "0.0")
    print("=" * 60)

    # 6. Build lookup maps for raw entity attributes
    entity_text_map = {}
    for df in [s1_df, s2_df, s3_df]:
        for _, row in df.iterrows():
            eid = str(row["entity_id"])
            if eid not in entity_text_map:
                entity_text_map[eid] = {
                    "name": str(row.get("clean_name", row.get("name_norm", ""))),
                    "address": str(row.get("clean_address", row.get("address_norm", ""))),
                    "pincode": str(row.get("pin_code", "")),
                    "city": str(row.get("city", "")),
                }

    val_df["s1_name"] = val_df["source1_entity_id"].astype(str).map(lambda x: entity_text_map.get(x, {}).get("name", ""))
    val_df["cand_name"] = val_df["candidate_entity_id"].astype(str).map(lambda x: entity_text_map.get(x, {}).get("name", ""))
    val_df["s1_address"] = val_df["source1_entity_id"].astype(str).map(lambda x: entity_text_map.get(x, {}).get("address", ""))
    val_df["cand_address"] = val_df["candidate_entity_id"].astype(str).map(lambda x: entity_text_map.get(x, {}).get("address", ""))
    val_df["s1_pin"] = val_df["source1_entity_id"].astype(str).map(lambda x: entity_text_map.get(x, {}).get("pincode", ""))
    val_df["cand_pin"] = val_df["candidate_entity_id"].astype(str).map(lambda x: entity_text_map.get(x, {}).get("pincode", ""))

    # 7. Error pattern flagging for False Positives:
    # High address similarity + low name similarity (e.g. shared building/complex, different business)
    # High name similarity + low address similarity (e.g. same franchise/chain, different branches)
    val_df["error_flag"] = ""
    fp_mask = val_df["error_type"] == "FP"
    high_addr_low_name = fp_mask & (val_df["addr_levenshtein"] >= 0.65) & (val_df["name_levenshtein"] < 0.40)
    high_name_low_addr = fp_mask & (val_df["name_levenshtein"] >= 0.65) & (val_df["addr_levenshtein"] < 0.35)
    val_df.loc[high_addr_low_name, "error_flag"] = "HIGH_ADDR_LOW_NAME (Shared Address/Building)"
    val_df.loc[high_name_low_addr, "error_flag"] = "HIGH_NAME_LOW_ADDR (Franchise/Different Location)"

    # 8. Separate FN and FP for report
    fn_df = val_df[val_df["error_type"] == "FN"].sort_values("pred_prob", ascending=True)
    fp_df = val_df[val_df["error_type"] == "FP"].sort_values("pred_prob", ascending=False)
    tp_df = val_df[val_df["error_type"] == "TP"]
    tn_df = val_df[val_df["error_type"] == "TN"]

    # 9. Print Feature Mean Summary for FN vs TP, and FP vs TN
    print("\n" + "=" * 70)
    print("FEATURE COMPARISON: FALSE NEGATIVES (FN) vs TRUE POSITIVES (TP)")
    print("(Shows which similarity features are weakest for missed true matches)")
    print("=" * 70)
    fn_means = fn_df[feature_cols].mean()
    tp_means = tp_df[feature_cols].mean()
    diff_fn_tp = (tp_means - fn_means).abs().sort_values(ascending=False)

    print(f"{'Feature':<28} | {'FN Mean':<10} | {'TP Mean':<10} | {'Delta (TP - FN)':<15}")
    print("-" * 70)
    for feat in diff_fn_tp.index[:10]:
        fn_m = fn_means[feat]
        tp_m = tp_means[feat]
        delta = tp_m - fn_m
        print(f"{feat:<28} | {fn_m:<10.4f} | {tp_m:<10.4f} | {delta:<+15.4f}")

    print("\n" + "=" * 70)
    print("FEATURE COMPARISON: FALSE POSITIVES (FP) vs TRUE NEGATIVES (TN)")
    print("(Shows which features tricked the model into false positives)")
    print("=" * 70)
    fp_means = fp_df[feature_cols].mean()
    tn_means = tn_df[feature_cols].mean()
    diff_fp_tn = (fp_means - tn_means).abs().sort_values(ascending=False)

    print(f"{'Feature':<28} | {'FP Mean':<10} | {'TN Mean':<10} | {'Delta (FP - TN)':<15}")
    print("-" * 70)
    for feat in diff_fp_tn.index[:10]:
        fp_m = fp_means[feat]
        tn_m = tn_means[feat]
        delta = fp_m - tn_m
        print(f"{feat:<28} | {fp_m:<10.4f} | {tn_m:<10.4f} | {delta:<+15.4f}")

    # Breakdown of FP error flags
    print("\n" + "=" * 70)
    print("FALSE POSITIVE PATTERN BREAKDOWN")
    print("=" * 70)
    flag_counts = fp_df["error_flag"].value_counts()
    for flag, cnt in flag_counts.items():
        desc = flag if flag else "OTHER_AMBIGUOUS_MATCH"
        print(f"  {desc:<50}: {cnt:,} ({cnt/len(fp_df)*100:.1f}%)")

    # 10. Save Error Analysis Report
    # Include all FN and FP in the report
    report_df = pd.concat([fn_df, fp_df], ignore_index=True)
    report_path = Path(output_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_cols = [
        "error_type", "error_flag", "pred_prob", "source1_entity_id", "candidate_entity_id",
        "s1_name", "cand_name", "s1_address", "cand_address", "s1_pin", "cand_pin",
    ] + feature_cols

    report_df[ordered_cols].to_csv(report_path, index=False)
    print(f"\nSaved Error Analysis CSV Report to: {report_path.resolve()} ({len(report_df):,} rows)")

    # Also save Markdown summary for instant reading
    md_path = report_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Error Analysis Report\n\n")
        f.write(f"- **Validation Pairs Evaluated**: {len(val_df):,}\n")
        f.write(f"- **True Positives (TP)**: {tp_count:,}\n")
        f.write(f"- **False Negatives (FN)**: {fn_count:,}\n")
        f.write(f"- **False Positives (FP)**: {fp_count:,}\n")
        f.write(f"- **Decision Threshold**: {decision_threshold:.4f}\n\n")
        f.write("## Top 5 Weakest Features in False Negatives (Missed Matches)\n\n")
        f.write("| Feature | FN Mean | TP Mean | Drop in FN |\n| :--- | :--- | :--- | :--- |\n")
        for feat in diff_fn_tp.index[:5]:
            f.write(f"| `{feat}` | {fn_means[feat]:.4f} | {tp_means[feat]:.4f} | {tp_means[feat] - fn_means[feat]:+.4f} |\n")
        
        f.write("\n## Top 5 Features Distinguishing False Positives from True Negatives\n\n")
        f.write("| Feature | FP Mean | TN Mean | Elevation in FP |\n| :--- | :--- | :--- | :--- |\n")
        for feat in diff_fp_tn.index[:5]:
            f.write(f"| `{feat}` | {fp_means[feat]:.4f} | {tn_means[feat]:.4f} | {fp_means[feat] - tn_means[feat]:+.4f} |\n")
        
        f.write("\n## Sample False Negatives (Lowest Confidence Missed True Matches)\n\n")
        f.write("| S1 Name | Cand Name | S1 Address | Cand Address | Prob | Name Jaccard | Addr Levenshtein |\n| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for _, row in fn_df.head(10).iterrows():
            f.write(f"| {row['s1_name'][:30]} | {row['cand_name'][:30]} | {row['s1_address'][:30]} | {row['cand_address'][:30]} | {row['pred_prob']:.3f} | {row['name_token_jaccard']:.2f} | {row['addr_levenshtein']:.2f} |\n")

        f.write("\n## Sample False Positives (Highest Confidence False Matches)\n\n")
        f.write("| Flag | S1 Name | Cand Name | S1 Address | Cand Address | Prob |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for _, row in fp_df.head(10).iterrows():
            flag = row['error_flag'] or "General FP"
            f.write(f"| {flag} | {row['s1_name'][:30]} | {row['cand_name'][:30]} | {row['s1_address'][:30]} | {row['cand_address'][:30]} | {row['pred_prob']:.3f} |\n")

    print(f"Saved Markdown Summary Report to: {md_path.resolve()}")
    elapsed = time.perf_counter() - start_time
    print(f"\nError Analysis completed in {elapsed:.2f}s ({elapsed/60:.2f} min).")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run error analysis on validation set.")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--model_path", type=str, default="models/lgbm_matcher.pkl")
    parser.add_argument("--output_report", type=str, default="reports/error_analysis.csv")
    parser.add_argument("--sample_n", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    run_error_analysis(
        dataset_dir=args.dataset_dir,
        model_path=args.model_path,
        output_report_path=args.output_report,
        sample_n=args.sample_n,
        threshold=args.threshold,
    )

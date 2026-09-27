from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.blocking import generate_candidate_pairs
from src.data_loading import load_test_data
from src.features import extract_candidate_features
from src.normalization import normalize_source_df


def predict_test_set(
    dataset_dir: Union[str, Path] = "dataset",
    model_path: Union[str, Path] = "models/lgbm_matcher.pkl",
    output_dir: Union[str, Path] = "output",
    threshold: Optional[float] = None,
    sample_n: Optional[int] = None,
    sample_frac: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run normalization, blocking, feature extraction, model scoring, and prediction formatting on test data.
    
    Args:
        dataset_dir: Root dataset directory containing dataset/test/ files.
        model_path: Path to saved model file.
        output_dir: Output directory path to write candidate_pairs.tsv and matching_results.tsv.
        threshold: Optional override for decision threshold.
        sample_n: Optional sample count of test Source 1 entities for fast dev runs.
        sample_frac: Optional sample fraction of test Source 1 entities.
        
    Returns:
        Tuple of (candidate_pairs_output_df, matching_results_output_df).
    """
    start_t = time.perf_counter()
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        for alt in [Path("../../dataset"), Path("../dataset")]:
            if alt.exists():
                dataset_path = alt
                break

    print(f"Loading test dataset from: {dataset_path.resolve()}")
    test_data = load_test_data(dataset_dir=dataset_path, sample_n=sample_n, sample_frac=sample_frac)

    if not test_data or "test_source1" not in test_data:
        raise FileNotFoundError(f"Test dataset files not found under {dataset_path}/test/")

    # 1. Normalize test source DataFrames
    print("\n--- Normalizing Test Source DataFrames ---")
    t0 = time.perf_counter()
    s1_df = normalize_source_df(test_data["test_source1"])
    s2_df = normalize_source_df(test_data["test_source2"])
    s3_df = normalize_source_df(test_data["test_source3"])
    norm_elapsed = time.perf_counter() - t0

    all_test_s1_ids = list(s1_df["entity_id"].astype(str))
    target_dfs = {"source2": s2_df, "source3": s3_df}

    # 2. Candidate blocking
    print("\n--- Running Candidate Blocking on Test Set ---")
    t0 = time.perf_counter()
    candidate_pairs = generate_candidate_pairs(s1_df, target_dfs)
    blocking_elapsed = time.perf_counter() - t0

    # 3. Extract features using exact same pipeline as training
    print("\n--- Extracting Pairwise Features on Test Set ---")
    t0 = time.perf_counter()
    feature_df, _ = extract_candidate_features(candidate_pairs, s1_df, target_dfs)
    features_elapsed = time.perf_counter() - t0

    # 4. Load trained model & threshold
    model_file = Path(model_path)
    if not model_file.exists():
        model_file = Path("models/lgbm_matcher.pkl")

    if not model_file.exists():
        raise FileNotFoundError(f"Trained model file not found: {model_path}")

    print(f"Loading trained model from: {model_file.resolve()}")
    saved_obj = joblib.load(model_file)
    model = saved_obj["model"]
    saved_threshold = saved_obj.get("threshold", 0.5)

    final_threshold = threshold if threshold is not None else saved_threshold
    print(f"Using decision threshold: {final_threshold:.2f}")

    # 5. Score candidate pairs
    t0 = time.perf_counter()
    if not feature_df.empty:
        feature_cols = [
            c
            for c in feature_df.columns
            if c not in ["source1_entity_id", "candidate_entity_id", "label"]
        ]
        probs = model.predict_proba(feature_df[feature_cols])[:, 1]
        feature_df = feature_df.copy()
        feature_df["prob"] = probs
    else:
        feature_df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "prob"])

    scoring_elapsed = time.perf_counter() - t0

    # 6. Format candidate_pairs.tsv and matching_results.tsv
    cand_map = {s1_id: set() for s1_id in all_test_s1_ids}
    match_map = {s1_id: set() for s1_id in all_test_s1_ids}

    s1_ids = feature_df["source1_entity_id"].astype(str).tolist() if not feature_df.empty else []
    cand_ids = feature_df["candidate_entity_id"].astype(str).tolist() if not feature_df.empty else []
    probs_list = feature_df["prob"].tolist() if not feature_df.empty else []

    for s1_id, c_id, prob in zip(s1_ids, cand_ids, probs_list):
        if c_id != s1_id:
            cand_map[s1_id].add(c_id)
            if prob >= final_threshold:
                match_map[s1_id].add(c_id)

    cand_rows = []
    match_rows = []

    for s1_id in all_test_s1_ids:
        cand_ids_str = ",".join(sorted(cand_map[s1_id]))
        match_ids_str = ",".join(sorted(match_map[s1_id]))

        cand_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cand_ids_str})
        match_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": match_ids_str})

    cand_out_df = pd.DataFrame(cand_rows)
    match_out_df = pd.DataFrame(match_rows)

    # 7. Write TSV output files
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cand_file = out_path / "candidate_pairs.tsv"
    match_file = out_path / "matching_results.tsv"

    cand_out_df.to_csv(cand_file, sep="\t", index=False)
    match_out_df.to_csv(match_file, sep="\t", index=False)

    overall_elapsed = time.perf_counter() - start_t

    print("\n" + "=" * 60)
    print("INFERENCE OUTPUT REPORT")
    print("=" * 60)
    print(f"Total Test Source 1 Entities : {len(all_test_s1_ids):,}")
    print(f"Total Candidate Pairs Scored : {len(feature_df):,}")
    print(f"Candidate Pairs File Written : {cand_file.resolve()}")
    print(f"Matching Results File Written: {match_file.resolve()}")
    print("-" * 60)
    print("STAGE RUNTIMES:")
    print(f"  Normalization Runtime      : {norm_elapsed:.2f}s")
    print(f"  Blocking Runtime           : {blocking_elapsed:.2f}s")
    print(f"  Feature Extraction         : {features_elapsed:.2f}s")
    print(f"  Model Scoring & Formatting : {scoring_elapsed:.2f}s")
    print(f"  Total Inference Runtime    : {overall_elapsed:.2f}s ({overall_elapsed/60:.2f} min)")
    print("=" * 60 + "\n")

    return cand_out_df, match_out_df


def run_inference(model: Any, candidate_features: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Legacy helper function for candidate pair probability estimation."""
    feature_cols = [c for c in candidate_features.columns if c not in ["source1_entity_id", "candidate_entity_id", "entity_id_1", "entity_id_2"]]
    X = candidate_features[feature_cols]
    probs = model.predict_proba(X)[:, 1]
    results = candidate_features.copy()
    results["match_probability"] = probs
    results["is_match"] = results["match_probability"] >= threshold
    return results[results["is_match"]]

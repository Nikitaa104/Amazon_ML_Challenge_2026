from pathlib import Path
import time
from typing import Dict, List, Optional, Set, Tuple, Union
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.blocking import compute_blocking_diagnostics, generate_candidate_pairs
from src.data_loading import load_train_data
from src.evaluate import parse_ground_truth
from src.features import extract_candidate_features
from src.normalization import normalize_source_df


def compute_macro_f05(
    predictions_map: Dict[str, Set[str]],
    ground_truth_map: Dict[str, Set[str]],
    all_s1_ids: Set[str],
) -> Tuple[float, float, float]:
    """Compute per-S1 macro-averaged F_0.5 metric, precision, and recall."""
    f05_list = []
    prec_list = []
    rec_list = []

    for s1_id in all_s1_ids:
        gt_set = ground_truth_map.get(s1_id, set())
        pred_set = predictions_map.get(s1_id, set())

        len_gt = len(gt_set)
        len_pred = len(pred_set)

        if len_gt == 0:
            if len_pred == 0:
                f05_i = 1.0
                prec_i = 1.0
                rec_i = 1.0
            else:
                f05_i = 0.0
                prec_i = 0.0
                rec_i = 0.0
        else:
            tp = len(gt_set.intersection(pred_set))
            prec_i = tp / len_pred if len_pred > 0 else 0.0
            rec_i = tp / len_gt
            denom = 0.25 * prec_i + rec_i
            if denom > 0:
                f05_i = (1.25 * prec_i * rec_i) / denom
            else:
                f05_i = 0.0

        f05_list.append(f05_i)
        prec_list.append(prec_i)
        rec_list.append(rec_i)

    macro_f05 = float(np.mean(f05_list)) if f05_list else 0.0
    macro_prec = float(np.mean(prec_list)) if prec_list else 0.0
    macro_rec = float(np.mean(rec_list)) if rec_list else 0.0

    return macro_f05, macro_prec, macro_rec


def build_training_set(
    train_data: Optional[Dict[str, pd.DataFrame]] = None,
    dataset_dir: Union[str, Path] = "dataset",
    sample_n: Optional[int] = None,
    sample_frac: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, tuple]:
    """Run normalization, blocking, feature extraction, and ground truth labeling on training data.
    
    Args:
        train_data: Optional pre-loaded train data dict.
        dataset_dir: Directory path containing train/ files if train_data is None.
        sample_n: Optional number of Source 1 entities to sample.
        sample_frac: Optional fraction of Source 1 entities to sample.
        
    Returns:
        Tuple of (feature_df, candidate_pairs_df, diagnostics, vectorizers).
    """
    overall_start = time.perf_counter()

    if train_data is None:
        path = Path(dataset_dir)
        if not path.exists():
            for alt in [Path("../../dataset"), Path("../dataset")]:
                if alt.exists():
                    path = alt
                    break
        train_data = load_train_data(dataset_dir=path, sample_n=sample_n, sample_frac=sample_frac)

    print("\n--- Step 1: Normalizing Source DataFrames ---")
    t0 = time.perf_counter()
    s1_df = normalize_source_df(train_data["train_source1"])
    s2_df = normalize_source_df(train_data["train_source2"])
    s3_df = normalize_source_df(train_data["train_source3"])
    gt_df = train_data["train_ground_truth"]
    norm_elapsed = time.perf_counter() - t0

    target_dfs = {"source2": s2_df, "source3": s3_df}

    # 1. Candidate blocking
    print("\n--- Step 2: Running Candidate Blocking ---")
    t0 = time.perf_counter()
    candidate_pairs = generate_candidate_pairs(s1_df, target_dfs)
    blocking_elapsed = time.perf_counter() - t0

    # 2. Compute features
    print("\n--- Step 3: Extracting Pairwise Features ---")
    t0 = time.perf_counter()
    feature_df, vectorizers = extract_candidate_features(candidate_pairs, s1_df, target_dfs)
    features_elapsed = time.perf_counter() - t0

    # 3. Ground truth labeling
    print("\n--- Step 4: Ground Truth Labeling ---")
    t0 = time.perf_counter()
    gt_map = parse_ground_truth(gt_df)
    labels = []

    s1_ids = feature_df["source1_entity_id"].astype(str).tolist()
    cand_ids = feature_df["candidate_entity_id"].astype(str).tolist()

    for s1_id, cand_id in zip(s1_ids, cand_ids):
        true_matches = gt_map.get(s1_id, set())
        label = 1 if cand_id in true_matches else 0
        labels.append(label)

    feature_df["label"] = labels
    label_elapsed = time.perf_counter() - t0

    total_pairs = len(feature_df)
    positives = sum(labels)
    negatives = total_pairs - positives
    pos_pct = (positives / total_pairs * 100.0) if total_pairs > 0 else 0.0

    total_target_count = len(s2_df) + len(s3_df)
    diagnostics = compute_blocking_diagnostics(
        candidate_pairs, gt_df, len(s1_df), total_target_count
    )

    overall_elapsed = time.perf_counter() - overall_start

    print("\n" + "=" * 60)
    print("TRAINING SET BUILD REPORT")
    print("=" * 60)
    print(f"Source 1 Entities Evaluated: {len(s1_df):,}")
    print(f"Total Candidate Pairs      : {total_pairs:,}")
    print(f"Positive Matches (1)       : {positives:,} ({pos_pct:.2f}%)")
    print(f"Negative Non-Matches (0)   : {negatives:,}")
    gt_total = diagnostics.get("gt_matches_total", diagnostics.get("total_ground_truth_matches", 0))
    gt_captured = diagnostics.get("gt_matches_captured", diagnostics.get("captured_ground_truth_matches", 0))
    print(f"Ground Truth Match Count   : {gt_total:,}")
    print(f"Captured Ground Truth      : {gt_captured:,}")
    print(f"Blocking Recall Ceiling    : {diagnostics['recall_ceiling'] * 100.0:.2f}%")
    print(f"Search Space Reduction     : {diagnostics['reduction_ratio'] * 100.0:.4f}%")
    print(f"Avg Candidates per S1      : {diagnostics['avg_candidates_per_s1']:.2f}")
    print("-" * 60)
    print("STAGE RUNTIMES:")
    print(f"  Normalization Runtime    : {norm_elapsed:.2f}s")
    print(f"  Blocking Runtime         : {blocking_elapsed:.2f}s")
    print(f"  Feature Extraction       : {features_elapsed:.2f}s")
    print(f"  Labeling Runtime         : {label_elapsed:.2f}s")
    print(f"  Total Build Runtime      : {overall_elapsed:.2f}s ({overall_elapsed/60:.2f} min)")
    print("=" * 60 + "\n")

    return feature_df, candidate_pairs, diagnostics, vectorizers


def train_model(
    feature_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    model_save_path: Union[str, Path] = "models/lgbm_matcher.txt",
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[lgb.LGBMClassifier, float, dict]:
    """Train LightGBM binary classifier with GroupKFold / GroupShuffleSplit validation and threshold tuning."""
    start_t = time.perf_counter()
    feature_cols = [
        c
        for c in feature_df.columns
        if c not in ["source1_entity_id", "candidate_entity_id", "label"]
    ]

    X = feature_df[feature_cols]
    y = feature_df["label"]
    groups = feature_df["source1_entity_id"]

    # Group-aware train/val split
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    val_feature_df = feature_df.iloc[val_idx]

    val_s1_ids = set(val_feature_df["source1_entity_id"].astype(str).unique())
    gt_map = parse_ground_truth(gt_df)

    print(f"Group Split: {len(train_idx):,} train pairs, {len(val_idx):,} validation pairs.")
    print(f"Validation set covers {len(val_s1_ids):,} unique Source 1 entities.")

    # Compute scale_pos_weight from training set to handle ~160:1 class imbalance
    n_neg = int((y.iloc[train_idx] == 0).sum())
    n_pos = int((y.iloc[train_idx] == 1).sum())
    scale_pos_weight = max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0
    print(f"Class imbalance: {n_neg:,} neg / {n_pos:,} pos  =>  scale_pos_weight={scale_pos_weight:.1f}")

    lgb_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "verbose": -1,
        "n_jobs": -1,
    }
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_feature_df = val_feature_df.copy()
    val_feature_df["prob"] = val_probs

    best_threshold = 0.5
    best_macro_f05 = -1.0
    best_metrics = {}

    thresholds = np.linspace(0.05, 0.95, 19)

    for thresh in thresholds:
        pred_map: Dict[str, Set[str]] = {}
        matched_pairs = val_feature_df[val_feature_df["prob"] >= thresh]

        for s1_id, group in matched_pairs.groupby("source1_entity_id"):
            pred_map[str(s1_id)] = set(group["candidate_entity_id"].astype(str))

        macro_f05, macro_prec, macro_rec = compute_macro_f05(
            pred_map, gt_map, val_s1_ids
        )

        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_threshold = float(thresh)
            best_metrics = {
                "macro_f05": macro_f05,
                "macro_precision": macro_prec,
                "macro_recall": macro_rec,
                "optimal_threshold": best_threshold,
            }

    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "booster_"):
        model.booster_.save_model(str(save_path))

    joblib_path = save_path.with_suffix(".pkl")
    joblib.dump({"model": model, "threshold": best_threshold}, joblib_path)

    elapsed = time.perf_counter() - start_t

    print("\n" + "=" * 60)
    print("MODEL TRAINING & THRESHOLD TUNING REPORT")
    print("=" * 60)
    print(f"Optimal Decision Threshold : {best_threshold:.2f}")
    print(f"Validation Macro F_0.5     : {best_metrics['macro_f05']:.4f}")
    print(f"Validation Macro Precision : {best_metrics['macro_precision']:.4f}")
    print(f"Validation Macro Recall    : {best_metrics['macro_recall']:.4f}")
    print(f"Training & Tuning Runtime  : {elapsed:.2f}s")
    print(f"Saved Model File           : {save_path.resolve()}")
    print("=" * 60 + "\n")

    return model, best_threshold, best_metrics


def predict_pair_match_probabilities(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Predict pair match probabilities using a trained classifier model."""
    return model.predict_proba(X)[:, 1]

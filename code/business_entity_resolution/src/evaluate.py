import pandas as pd
from typing import Dict, Set, Tuple


def parse_ground_truth(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Parse ground truth DataFrame into dictionary mapping source1_entity_id to set of matched_entity_ids.
    
    Args:
        gt_df: Ground truth DataFrame with source1_entity_id and matched_entity_ids columns.
        
    Returns:
        Dictionary mapping entity ID to set of matched IDs.
    """
    gt_map = {}
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched_str = str(row.get("matched_entity_ids", "") or "")
        if pd.isna(matched_str) or matched_str.lower() in ("nan", "none", ""):
            gt_map[s1_id] = set()
        else:
            gt_map[s1_id] = set(m.strip() for m in matched_str.split(",") if m.strip())
    return gt_map


def compute_metrics(
    predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame
) -> Dict[str, float]:
    """Compute Precision, Recall, and F1-score for entity resolution match pairs.
    
    Args:
        predictions_df: Predictions DataFrame with source1_entity_id, matched_entity_ids.
        ground_truth_df: Ground truth DataFrame with source1_entity_id, matched_entity_ids.
        
    Returns:
        Dictionary containing precision, recall, f1_score.
    """
    gt_map = parse_ground_truth(ground_truth_df)
    pred_map = parse_ground_truth(predictions_df)
    
    tp, fp, fn = 0, 0, 0
    
    all_keys = set(gt_map.keys()).union(set(pred_map.keys()))
    
    for key in all_keys:
        gt_set = gt_map.get(key, set())
        pred_set = pred_map.get(key, set())
        
        tp += len(gt_set.intersection(pred_set))
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

import argparse
from pathlib import Path
import sys
import time

from src.data_loading import load_train_data
from src.inference import predict_test_set
from src.model import build_training_set, train_model
from src.normalization import normalize_source_df
from src.validate import validate


def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Training & Inference Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "validate", "all"],
        default="all",
        help="Execution mode: 'train' to train model, 'predict' to run inference, 'validate' to run submission validator, or 'all'.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Root path of dataset containing train/ and test/ directories.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/lgbm_matcher.pkl",
        help="Path to save or load trained model file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to write candidate_pairs.tsv and matching_results.tsv output files.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Optional number of Source 1 entities to sample for fast local dev/testing.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional fraction of Source 1 entities to sample for fast local dev/testing.",
    )
    args = parser.parse_args()

    overall_start = time.perf_counter()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        for alt in [Path("../../dataset"), Path("../dataset")]:
            if alt.exists():
                dataset_path = alt
                break

    print("=" * 70)
    print(f"   BUSINESS ENTITY RESOLUTION PIPELINE (MODE: {args.mode.upper()})   ")
    print("=" * 70)
    print(f"Dataset Location : {dataset_path.resolve()}")
    if args.sample_n:
        print(f"Sampling Mode    : --sample-n {args.sample_n:,} Source 1 entities")
    elif args.sample_frac:
        print(f"Sampling Mode    : --sample-frac {args.sample_frac * 100:.1f}% Source 1 entities")
    else:
        print("Sampling Mode    : FULL DATASET (No sampling)")
    print("=" * 70 + "\n")

    if args.mode in ["train", "all"]:
        print("==================== STAGE 1: TRAINING MODEL ====================")
        train_data = load_train_data(
            dataset_dir=dataset_path, sample_n=args.sample_n, sample_frac=args.sample_frac
        )
        if not train_data:
            print("Error: Training data not found. Exiting training stage.")
            if args.mode == "train":
                sys.exit(1)
        else:
            feature_df, cand_df, diag, vectorizers = build_training_set(
                train_data=train_data,
                dataset_dir=dataset_path,
                sample_n=args.sample_n,
                sample_frac=args.sample_frac,
            )
            s1_df = normalize_source_df(train_data["train_source1"])
            gt_df = train_data["train_ground_truth"]

            train_model(
                feature_df=feature_df,
                gt_df=gt_df,
                s1_df=s1_df,
                model_save_path=args.model_path,
            )

    if args.mode in ["predict", "all"]:
        print("==================== STAGE 2: TEST INFERENCE ====================")
        try:
            predict_test_set(
                dataset_dir=dataset_path,
                model_path=args.model_path,
                output_dir=args.output_dir,
                sample_n=args.sample_n,
                sample_frac=args.sample_frac,
            )
        except FileNotFoundError as e:
            print(f"Inference error: {e}")
            sys.exit(1)

    if args.mode in ["validate", "predict", "all"]:
        print("==================== STAGE 3: VALIDATING SUBMISSION FILES ====================")
        success = validate(output_dir=args.output_dir, dataset_dir=str(dataset_path))
        if not success:
            sys.exit(1)

    overall_elapsed = time.perf_counter() - overall_start
    print("=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {overall_elapsed:.2f}s ({overall_elapsed/60:.2f} min)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

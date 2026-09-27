import os
from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd


def load_tsv(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a tab-separated (.tsv) file into a pandas DataFrame with explicit string dtypes.
    
    Args:
        filepath: Path to the TSV file.
        
    Returns:
        pd.DataFrame containing the file contents with string dtypes.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath.resolve()}")
    return pd.read_csv(filepath, sep="\t", dtype=str, engine="c")


def sample_source1_df(
    s1_df: pd.DataFrame,
    sample_n: Optional[int] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Randomly sample Source 1 entities for fast local dev/debug runs.
    
    Args:
        s1_df: Source 1 DataFrame.
        sample_n: Exact number of S1 rows to sample.
        sample_frac: Fraction of S1 rows to sample.
        random_state: Seed for reproducibility.
        
    Returns:
        Sampled Source 1 DataFrame.
    """
    if sample_n is not None and sample_n > 0:
        n = min(sample_n, len(s1_df))
        print(f"Sampling {n} Source 1 entities (out of {len(s1_df)} total)...")
        return s1_df.sample(n=n, random_state=random_state).reset_index(drop=True)
    elif sample_frac is not None and 0.0 < sample_frac < 1.0:
        print(f"Sampling {sample_frac * 100:.1f}% Source 1 entities (out of {len(s1_df)} total)...")
        return s1_df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    return s1_df


def load_train_data(
    dataset_dir: Union[str, Path] = "dataset",
    sample_n: Optional[int] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Load training datasets and ground truth with optional Source 1 sampling.
    
    Args:
        dataset_dir: Root directory containing dataset/train/ and dataset/test/.
        sample_n: Optional number of Source 1 entities to sample.
        sample_frac: Optional fraction of Source 1 entities to sample.
        random_state: Seed for sampling reproducibility.
        
    Returns:
        Dictionary mapping dataset file names to DataFrames.
    """
    train_dir = Path(dataset_dir) / "train"
    files = {
        "train_source1": train_dir / "train_source1.tsv",
        "train_source2": train_dir / "train_source2.tsv",
        "train_source3": train_dir / "train_source3.tsv",
        "train_ground_truth": train_dir / "train_ground_truth.tsv",
    }

    data = {}
    for name, path in files.items():
        if path.exists():
            data[name] = load_tsv(path)
        else:
            print(f"Warning: {path} does not exist.")

    if "train_source1" in data and (sample_n is not None or sample_frac is not None):
        data["train_source1"] = sample_source1_df(
            data["train_source1"], sample_n=sample_n, sample_frac=sample_frac, random_state=random_state
        )
        sampled_s1_ids = set(data["train_source1"]["entity_id"].astype(str))

        if "train_ground_truth" in data:
            gt_df = data["train_ground_truth"]
            gt_s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else gt_df.columns[0]
            data["train_ground_truth"] = gt_df[gt_df[gt_s1_col].astype(str).isin(sampled_s1_ids)].reset_index(drop=True)

    return data


def load_test_data(
    dataset_dir: Union[str, Path] = "dataset",
    sample_n: Optional[int] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Load test datasets with optional Source 1 sampling.
    
    Args:
        dataset_dir: Root directory containing dataset/train/ and dataset/test/.
        sample_n: Optional number of test Source 1 entities to sample.
        sample_frac: Optional fraction of test Source 1 entities to sample.
        random_state: Seed for sampling reproducibility.
        
    Returns:
        Dictionary mapping dataset file names to DataFrames.
    """
    test_dir = Path(dataset_dir) / "test"
    files = {
        "test_source1": test_dir / "test_source1.tsv",
        "test_source2": test_dir / "test_source2.tsv",
        "test_source3": test_dir / "test_source3.tsv",
    }

    data = {}
    for name, path in files.items():
        if path.exists():
            data[name] = load_tsv(path)
        else:
            print(f"Warning: {path} does not exist.")

    if "test_source1" in data and (sample_n is not None or sample_frac is not None):
        data["test_source1"] = sample_source1_df(
            data["test_source1"], sample_n=sample_n, sample_frac=sample_frac, random_state=random_state
        )

    return data


def load_all_data(dataset_dir: Union[str, Path] = "dataset") -> Dict[str, pd.DataFrame]:
    """Load all train and test datasets.
    
    Args:
        dataset_dir: Root directory of dataset.
        
    Returns:
        Dictionary mapping dataset names to DataFrames.
    """
    data = {}
    data.update(load_train_data(dataset_dir))
    data.update(load_test_data(dataset_dir))
    return data


def sanity_check(
    data_dict: Optional[Dict[str, pd.DataFrame]] = None,
    dataset_dir: Union[str, Path] = "dataset",
    sample_size: int = 3,
) -> None:
    """Print sanity-check diagnostics including row counts, null counts, and sample rows per file."""
    if data_dict is None:
        print(f"Loading data from directory: {dataset_dir}")
        data_dict = load_all_data(dataset_dir)

    if not data_dict:
        print("No datasets loaded. Please check dataset path.")
        return

    print("=" * 60)
    print("DATASET SANITY CHECK REPORT")
    print("=" * 60)

    for name, df in data_dict.items():
        print(f"\n--- File: {name} ---")
        print(f"Row Count   : {len(df):,}")
        print(f"Column Count: {len(df.columns)}")
        print(f"Columns     : {list(df.columns)}")
        print("\nNull Value Counts:")
        null_counts = df.isnull().sum()
        for col, val in null_counts.items():
            print(f"  {col:20s}: {val:,}")

        print(f"\nSample Rows (top {min(sample_size, len(df))}):")
        print(df.head(sample_size).to_string(index=False))
        print("-" * 60)


if __name__ == "__main__":
    possible_paths = [Path("dataset"), Path("../../dataset"), Path("../dataset")]
    found_dir = None
    for p in possible_paths:
        if p.exists():
            found_dir = p
            break
    if found_dir:
        sanity_check(dataset_dir=found_dir)
    else:
        sanity_check(dataset_dir="dataset")

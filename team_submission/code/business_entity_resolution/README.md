# Business Entity Resolution

> **Amazon ML Challenge 2026 | Team Submission**

End-to-end multi-source business entity resolution pipeline using multi-key candidate
blocking, pairwise string & TF-IDF feature engineering, LightGBM binary classification,
and per-entity macro-averaged F₀.₅ threshold optimization.

---

## Project Structure

```
code/business_entity_resolution/
├── README.md
├── requirements.txt
├── models/
│   └── lgbm_matcher.pkl          # trained model (produced by --mode train)
├── output/
│   ├── candidate_pairs.tsv       # submission file 1 (produced by --mode predict)
│   └── matching_results.tsv      # submission file 2 (produced by --mode predict)
└── src/
    ├── __init__.py
    ├── data_loading.py            # TSV ingestion, sampling
    ├── normalization.py           # text normalization, PIN extraction
    ├── blocking.py                # multi-key inverted-index candidate generation
    ├── features.py                # 14-dim pairwise feature extraction
    ├── model.py                   # LightGBM training + F0.5 threshold grid search
    ├── inference.py               # test-set scoring and submission file writer
    ├── evaluate.py                # macro F0.5 metric implementation
    ├── validate.py                # submission format validator
    └── main.py                    # CLI entry point
```

---

## Dataset Requirements

Place the following TSV files under `dataset/train/` and `dataset/test/` relative to
this directory (or pass `--dataset_dir` to point elsewhere):

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

**File schemas:**
- **Source files**: `entity_id`, `business_name`, `business_address`, `country`
- **Ground truth**: `source1_entity_id`, `matched_entity_ids` (comma-separated entity IDs)

---

## Installation

```bash
# From the code/business_entity_resolution/ directory:
pip install -r requirements.txt
```

All dependencies are pinned in `requirements.txt`. Python 3.9+ is recommended.

---

## Running the Pipeline

All commands are run from `code/business_entity_resolution/` as the working directory.

### Mode: `train` — Train the model only

Loads training data, runs blocking + feature extraction, trains LightGBM, tunes the
F₀.₅ threshold, and saves the model to `models/lgbm_matcher.pkl`.

```bash
python -m src.main --mode train
```

### Mode: `predict` — Run inference only (requires trained model)

Loads the saved model, runs blocking + feature extraction on test data, applies the
tuned threshold, and writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`.

```bash
python -m src.main --mode predict
```

### Mode: `validate` — Validate submission files only

Checks that existing output files conform to the required submission format
(row counts, column names, deduplication, no self-matches).

```bash
python -m src.main --mode validate
```

### Mode: `all` — Full end-to-end pipeline (train → predict → validate)

```bash
python -m src.main --mode all
```

---

## Optional Flags

### `--sample-n N`

Sample **N** Source 1 entities (and their corresponding candidates/ground truth rows)
for fast local development and testing. Useful for smoke-testing the pipeline without
loading the full 2.2M-row dataset.

```bash
# Train on a 5,000-entity subsample (fast dev/debug run)
python -m src.main --mode train --sample-n 5000

# Full train+predict+validate on 5,000 entities
python -m src.main --mode all --sample-n 5000
```

### `--sample-frac F`

Alternative to `--sample-n`: sample a **fraction** (0.0–1.0) of Source 1 entities.

```bash
# Train on 10% of the dataset
python -m src.main --mode train --sample-frac 0.1
```

### `--dataset_dir PATH`

Override the default dataset root (default: `dataset`). Useful when running from a
different working directory or when the dataset lives elsewhere.

```bash
python -m src.main --mode train --dataset_dir ../../dataset
```

### `--model_path PATH`

Override the default model save/load path (default: `models/lgbm_matcher.pkl`).

```bash
python -m src.main --mode predict --model_path models/lgbm_matcher_v2.pkl
```

### `--output_dir PATH`

Override the default output directory (default: `output`).

```bash
python -m src.main --mode predict --output_dir output/run_v2
```

---

## Pipeline Architecture

```
Source TSVs
    │
    ▼
[normalization.py]   Lowercase, strip punctuation, expand legal suffixes &
                     address abbreviations, extract PIN/ZIP codes
    │
    ▼
[blocking.py]        Multi-key candidate generation:
                       1. Inverted-index token blocking on name_tokens
                       2. NYSIIS phonetic blocking on primary name token
                       3. Character trigram blocking on name_norm
                       4. Address-token blocking on address_norm
                     Union of all four strategies → candidate pairs
    │
    ▼
[features.py]        14-dimensional pairwise feature vector per candidate pair:
                     name_token_jaccard, name_levenshtein, name_jaro_winkler,
                     name_tfidf_cosine, name_stripped_exact_match,
                     name_len_diff_norm, addr_token_jaccard, addr_levenshtein,
                     addr_jaro_winkler, addr_tfidf_cosine, pin_exact_match,
                     pin_both_present, addr_token_overlap, country_exact_match
    │
    ▼
[model.py]           LightGBM binary classifier
                     GroupShuffleSplit on source1_entity_id (no leakage)
                     Grid search tau* in [0.05, 0.95] to max macro F0.5
    │
    ▼
[inference.py]       Score test candidates → apply tau* → write submission TSVs
    │
    ▼
[validate.py]        Format compliance check
```

---

## Output Files

| File | Description |
|---|---|
| `output/candidate_pairs.tsv` | All candidate (S1, candidate) pairs considered |
| `output/matching_results.tsv` | Final predicted matches after threshold filtering |
| `models/lgbm_matcher.pkl` | Serialized trained LightGBM model + threshold |

---

## Reproducibility

All random seeds are fixed. Given identical input data, the pipeline produces identical
outputs. The model is MIT-licensed (`lightgbm`) with ~15,000 tree parameters — well
under the 8B parameter competition ceiling.

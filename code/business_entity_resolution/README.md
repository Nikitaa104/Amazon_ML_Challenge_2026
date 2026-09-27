# Business Entity Resolution

This project implements an end-to-end, multi-source business entity resolution pipeline using multi-key candidate blocking, string & TF-IDF feature engineering, LightGBM classification with group-aware cross validation, and per-entity $F_{0.5}$ threshold optimization.

## Project Structure

```
code/business_entity_resolution/
├── README.md
├── requirements.txt
├── models/
│   └── lgbm_matcher.pkl
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
└── src/
    ├── __init__.py
    ├── data_loading.py
    ├── normalization.py
    ├── blocking.py
    ├── features.py
    ├── model.py
    ├── inference.py
    ├── evaluate.py
    └── main.py
```

## Dataset Requirements

Data files should be tab-separated (`.tsv`) files located under `dataset/train/` and `dataset/test/`:

- `dataset/train/train_source1.tsv`
- `dataset/train/train_source2.tsv`
- `dataset/train/train_source3.tsv`
- `dataset/train/train_ground_truth.tsv`
- `dataset/test/test_source1.tsv`
- `dataset/test/test_source2.tsv`
- `dataset/test/test_source3.tsv`

### File Schemas
- **Source files**: `entity_id`, `business_name`, `business_address`, `country`
- **Ground truth**: `source1_entity_id`, `matched_entity_ids` (comma-separated entity IDs)

## Pipeline Architecture

1. **Normalization (`src/normalization.py`)**:
   - Lowercases, strips punctuation, and collapses whitespace.
   - Normalizes legal entity suffixes (`inc` -> `incorporated`, `corp` -> `corporation`, `pvt` -> `private`, `ltd` -> `limited`, `llc`, etc.).
   - Expands address abbreviations (`st` -> `street`, `rd` -> `road`, `bd` -> `boulevard`).
   - Extracts 5-6 digit PIN/ZIP codes (`pin_code`).
   - Produces `name_norm`, `name_tokens`, `address_norm`, `address_tokens`, `pin_code`.

2. **Candidate Blocking (`src/blocking.py`)**:
   - Multi-key candidate retrieval combining Token Overlap, Soundex Phonetic indexing, PIN-code matching, and Character Trigram matching.
   - Computes candidate reduction ratio and blocking recall ceiling.

3. **Feature Engineering (`src/features.py`)**:
   - Pairwise metric calculations: token Jaccard, normalized Levenshtein ratio, Jaro-Winkler distance, 2-4 char n-gram TF-IDF cosine similarity, exact core name match, length difference, PIN presence/equality, address token overlap, and country exact match.

4. **Model Training & Validation (`src/model.py`)**:
   - Group-aware train/validation split (`GroupShuffleSplit` on `source1_entity_id`).
   - Trains LightGBM binary classifier (`LGBMClassifier`).
   - Grid-searches threshold to maximize per-Source-1 macro-averaged $F_{0.5}$ metric.

5. **Inference & Submission Output (`src/inference.py`)**:
   - Runs normalization, blocking, and feature extraction on test sets.
   - Generates `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
   - Guarantees exact 1-to-1 representation of every test Source 1 entity, deduplicated match sets, and exclusion of self-IDs.

## Execution CLI

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Data Sanity Check:**
   ```bash
   python -m src.data_loading
   ```

3. **Train Model:**
   ```bash
   python -m src.main --mode train
   ```

4. **Run Test Inference:**
   ```bash
   python -m src.main --mode predict
   ```

5. **Run End-to-End Pipeline (Train + Predict):**
   ```bash
   python -m src.main --mode all
   ```

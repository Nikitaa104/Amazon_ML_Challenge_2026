# Business Entity Resolution — Solution Documentation

> **Amazon ML Challenge 2026 | Team Submission**

---

## 1. Methodology Overview

This solution implements a **pairwise entity resolution** pipeline that resolves business
records from Source 1 against candidate records from Source 2 and Source 3. The approach
separates the problem into two sequential stages: a recall-maximizing **blocking /
candidate generation stage**, followed by a precision-tuned **LightGBM binary classifier**
that scores candidate pairs. The final decision threshold is tuned to directly maximize
the competition metric — **macro-averaged F₀.₅ per Source 1 entity**.

### Pipeline Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌───────────────────────┐
│  Source 1 / 2 / 3   │────►│  Text Normalization  │────►│   Multi-Key Blocking  │
│  (.tsv files)       │     │  (normalization.py)  │     │   (blocking.py)       │
└─────────────────────┘     └─────────────────────┘     └──────────┬────────────┘
                                                                    │ candidate pairs
                                                                    ▼
┌─────────────────────┐     ┌─────────────────────┐     ┌───────────────────────┐
│  Output Submission  │◄────│ Inference & Scoring  │◄────│  Pairwise Features    │
│  (candidate_pairs   │     │ (inference.py)       │◄────│  (features.py)        │
│   matching_results) │     │  + tuned threshold   │     │                       │
└─────────────────────┘     └──────────┬──────────┘     └───────────────────────┘
                                        │ uses trained model
                             ┌──────────┴──────────┐
                             │  LightGBM Classifier │
                             │  + F0.5 Threshold    │
                             │  Tuning (model.py)   │
                             └─────────────────────┘
```

### Stage Summary

| Stage | Module | Purpose |
|---|---|---|
| Text Normalization | `src/normalization.py` | Lowercase, strip punctuation, expand legal suffixes & address abbreviations, extract PIN codes |
| Candidate Blocking | `src/blocking.py` | Union of four inverted indices to generate candidate pairs |
| Feature Engineering | `src/features.py` | 14-dimensional pairwise similarity vector per candidate pair |
| Model Training | `src/model.py` | Group-aware LightGBM training + F₀.₅ threshold grid search |
| Inference | `src/inference.py` | Score test candidate pairs, apply threshold, write submission files |
| Validation | `src/validate.py` | Verify output format compliance |

---

## 2. Blocking & Candidate Generation Strategy

The blocking stage reduces the O(|S1| x (|S2| + |S3|)) Cartesian product search space
by constructing **four complementary inverted indices** whose candidate sets are
**unioned** — any blocking key hit surfaces the pair for downstream scoring.

### Blocking Strategies

| # | Strategy | Key Used | Captures |
|---|---|---|---|
| 1 | **Inverted-index token blocking** | Unigram tokens from `name_tokens` (normalized business name) | Entities sharing any core name word |
| 2 | **NYSIIS phonetic blocking** | NYSIIS code of the primary name token | Phonetic name variants (e.g. "Smyth" vs. "Smith") |
| 3 | **Character trigram blocking** | Character 3-grams from `name_norm` | Typo/spelling variations, partial name matches |
| 4 | **Address-token blocking** | Unigram tokens from `address_norm` | Entities at the same address with differing names |

High-frequency tokens (appearing in > 10% of documents, `max_token_freq_ratio=0.10`) are
pruned from token-blocking postings to prevent pair explosion on generic business words.

### Tuned Hyperparameters (post dev-set tuning)

| Parameter | Change | Effect |
|---|---|---|
| `max_candidates_per_s1` | Loosened from baseline | Allows more candidates per S1 entity |
| `max_token_postings` | Loosened from baseline | Accepts longer posting lists for common tokens |
| `max_trigram_postings` | Loosened from baseline | Accepts longer trigram posting lists |

### Empirical Recall Ceiling — 5,000-entity Dev Sample

| Configuration | Blocking Recall Ceiling | Avg Candidates per S1 Entity |
|---|---|---|
| Baseline (original caps, Soundex only, no address blocking) | **26.19%** | ~75 |
| **Updated (loosened caps + NYSIIS + address blocking)** | **51.18%** | ~299.5 |

> **TODO: insert final full-scale numbers here**
>
> Once the current full-scale training run on the 2.2M-row dataset completes, fill in:
> - Full-scale blocking recall ceiling: `___`
> - Full-scale average candidates per S1 entity: `___`
> - Blocking stage wall-clock runtime: `___` minutes

---

## 3. Feature Engineering Details

For each surviving candidate pair (S1 entity, Candidate entity), a **14-dimensional numeric
feature vector** is computed. Features are computed **only on candidate pairs post-blocking**
— never on the full Cartesian product.

| Category | Feature Name | Description |
| :--- | :--- | :--- |
| **Name** | `name_token_jaccard` | Jaccard token-set similarity on core `name_tokens` |
| **Name** | `name_levenshtein` | Normalized Levenshtein edit distance ratio on `name_norm` (via `rapidfuzz`) |
| **Name** | `name_jaro_winkler` | Jaro-Winkler string similarity on `name_norm` |
| **Name** | `name_tfidf_cosine` | Cosine similarity of character 2-4-gram TF-IDF vectors (fit on all names) |
| **Name** | `name_stripped_exact_match` | Binary flag: exact match after legal-suffix stripping |
| **Name** | `name_len_diff_norm` | Normalized absolute length difference: |l1 - l2| / max(l1, l2, 1) |
| **Address** | `addr_token_jaccard` | Jaccard token-set similarity on `address_norm` |
| **Address** | `addr_levenshtein` | Normalized Levenshtein edit distance ratio on `address_norm` (via `rapidfuzz`) |
| **Address** | `addr_jaro_winkler` | Jaro-Winkler string similarity on `address_norm` |
| **Address** | `addr_tfidf_cosine` | Cosine similarity of character 2-4-gram TF-IDF vectors (fit on all addresses) |
| **Address** | `pin_exact_match` | Binary flag: exact match between extracted PIN/ZIP codes |
| **Address** | `pin_both_present` | Binary flag: both records contain non-empty PIN codes |
| **Address** | `addr_token_overlap` | Token overlap ratio relative to minimum token-set length |
| **Country** | `country_exact_match` | Binary flag: exact country string equality |

Libraries: `rapidfuzz` (Levenshtein, Jaro-Winkler), `scikit-learn` TfidfVectorizer
(character n-grams), standard Python set ops (Jaccard, token overlap).

---

## 4. Model Architecture & License Compliance

### Model

- **Framework**: LightGBM Binary Classifier (`lightgbm.LGBMClassifier`)
- **License**: MIT License — fully compliant with competition requirements
- **Parameter count**: ~15,000 tree split parameters (well under the 8 Billion parameter ceiling)

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 200 |
| `learning_rate` | 0.05 |
| `max_depth` | 6 |
| `num_leaves` | 31 |

### Group-Aware Train / Validation Split

Dataset splitting uses `GroupShuffleSplit` on `source1_entity_id` to prevent data
leakage. All candidate pairs for a given S1 entity fall entirely in the train fold
or the validation fold — never split across both.

### Decision Threshold Tuning for Macro F0.5

The threshold tau* is selected via grid search over tau in [0.05, 0.95] to maximize
macro-averaged F0.5 on the validation split — not accuracy or plain F1.

    F0.5 = (1.25 x Precision x Recall) / (0.25 x Precision + Recall)

Edge cases handled per S1 entity:
- Both empty (|T| = 0, |P| = 0): F0.5 = 1.0
- False positive on true singleton (|T| = 0, |P| > 0): F0.5 = 0.0
- Zero denominator: F0.5 = 0.0

### Performance Results

> **TODO: insert final numbers here** (fill in from training log once run completes)
>
> - Validation macro-averaged F0.5: `___`
> - Validation Precision: `___`
> - Validation Recall: `___`
> - Optimal threshold tau*: `___`
> - Total training pipeline wall-clock time: `___` minutes
> - Feature engineering runtime: `___` minutes
> - Blocking stage runtime: `___` minutes

---

## 5. Known Limitations

### (a) Blocking Recall Ceiling Is a Hard Upper Bound on Model Quality

The blocking stage structurally cannot surface every true match. Some true positive pairs
are never generated as candidates, so the downstream classifier cannot recover them
regardless of its accuracy. On the 5,000-entity dev sample, the ceiling reached **51.18%**
after tuning, meaning ~49% of true matches remain structurally unreachable. The full-scale
ceiling is pending (see TODO in Section 2).

### (b) France Is an Unseen Country at Train Time

The training data does not include French entities, but the test set does. The pipeline
handles this gracefully — global normalization applies, the country exact-match feature will
fire correctly — but **generalization to French business name and address patterns is
unverified**. Precision and recall on French test entities may be lower than on countries
seen during training.

### (c) Candidate Set Size Was Traded Off Against Recall

The updated ranking criterion rewards smaller `candidate_pairs.tsv` per entity in addition
to match quality. Loosening blocking caps improves recall ceiling but increases candidate
volume (~299.5 avg per S1 entity on dev sample). This was a deliberate tradeoff: more
candidates yields a higher recall ceiling and better potential F0.5, at the cost of a
larger submission file and slower feature computation.

---

## 6. Reproducibility

All randomness is seeded. The full pipeline is reproducible given identical input data.

```bash
# Working directory: code/business_entity_resolution/
pip install -r requirements.txt
python -m src.main --mode all
```

For fast local testing without the full 2.2M-row dataset:

```bash
python -m src.main --mode train --sample-n 5000
```

---

## 7. TODO Checklist (Fill In After Training Run Completes)

- [ ] **Blocking recall ceiling — full scale** (Section 2 table)
- [ ] **Avg candidates per S1 entity — full scale** (Section 2 table)
- [ ] **Blocking stage runtime** (Section 2 TODO block)
- [ ] **Feature engineering runtime** (Section 4 TODO block)
- [ ] **Total pipeline runtime** (Section 4 TODO block)
- [ ] **Validation macro-averaged F0.5** (Section 4 TODO block)
- [ ] **Validation Precision** (Section 4 TODO block)
- [ ] **Validation Recall** (Section 4 TODO block)
- [ ] **Optimal threshold tau*** (Section 4 TODO block)
- [ ] **France entity spot-check** — verify inference output has non-empty predictions for French test entities

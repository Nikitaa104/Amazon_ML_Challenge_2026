"""
Feature extraction for business entity resolution.

PERFORMANCE: All pairwise string similarity metrics are computed using batch/
vectorized operations — NO Python for-loops per pair:
  - Levenshtein / Jaro-Winkler: rapidfuzz list comps (fast C-level bindings)
  - TF-IDF cosine: sparse element-wise matrix product
  - Token Jaccard / overlap: vectorized Python over sets
  - All scalar features: NumPy array ops
"""

import time
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import rapidfuzz
    from rapidfuzz import distance, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    import difflib

try:
    import jellyfish
    HAS_JELLYFISH = True
except ImportError:
    HAS_JELLYFISH = False


# ──────────────────────────────────────────────────────────────────────────────
# Single-pair helpers kept for backward-compat (not called in main pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def compute_levenshtein_ratio(str1: str, str2: str) -> float:
    """Compute normalized Levenshtein ratio in range [0, 1]."""
    if not str1 or not str2:
        return 1.0 if str1 == str2 else 0.0
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(str1, str2) / 100.0
    import difflib
    return difflib.SequenceMatcher(None, str1, str2).ratio()


def compute_jaro_winkler(str1: str, str2: str) -> float:
    """Compute Jaro-Winkler similarity score in range [0, 1]."""
    if not str1 or not str2:
        return 1.0 if str1 == str2 else 0.0
    if HAS_JELLYFISH:
        return jellyfish.jaro_winkler_similarity(str1, str2)
    elif HAS_RAPIDFUZZ:
        return distance.JaroWinkler.similarity(str1, str2)
    import difflib
    return difflib.SequenceMatcher(None, str1, str2).ratio()


def compute_token_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0


def compute_token_overlap_ratio(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute token overlap ratio relative to minimum set size."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    min_len = min(len(tokens1), len(tokens2))
    return intersection / min_len if min_len > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF vectorizer fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_tfidf_vectorizers(
    all_dfs: List[pd.DataFrame],
    max_sample_size: int = 50000,
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    """Fit shared character (2-4 n-gram) TF-IDF vectorizers on names and addresses."""
    start_t = time.perf_counter()
    name_corpus: List[str] = []
    addr_corpus: List[str] = []

    for df in all_dfs:
        name_col = "name_norm" if "name_norm" in df.columns else "clean_name"
        addr_col = "address_norm" if "address_norm" in df.columns else "clean_address"

        if name_col in df.columns:
            s = df[name_col].dropna().astype(str)
            if len(s) > max_sample_size:
                s = s.sample(max_sample_size, random_state=42)
            name_corpus.extend(s.tolist())

        if addr_col in df.columns:
            s = df[addr_col].dropna().astype(str)
            if len(s) > max_sample_size:
                s = s.sample(max_sample_size, random_state=42)
            addr_corpus.extend(s.tolist())

    if not name_corpus:
        name_corpus = [""]
    if not addr_corpus:
        addr_corpus = [""]

    # Using max_features to limit vocabulary size and significantly speed up 
    # vectorization/dot products at scale
    name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=100_000)
    name_vec.fit(name_corpus)

    addr_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=100_000)
    addr_vec.fit(addr_corpus)

    elapsed = time.perf_counter() - start_t
    print(f"Fit TF-IDF vectorizers on sample of {len(name_corpus):,} names & addresses in {elapsed:.2f}s.", flush=True)
    return name_vec, addr_vec


# ──────────────────────────────────────────────────────────────────────────────
# Batch string-similarity helpers (vectorized, NO per-pair Python loops)
# ──────────────────────────────────────────────────────────────────────────────

def _batch_levenshtein(queries: List[str], choices: List[str]) -> np.ndarray:
    """Element-wise normalized Levenshtein ratio in [0,1] for parallel lists."""
    n = len(queries)
    if n == 0:
        return np.array([], dtype=np.float32)

    if HAS_RAPIDFUZZ:
        # List comprehension calling C-level fuzz.ratio is very fast. 
        # process.cdist computes a full NxN matrix which causes OOM.
        ratios = [fuzz.ratio(q, c) for q, c in zip(queries, choices)]
        return np.array(ratios, dtype=np.float32) / 100.0

    import difflib
    return np.array(
        [difflib.SequenceMatcher(None, q, c).ratio() for q, c in zip(queries, choices)],
        dtype=np.float32,
    )


def _batch_jaro_winkler(queries: List[str], choices: List[str]) -> np.ndarray:
    """Element-wise Jaro-Winkler similarity in batch."""
    n = len(queries)
    if n == 0:
        return np.array([], dtype=np.float32)

    if HAS_RAPIDFUZZ:
        scores = [distance.JaroWinkler.similarity(q, c) for q, c in zip(queries, choices)]
        return np.array(scores, dtype=np.float32)

    if HAS_JELLYFISH:
        return np.array(
            [jellyfish.jaro_winkler_similarity(q, c) for q, c in zip(queries, choices)],
            dtype=np.float32,
        )
    import difflib
    return np.array(
        [difflib.SequenceMatcher(None, q, c).ratio() for q, c in zip(queries, choices)],
        dtype=np.float32,
    )


def _batch_token_jaccard(tokens1_list: List[Set[str]], tokens2_list: List[Set[str]]) -> np.ndarray:
    """Vectorized Jaccard similarity over parallel lists of token sets."""
    n = len(tokens1_list)
    if n == 0:
        return np.array([], dtype=np.float32)
    result = np.empty(n, dtype=np.float32)
    for i, (t1, t2) in enumerate(zip(tokens1_list, tokens2_list)):
        if not t1 or not t2:
            result[i] = 0.0
        else:
            inter = len(t1 & t2)
            union = len(t1 | t2)
            result[i] = inter / union if union else 0.0
    return result


def _batch_token_overlap(tokens1_list: List[Set[str]], tokens2_list: List[Set[str]]) -> np.ndarray:
    """Vectorized token overlap ratio relative to the smaller set."""
    n = len(tokens1_list)
    if n == 0:
        return np.array([], dtype=np.float32)
    result = np.empty(n, dtype=np.float32)
    for i, (t1, t2) in enumerate(zip(tokens1_list, tokens2_list)):
        if not t1 or not t2:
            result[i] = 0.0
        else:
            inter = len(t1 & t2)
            mn = min(len(t1), len(t2))
            result[i] = inter / mn if mn else 0.0
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main feature extraction — fully vectorized, NO per-pair Python loop
# ──────────────────────────────────────────────────────────────────────────────

def extract_candidate_features(
    candidate_pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    target_dfs: Dict[str, pd.DataFrame],
    fitted_vectorizers: Optional[Tuple[TfidfVectorizer, TfidfVectorizer]] = None,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, Tuple[TfidfVectorizer, TfidfVectorizer]]:
    """
    Extract pairwise numeric similarity features between Source 1 records and
    candidate records.  ALL string metrics are computed in bulk:
      - TF-IDF cosine via sparse matrix element-wise product
      - Levenshtein / Jaro-Winkler via rapidfuzz list comps
      - Token Jaccard / overlap via vectorized Python over sets
      - Scalar features via NumPy array ops
    Per-stage wall-clock times are printed for profiling.
    """
    feature_start = time.perf_counter()

    FEATURE_COLS = [
        "source1_entity_id", "candidate_entity_id",
        "name_token_jaccard", "name_levenshtein", "name_jaro_winkler",
        "name_tfidf_cosine", "name_stripped_exact_match", "name_len_diff_norm",
        "addr_token_jaccard", "addr_levenshtein", "addr_jaro_winkler",
        "addr_tfidf_cosine", "pin_exact_match", "pin_both_present",
        "addr_token_overlap", "country_exact_match",
    ]

    if candidate_pairs.empty:
        return pd.DataFrame(columns=FEATURE_COLS), fitted_vectorizers or (None, None)

    # ── 0. Fit / reuse vectorizers ────────────────────────────────────────────
    t0 = time.perf_counter()
    all_dfs = [s1_df] + list(target_dfs.values())
    if fitted_vectorizers is None:
        name_vec, addr_vec = fit_tfidf_vectorizers(all_dfs)
    else:
        name_vec, addr_vec = fitted_vectorizers
    print(f"  [features] vectorizer ready in {time.perf_counter()-t0:.2f}s", flush=True)

    # ── 1. Build entity lookup map (only for required IDs) ────────────────────
    t0 = time.perf_counter()
    s1_col   = "source1_entity_id" if "source1_entity_id" in candidate_pairs.columns else "entity_id_1"
    cand_col = "candidate_entity_id" if "candidate_entity_id" in candidate_pairs.columns else "entity_id_2"

    id1_arr = candidate_pairs[s1_col].astype(str).to_numpy()
    id2_arr = candidate_pairs[cand_col].astype(str).to_numpy()
    required_eids = set(id1_arr) | set(id2_arr)

    entity_map: Dict[str, dict] = {}
    for df in all_dfs:
        name_norm_col     = "name_norm"      if "name_norm"      in df.columns else "clean_name"
        name_stripped_col = "name_stripped"  if "name_stripped"  in df.columns else name_norm_col
        name_tokens_col   = "name_tokens"    if "name_tokens"    in df.columns else name_norm_col
        addr_norm_col     = "address_norm"   if "address_norm"   in df.columns else "clean_address"
        addr_tokens_col   = "address_tokens" if "address_tokens" in df.columns else addr_norm_col

        sub = df[df["entity_id"].isin(required_eids)]
        if sub.empty:
            continue

        for row in sub.itertuples(index=False):
            eid = str(getattr(row, "entity_id"))
            if eid in entity_map:
                continue
            nn  = str(getattr(row, name_norm_col,      "") or "")
            ns  = str(getattr(row, name_stripped_col,   nn) or "")
            nt  = str(getattr(row, name_tokens_col,     nn) or "")
            an  = str(getattr(row, addr_norm_col,       "") or "")
            at  = str(getattr(row, addr_tokens_col,     an) or "")
            pin = str(getattr(row, "pin_code", "") or "").strip()
            cty = str(getattr(row, "country",  "") or "").strip().lower()
            entity_map[eid] = {
                "name_norm":      nn,
                "name_stripped":  ns,
                "name_tokens":    set(nt.split()),
                "address_norm":   an,
                "address_tokens": set(at.split()),
                "pin_code":       pin,
                "country":        cty,
            }
    print(f"  [features] entity map ({len(entity_map):,} entities) in {time.perf_counter()-t0:.2f}s", flush=True)

    # ── 2. Gather aligned arrays for all pairs ────────────────────────────────
    t0 = time.perf_counter()
    EMPTY = {"name_norm": "", "name_stripped": "", "name_tokens": set(),
             "address_norm": "", "address_tokens": set(), "pin_code": "", "country": ""}

    recs1 = [entity_map.get(eid, EMPTY) for eid in id1_arr]
    recs2 = [entity_map.get(eid, EMPTY) for eid in id2_arr]

    n1_texts  = [r["name_norm"]      for r in recs1]
    n2_texts  = [r["name_norm"]      for r in recs2]
    a1_texts  = [r["address_norm"]   for r in recs1]
    a2_texts  = [r["address_norm"]   for r in recs2]
    ns1_arr   = np.array([r["name_stripped"]  for r in recs1])
    ns2_arr   = np.array([r["name_stripped"]  for r in recs2])
    nt1_list  = [r["name_tokens"]    for r in recs1]
    nt2_list  = [r["name_tokens"]    for r in recs2]
    at1_list  = [r["address_tokens"] for r in recs1]
    at2_list  = [r["address_tokens"] for r in recs2]
    pin1_arr  = np.array([r["pin_code"] for r in recs1])
    pin2_arr  = np.array([r["pin_code"] for r in recs2])
    cty1_arr  = np.array([r["country"]  for r in recs1])
    cty2_arr  = np.array([r["country"]  for r in recs2])
    n_len1    = np.array([len(t) for t in n1_texts], dtype=np.float32)
    n_len2    = np.array([len(t) for t in n2_texts], dtype=np.float32)
    print(f"  [features] aligned {len(id1_arr):,} pairs in {time.perf_counter()-t0:.2f}s", flush=True)

    # ── 3. TF-IDF cosine (sparse element-wise product, no loop) ──────────────
    t0 = time.perf_counter()
    # Batch transform (fast)
    name_mat1 = name_vec.transform(n1_texts)
    name_mat2 = name_vec.transform(n2_texts)
    
    # We must explicitly cast to csr_matrix before multiply to avoid inefficient 
    # dense operations depending on scikit-learn version
    from scipy.sparse import csr_matrix
    if not isinstance(name_mat1, csr_matrix): name_mat1 = name_mat1.tocsr()
    if not isinstance(name_mat2, csr_matrix): name_mat2 = name_mat2.tocsr()
    
    # Multiply element-wise then sum rows (fast exact cosine distance for sparse matrices)
    name_tfidf_cosines = np.asarray(name_mat1.multiply(name_mat2).sum(axis=1)).ravel().astype(np.float32)

    addr_mat1 = addr_vec.transform(a1_texts)
    addr_mat2 = addr_vec.transform(a2_texts)
    if not isinstance(addr_mat1, csr_matrix): addr_mat1 = addr_mat1.tocsr()
    if not isinstance(addr_mat2, csr_matrix): addr_mat2 = addr_mat2.tocsr()
    addr_tfidf_cosines = np.asarray(addr_mat1.multiply(addr_mat2).sum(axis=1)).ravel().astype(np.float32)
    print(f"  [features] TF-IDF cosine in {time.perf_counter()-t0:.2f}s", flush=True)

    # ── 4. Batch Levenshtein ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    name_lev = _batch_levenshtein(n1_texts, n2_texts)
    addr_lev = _batch_levenshtein(a1_texts, a2_texts)
    print(f"  [features] Levenshtein in {time.perf_counter()-t0:.2f}s", flush=True)

    # ── 5. Batch Jaro-Winkler ────────────────────────────────────────────────
    t0 = time.perf_counter()
    name_jw = _batch_jaro_winkler(n1_texts, n2_texts)
    addr_jw = _batch_jaro_winkler(a1_texts, a2_texts)
    print(f"  [features] Jaro-Winkler in {time.perf_counter()-t0:.2f}s", flush=True)

    # ── 6. Token Jaccard & overlap (vectorized over sets) ────────────────────
    t0 = time.perf_counter()
    name_jaccard = _batch_token_jaccard(nt1_list, nt2_list)
    addr_jaccard = _batch_token_jaccard(at1_list, at2_list)
    addr_overlap = _batch_token_overlap(at1_list, at2_list)
    print(f"  [features] Jaccard/overlap in {time.perf_counter()-t0:.2f}s", flush=True)

    # ── 7. Scalar features (NumPy vectorized) ────────────────────────────────
    t0 = time.perf_counter()
    name_exact    = (ns1_arr == ns2_arr).astype(np.float32)
    max_n_len     = np.maximum(n_len1, n_len2)
    max_n_len[max_n_len == 0] = 1.0
    name_len_diff = np.abs(n_len1 - n_len2) / max_n_len

    invalid_vals = {"", "none", "nan"}
    valid_pin1   = np.array([p.lower() not in invalid_vals for p in pin1_arr], dtype=np.float32)
    valid_pin2   = np.array([p.lower() not in invalid_vals for p in pin2_arr], dtype=np.float32)
    pin_both     = valid_pin1 * valid_pin2
    pin_exact    = pin_both * (pin1_arr == pin2_arr).astype(np.float32)
    country_match = ((cty1_arr != "") & (cty2_arr != "") & (cty1_arr == cty2_arr)).astype(np.float32)
    print(f"  [features] scalar features in {time.perf_counter()-t0:.2f}s", flush=True)

    # ── 8. Assemble output DataFrame ──────────────────────────────────────────
    t0 = time.perf_counter()
    feature_df = pd.DataFrame({
        "source1_entity_id":         id1_arr,
        "candidate_entity_id":       id2_arr,
        "name_token_jaccard":        name_jaccard,
        "name_levenshtein":          name_lev,
        "name_jaro_winkler":         name_jw,
        "name_tfidf_cosine":         name_tfidf_cosines,
        "name_stripped_exact_match": name_exact,
        "name_len_diff_norm":        name_len_diff,
        "addr_token_jaccard":        addr_jaccard,
        "addr_levenshtein":          addr_lev,
        "addr_jaro_winkler":         addr_jw,
        "addr_tfidf_cosine":         addr_tfidf_cosines,
        "pin_exact_match":           pin_exact,
        "pin_both_present":          pin_both,
        "addr_token_overlap":        addr_overlap,
        "country_exact_match":       country_match,
    })
    print(f"  [features] DataFrame assembled in {time.perf_counter()-t0:.2f}s", flush=True)

    total_elapsed = time.perf_counter() - feature_start
    rate = len(feature_df) / total_elapsed if total_elapsed > 0 else 0
    print(
        f"Extracted features for {len(feature_df):,} pairs in {total_elapsed:.2f}s "
        f"({rate:,.0f} pairs/s).",
        flush=True,
    )

    return feature_df, (name_vec, addr_vec)


def extract_pair_features(
    pairs: pd.DataFrame, df1: pd.DataFrame, df2: pd.DataFrame
) -> pd.DataFrame:
    """Wrapper function for candidate pair feature extraction between two DataFrames."""
    feature_df, _ = extract_candidate_features(pairs, df1, {"source2": df2})
    return feature_df

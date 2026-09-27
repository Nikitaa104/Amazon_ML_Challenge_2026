import difflib
import time
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import rapidfuzz
    from rapidfuzz import distance, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

try:
    import jellyfish
    HAS_JELLYFISH = True
except ImportError:
    HAS_JELLYFISH = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def compute_levenshtein_ratio(str1: str, str2: str) -> float:
    """Compute normalized Levenshtein ratio in range [0, 1]."""
    if not str1 or not str2:
        return 1.0 if str1 == str2 else 0.0
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(str1, str2) / 100.0
    return difflib.SequenceMatcher(None, str1, str2).ratio()


def compute_jaro_winkler(str1: str, str2: str) -> float:
    """Compute Jaro-Winkler similarity score in range [0, 1]."""
    if not str1 or not str2:
        return 1.0 if str1 == str2 else 0.0
    if HAS_JELLYFISH:
        return jellyfish.jaro_winkler_similarity(str1, str2)
    elif HAS_RAPIDFUZZ:
        return distance.JaroWinkler.similarity(str1, str2)
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


def fit_tfidf_vectorizers(
    all_dfs: List[pd.DataFrame],
    max_sample_size: int = 200000,
) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    """Fit shared character (2-4 n-gram) TF-IDF vectorizers on names and addresses.
    
    Args:
        all_dfs: List of DataFrames (S1, S2, S3) containing normalized columns.
        max_sample_size: Maximum corpus sample size to fit TF-IDF efficiently.
        
    Returns:
        Tuple of (name_vectorizer, address_vectorizer).
    """
    start_t = time.perf_counter()
    name_corpus = []
    addr_corpus = []

    for df in all_dfs:
        name_col = "name_norm" if "name_norm" in df.columns else "clean_name"
        addr_col = "address_norm" if "address_norm" in df.columns else "clean_address"

        if name_col in df.columns:
            name_corpus.extend(df[name_col].fillna("").astype(str).tolist())
        if addr_col in df.columns:
            addr_corpus.extend(df[addr_col].fillna("").astype(str).tolist())

    if not name_corpus:
        name_corpus = [""]
    if not addr_corpus:
        addr_corpus = [""]

    if len(name_corpus) > max_sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(name_corpus), size=max_sample_size, replace=False)
        name_corpus = [name_corpus[i] for i in idx]

    if len(addr_corpus) > max_sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(addr_corpus), size=max_sample_size, replace=False)
        addr_corpus = [addr_corpus[i] for i in idx]

    name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    name_vec.fit(name_corpus)

    addr_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    addr_vec.fit(addr_corpus)

    elapsed = time.perf_counter() - start_t
    print(f"Fit TF-IDF vectorizers on sample of {len(name_corpus):,} names & addresses in {elapsed:.2f}s.")
    return name_vec, addr_vec


def extract_candidate_features(
    candidate_pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    target_dfs: Dict[str, pd.DataFrame],
    fitted_vectorizers: Optional[Tuple[TfidfVectorizer, TfidfVectorizer]] = None,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, Tuple[TfidfVectorizer, TfidfVectorizer]]:
    """Extract pairwise numeric similarity features between Source 1 records and candidate records.
    
    Args:
        candidate_pairs: DataFrame containing candidate pairs.
        s1_df: Source 1 normalized DataFrame.
        target_dfs: Dict mapping target source name ('source2', 'source3') to normalized DataFrames.
        fitted_vectorizers: Optional pre-fitted vectorizers.
        show_progress: Whether to show progress bars and stage timers.
        
    Returns:
        Tuple of (feature_matrix_df, (name_vec, addr_vec)).
    """
    start_t = time.perf_counter()
    if candidate_pairs.empty:
        empty_df = pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "name_token_jaccard",
                "name_levenshtein",
                "name_jaro_winkler",
                "name_tfidf_cosine",
                "name_stripped_exact_match",
                "name_len_diff_norm",
                "addr_token_jaccard",
                "addr_levenshtein",
                "addr_jaro_winkler",
                "addr_tfidf_cosine",
                "pin_exact_match",
                "pin_both_present",
                "addr_token_overlap",
                "country_exact_match",
            ]
        )
        return empty_df, fitted_vectorizers or (None, None)

    all_dfs = [s1_df] + list(target_dfs.values())
    if fitted_vectorizers is None:
        name_vec, addr_vec = fit_tfidf_vectorizers(all_dfs)
    else:
        name_vec, addr_vec = fitted_vectorizers

    s1_col = "source1_entity_id" if "source1_entity_id" in candidate_pairs.columns else "entity_id_1"
    cand_col = "candidate_entity_id" if "candidate_entity_id" in candidate_pairs.columns else "entity_id_2"

    id1_list = candidate_pairs[s1_col].astype(str).tolist()
    id2_list = candidate_pairs[cand_col].astype(str).tolist()

    # Highly optimized: only build entity map for entity IDs present in candidate pairs!
    required_eids = set(id1_list).union(set(id2_list))

    entity_map: Dict[str, dict] = {}
    for df in all_dfs:
        name_norm_col = "name_norm" if "name_norm" in df.columns else "clean_name"
        name_stripped_col = "name_stripped" if "name_stripped" in df.columns else name_norm_col
        name_tokens_col = "name_tokens" if "name_tokens" in df.columns else name_norm_col
        addr_norm_col = "address_norm" if "address_norm" in df.columns else "clean_address"
        addr_tokens_col = "address_tokens" if "address_tokens" in df.columns else addr_norm_col
        pin_col = "pin_code"
        country_col = "country"

        # Filter df to required_eids only for speed
        mask = df["entity_id"].astype(str).isin(required_eids)
        sub_df = df[mask]
        if sub_df.empty:
            continue

        e_ids = sub_df["entity_id"].astype(str).tolist()
        n_norms = sub_df[name_norm_col].fillna("").astype(str).tolist()
        n_strippeds = sub_df[name_stripped_col].fillna("").astype(str).tolist() if name_stripped_col in sub_df.columns else n_norms
        n_tokens_list = sub_df[name_tokens_col].fillna("").astype(str).tolist() if name_tokens_col in sub_df.columns else n_norms
        a_norms = sub_df[addr_norm_col].fillna("").astype(str).tolist()
        a_tokens_list = sub_df[addr_tokens_col].fillna("").astype(str).tolist() if addr_tokens_col in sub_df.columns else a_norms
        pins = sub_df[pin_col].fillna("").astype(str).tolist() if pin_col in sub_df.columns else [""] * len(sub_df)
        countries = sub_df[country_col].fillna("").astype(str).tolist() if country_col in sub_df.columns else [""] * len(sub_df)

        for e_id, nn, ns, nt, an, at, pin, c in zip(
            e_ids, n_norms, n_strippeds, n_tokens_list, a_norms, a_tokens_list, pins, countries
        ):
            if e_id not in entity_map:
                entity_map[e_id] = {
                    "name_norm": nn,
                    "name_stripped": ns,
                    "name_tokens": set(nt.split()),
                    "address_norm": an,
                    "address_tokens": set(at.split()),
                    "pin_code": pin.strip(),
                    "country": c.strip().lower(),
                }

    # 2. Compute TF-IDF sparse matrix dot products in bulk for candidate pairs
    n1_texts = [entity_map[id1]["name_norm"] if id1 in entity_map else "" for id1 in id1_list]
    n2_texts = [entity_map[id2]["name_norm"] if id2 in entity_map else "" for id2 in id2_list]

    a1_texts = [entity_map[id1]["address_norm"] if id1 in entity_map else "" for id1 in id1_list]
    a2_texts = [entity_map[id2]["address_norm"] if id2 in entity_map else "" for id2 in id2_list]

    # Vectorized TF-IDF cosine similarity via sparse matrix multiplication
    name_mat1 = name_vec.transform(n1_texts)
    name_mat2 = name_vec.transform(n2_texts)
    name_tfidf_cosines = np.asarray(name_mat1.multiply(name_mat2).sum(axis=1)).ravel()

    addr_mat1 = addr_vec.transform(a1_texts)
    addr_mat2 = addr_vec.transform(a2_texts)
    addr_tfidf_cosines = np.asarray(addr_mat1.multiply(addr_mat2).sum(axis=1)).ravel()

    # 3. Compute remaining string features over candidate pair tuples
    feature_rows = []
    iterator = zip(id1_list, id2_list, name_tfidf_cosines, addr_tfidf_cosines)

    if show_progress and HAS_TQDM:
        iterator = tqdm(iterator, total=len(candidate_pairs), desc=f"Extracting features ({len(candidate_pairs):,} candidate pairs)")

    for id1, id2, name_tfidf_sim, addr_tfidf_sim in iterator:
        rec1 = entity_map.get(id1, {})
        rec2 = entity_map.get(id2, {})

        if not rec1 or not rec2:
            continue

        n1, n2 = rec1["name_norm"], rec2["name_norm"]
        ns1, ns2 = rec1["name_stripped"], rec2["name_stripped"]
        nt1, nt2 = rec1["name_tokens"], rec2["name_tokens"]

        a1, a2 = rec1["address_norm"], rec2["address_norm"]
        at1, at2 = rec1["address_tokens"], rec2["address_tokens"]

        pin1, pin2 = rec1["pin_code"], rec2["pin_code"]
        c1, c2 = rec1["country"], rec2["country"]

        name_token_jaccard = compute_token_jaccard(nt1, nt2)
        name_levenshtein = compute_levenshtein_ratio(n1, n2)
        name_jaro_winkler = compute_jaro_winkler(n1, n2)

        name_stripped_exact_match = 1.0 if (ns1 and ns2 and ns1 == ns2) else 0.0
        max_n_len = max(len(n1), len(n2), 1)
        name_len_diff_norm = abs(len(n1) - len(n2)) / max_n_len

        addr_token_jaccard = compute_token_jaccard(at1, at2)
        addr_levenshtein = compute_levenshtein_ratio(a1, a2)
        addr_jaro_winkler = compute_jaro_winkler(a1, a2)

        valid_pin1 = pin1 and pin1.lower() not in ("none", "nan", "")
        valid_pin2 = pin2 and pin2.lower() not in ("none", "nan", "")

        pin_both_present = 1.0 if (valid_pin1 and valid_pin2) else 0.0
        pin_exact_match = 1.0 if (pin_both_present == 1.0 and pin1 == pin2) else 0.0
        addr_token_overlap = compute_token_overlap_ratio(at1, at2)

        country_exact_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0

        feat = {
            "source1_entity_id": id1,
            "candidate_entity_id": id2,
            "name_token_jaccard": name_token_jaccard,
            "name_levenshtein": name_levenshtein,
            "name_jaro_winkler": name_jaro_winkler,
            "name_tfidf_cosine": float(name_tfidf_sim),
            "name_stripped_exact_match": name_stripped_exact_match,
            "name_len_diff_norm": name_len_diff_norm,
            "addr_token_jaccard": addr_token_jaccard,
            "addr_levenshtein": addr_levenshtein,
            "addr_jaro_winkler": addr_jaro_winkler,
            "addr_tfidf_cosine": float(addr_tfidf_sim),
            "pin_exact_match": pin_exact_match,
            "pin_both_present": pin_both_present,
            "addr_token_overlap": addr_token_overlap,
            "country_exact_match": country_exact_match,
        }
        feature_rows.append(feat)

    feature_df = pd.DataFrame(feature_rows)
    elapsed = time.perf_counter() - start_t
    if show_progress:
        print(f"Extracted features for {len(feature_df):,} pairs in {elapsed:.2f}s ({len(feature_df)/elapsed:,.0f} pairs/s).")

    return feature_df, (name_vec, addr_vec)


def extract_pair_features(
    pairs: pd.DataFrame, df1: pd.DataFrame, df2: pd.DataFrame
) -> pd.DataFrame:
    """Wrapper function for candidate pair feature extraction between two DataFrames."""
    feature_df, _ = extract_candidate_features(pairs, df1, {"source2": df2})
    return feature_df

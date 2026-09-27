from collections import defaultdict
import time
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd

try:
    import jellyfish

    def get_phonetic_code(token: str) -> str:
        """Compute Soundex phonetic code for a string token using jellyfish."""
        if not token:
            return ""
        return jellyfish.soundex(token)

except ImportError:

    def get_phonetic_code(token: str) -> str:
        """Fallback phonetic representation if jellyfish is unavailable."""
        if not token:
            return ""
        return token[:4].lower()

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Generic corporate stopwords to prune upfront from token blocking
GENERIC_STOPWORDS: Set[str] = {
    "ltd", "limited", "pvt", "private", "inc", "incorporated", "co", "company",
    "corp", "corporation", "llc", "gmbh", "sa", "bv", "nv", "spa", "services",
    "service", "enterprises", "enterprise", "trading", "group", "solutions",
    "solution", "global", "international", "tech", "technology", "technologies",
    "consulting", "holdings", "ventures", "industries", "industry", "india",
    "us", "usa", "uk", "north", "south", "east", "west", "city", "market",
    "mart", "center", "centre", "shop", "store", "hotel", "restaurant",
    "express", "logistics", "auto", "motors", "pharma", "medical", "hospital",
    "school", "club", "association", "foundation", "bank", "finance",
    "financial", "management", "systems", "digital", "media", "studio",
    "design", "the", "and", "for", "of", "in", "to", "a", "an", "at", "by", "or"
}


def get_trigrams(text: str) -> Set[str]:
    """Generate set of character 3-grams from a text string (prefix only for efficiency)."""
    if not text or len(text) < 3:
        return set()
    no_space = "".join(text.split())
    no_space = no_space[:10]
    return {no_space[i : i + 3] for i in range(len(no_space) - 2)}


class BlockingIndex:
    """Inverted index data structure for multi-key candidate blocking."""

    def __init__(
        self,
        max_token_freq_ratio: float = 0.001,
        max_token_postings: int = 5000,
        max_trigram_postings: int = 2000,
        max_phonetic_postings: int = 2000,
    ):
        self.max_token_freq_ratio = max_token_freq_ratio
        self.max_token_postings = max_token_postings
        self.max_trigram_postings = max_trigram_postings
        self.max_phonetic_postings = max_phonetic_postings

        self.token_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.phonetic_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.pincode_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.trigram_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.high_freq_tokens: Set[str] = set(GENERIC_STOPWORDS)
        self.pruned_trigrams: Set[str] = set()

    def build_index(self, target_dfs: Dict[str, pd.DataFrame], show_progress: bool = True) -> float:
        """Build inverted indices across target DataFrames using fast tuple iteration."""
        start_t = time.perf_counter()
        token_doc_counts: Dict[str, int] = defaultdict(int)
        total_records = sum(len(df) for df in target_dfs.values())

        # Pass 1: count token doc frequencies for pruning
        for source_name, df in target_dfs.items():
            name_col = "name_tokens" if "name_tokens" in df.columns else "clean_name"
            names = df[name_col].fillna("").astype(str).tolist()
            for name_val in names:
                tokens = set(name_val.split())
                for t in tokens:
                    if t and t not in GENERIC_STOPWORDS:
                        token_doc_counts[t] += 1

        max_allowed_freq = min(
            self.max_token_postings,
            max(50, int(total_records * self.max_token_freq_ratio))
        )
        for t, c in token_doc_counts.items():
            if c > max_allowed_freq:
                self.high_freq_tokens.add(t)

        if show_progress:
            print(
                f"High-freq token pruning: {len(self.high_freq_tokens):,} tokens pruned "
                f"(>{max_allowed_freq:,} occurrences or corporate stopwords)."
            )

        # Pass 2: populate indices
        for source_name, df in target_dfs.items():
            name_tokens_col = "name_tokens" if "name_tokens" in df.columns else "clean_name"
            name_norm_col = "name_norm" if "name_norm" in df.columns else "clean_name"
            pin_col = "pin_code"

            e_ids = df["entity_id"].astype(str).tolist()
            n_tokens_list = df[name_tokens_col].fillna("").astype(str).tolist()
            n_norms = df[name_norm_col].fillna("").astype(str).tolist()
            pin_codes = df[pin_col].fillna("").astype(str).tolist() if pin_col in df.columns else [""] * len(df)

            iterator = zip(e_ids, n_tokens_list, n_norms, pin_codes)
            if show_progress and HAS_TQDM and len(df) > 10000:
                iterator = tqdm(iterator, total=len(df), desc=f"Indexing {source_name} ({len(df):,} records)")

            for e_id, n_tokens_str, n_norm, pin_code in iterator:
                item = (e_id, source_name)
                tokens = set(n_tokens_str.split())

                # 1. Token index (freq-pruned)
                for token in tokens:
                    if token and token not in self.high_freq_tokens:
                        lst = self.token_index[token]
                        if len(lst) < self.max_token_postings:
                            lst.append(item)

                # 2. Phonetic index (cap at max_phonetic_postings)
                sig_tokens = [t for t in tokens if t not in GENERIC_STOPWORDS]
                first_token = sig_tokens[0] if sig_tokens else next(iter(tokens), "")
                if first_token:
                    p_code = get_phonetic_code(first_token)
                    if p_code and len(self.phonetic_index[p_code]) < self.max_phonetic_postings:
                        self.phonetic_index[p_code].append(item)

                # 3. PIN code index (high selectivity)
                clean_pin = pin_code.strip()
                if clean_pin and clean_pin.lower() not in ("none", "nan", ""):
                    self.pincode_index[clean_pin].append(item)

                # 4. Trigram index (prefix-only + hard posting-list cap)
                trigrams = get_trigrams(n_norm)
                for tg in trigrams:
                    if tg not in self.pruned_trigrams:
                        lst = self.trigram_index[tg]
                        if len(lst) < self.max_trigram_postings:
                            lst.append(item)
                        else:
                            self.pruned_trigrams.add(tg)
                            del self.trigram_index[tg]

        elapsed = time.perf_counter() - start_t
        if show_progress:
            print(
                f"Built inverted index for {total_records:,} target records in {elapsed:.2f}s.\n"
                f"  Token keys      : {len(self.token_index):,}\n"
                f"  Phonetic keys   : {len(self.phonetic_index):,}\n"
                f"  PIN keys        : {len(self.pincode_index):,}\n"
                f"  Trigram keys    : {len(self.trigram_index):,}  "
                f"(pruned {len(self.pruned_trigrams):,} high-freq trigrams)"
            )
        return elapsed


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    target_dfs: Dict[str, pd.DataFrame],
    max_token_freq_ratio: float = 0.001,
    max_token_postings: int = 5000,
    max_trigram_postings: int = 2000,
    max_phonetic_postings: int = 2000,
    max_candidates_per_s1: int = 200,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Generate candidate pairs for all Source 1 entities using union of blocking keys."""
    start_t = time.perf_counter()
    index = BlockingIndex(
        max_token_freq_ratio=max_token_freq_ratio,
        max_token_postings=max_token_postings,
        max_trigram_postings=max_trigram_postings,
        max_phonetic_postings=max_phonetic_postings,
    )
    index.build_index(target_dfs, show_progress=show_progress)

    name_tokens_col = "name_tokens" if "name_tokens" in s1_df.columns else "clean_name"
    name_norm_col = "name_norm" if "name_norm" in s1_df.columns else "clean_name"
    pin_col = "pin_code"

    e_ids = s1_df["entity_id"].astype(str).tolist()
    n_tokens_list = s1_df[name_tokens_col].fillna("").astype(str).tolist()
    n_norms = s1_df[name_norm_col].fillna("").astype(str).tolist()
    pin_codes = s1_df[pin_col].fillna("").astype(str).tolist() if pin_col in s1_df.columns else [""] * len(s1_df)

    candidate_records = []

    iterator = zip(e_ids, n_tokens_list, n_norms, pin_codes)
    if show_progress and HAS_TQDM:
        iterator = tqdm(
            iterator,
            total=len(s1_df),
            desc=f"Blocking candidates for {len(s1_df):,} S1 records",
        )

    for s1_id, n_tokens_str, name_norm, pin_code in iterator:
        tokens = set(n_tokens_str.split())
        clean_pin = pin_code.strip()

        # matches dictionary stores count of keys that fired instead of a set 
        # to avoid huge overhead of set creation per pair
        matches: Dict[Tuple[str, str], int] = defaultdict(int)

        # 1. Token overlap blocking
        for token in tokens:
            if token and token not in index.high_freq_tokens:
                for item in index.token_index.get(token, []):
                    matches[item] += 1

        # 2. Phonetic blocking
        sig_tokens = [t for t in tokens if t not in GENERIC_STOPWORDS]
        first_token = sig_tokens[0] if sig_tokens else next(iter(tokens), "")
        if first_token:
            p_code = get_phonetic_code(first_token)
            if p_code:
                for item in index.phonetic_index.get(p_code, []):
                    matches[item] += 1

        # 3. Pin code blocking
        if clean_pin and clean_pin.lower() not in ("none", "nan", ""):
            for item in index.pincode_index.get(clean_pin, []):
                matches[item] += 1

        # 4. Trigram blocking
        for tg in get_trigrams(name_norm):
            if tg not in index.pruned_trigrams:
                for item in index.trigram_index.get(tg, []):
                    matches[item] += 1

        # Prioritize candidates by number of keys matched and cap at max_candidates_per_s1
        cand_items = sorted(matches.items(), key=lambda kv: kv[1], reverse=True)
        if len(cand_items) > max_candidates_per_s1:
            cand_items = cand_items[:max_candidates_per_s1]

        for (cand_id, source), keys_fired_count in cand_items:
            candidate_records.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                    "source": source,
                    "blocking_keys_fired": str(keys_fired_count), # Using count as placeholder
                    "entity_id_1": s1_id,
                    "entity_id_2": cand_id,
                }
            )

    cand_df = pd.DataFrame(candidate_records)
    elapsed = time.perf_counter() - start_t
    if show_progress:
        avg = len(cand_df) / len(s1_df) if len(s1_df) > 0 else 0
        print(
            f"Generated {len(cand_df):,} candidate pairs for {len(s1_df):,} S1 records "
            f"in {elapsed:.2f}s ({len(s1_df)/elapsed:,.0f} S1/s, avg {avg:.1f} cands/S1).",
            flush=True
        )

    return cand_df


def compute_blocking_diagnostics(
    candidate_pairs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    total_s1_count: int,
    total_target_count: int,
) -> Dict[str, float]:
    """Compute candidate blocking evaluation metrics."""
    gt_pairs: Set[Tuple[str, str]] = set()
    gt_s1_col = "source1_entity_id" if "source1_entity_id" in ground_truth_df.columns else ground_truth_df.columns[0]
    gt_match_col = "matched_entity_ids" if "matched_entity_ids" in ground_truth_df.columns else ground_truth_df.columns[1]

    for _, row in ground_truth_df.iterrows():
        s1_id = str(row[gt_s1_col]).strip()
        matched_str = str(row.get(gt_match_col, "") or "")
        if matched_str and matched_str.lower() not in ("nan", "none"):
            for m_id in matched_str.split(","):
                m_id = m_id.strip()
                if m_id:
                    gt_pairs.add((s1_id, m_id))

    cand_pairs: Set[Tuple[str, str]] = set()
    if not candidate_pairs_df.empty:
        c_s1_col = "source1_entity_id" if "source1_entity_id" in candidate_pairs_df.columns else "entity_id_1"
        c_cand_col = "candidate_entity_id" if "candidate_entity_id" in candidate_pairs_df.columns else "entity_id_2"
        for s1, c in zip(
            candidate_pairs_df[c_s1_col].astype(str).tolist(),
            candidate_pairs_df[c_cand_col].astype(str).tolist(),
        ):
            cand_pairs.add((s1, c))

    hits = len(gt_pairs.intersection(cand_pairs))
    total_gt = len(gt_pairs)
    recall_ceiling = hits / total_gt if total_gt > 0 else 0.0

    total_possible_pairs = total_s1_count * total_target_count
    actual_pairs = len(candidate_pairs_df)
    reduction_ratio = 1.0 - (actual_pairs / total_possible_pairs) if total_possible_pairs > 0 else 1.0
    avg_candidates_per_s1 = actual_pairs / total_s1_count if total_s1_count > 0 else 0.0

    return {
        "recall_ceiling": recall_ceiling,
        "reduction_ratio": reduction_ratio,
        "gt_matches_total": float(total_gt),
        "gt_matches_captured": float(hits),
        "candidate_pairs_count": float(actual_pairs),
        "avg_candidates_per_s1": avg_candidates_per_s1,
    }

from collections import defaultdict
import time
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd

try:
    import jellyfish

    def get_soundex_code(token: str) -> str:
        """Compute Soundex phonetic code for a string token using jellyfish."""
        if not token:
            return ""
        return jellyfish.soundex(token)

    def get_nysiis_code(token: str) -> str:
        """Compute NYSIIS phonetic code (secondary signal, more accurate for complex names)."""
        if not token:
            return ""
        try:
            return jellyfish.nysiis(token)
        except Exception:
            return token[:4].lower()

    def get_phonetic_code(token: str) -> str:
        """Compute Soundex phonetic code (primary, backward-compatible alias)."""
        return get_soundex_code(token)

except ImportError:

    def get_soundex_code(token: str) -> str:
        """Fallback if jellyfish unavailable."""
        if not token:
            return ""
        return token[:4].lower()

    def get_nysiis_code(token: str) -> str:
        """Fallback if jellyfish unavailable."""
        if not token:
            return ""
        return token[:5].lower()

    def get_phonetic_code(token: str) -> str:
        """Fallback if jellyfish unavailable."""
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

# Address-specific stopwords - very common address terms pruned from address blocking
ADDRESS_STOPWORDS: Set[str] = {
    "road", "rd", "street", "st", "avenue", "ave", "boulevard", "blvd",
    "drive", "dr", "lane", "ln", "court", "ct", "place", "pl", "square", "sq",
    "parkway", "pkwy", "highway", "hwy", "suite", "ste", "apartment", "apt",
    "floor", "fl", "building", "bldg", "sector", "block", "phase", "plot",
    "no", "number", "num", "near", "opp", "opposite", "behind", "next",
    "new", "old", "main", "cross", "circle", "ring", "inner", "outer",
    "the", "and", "of", "in", "at", "by", "to", "a", "an",
}


def get_trigrams(text: str) -> Set[str]:
    """Generate set of char 3-grams from a text string (prefix-only, 12 chars)."""
    if not text or len(text) < 3:
        return set()
    no_space = "".join(text.split())[:12]
    return {no_space[i : i + 3] for i in range(len(no_space) - 2)}


def get_address_trigrams(text: str) -> Set[str]:
    """Generate char 3-grams from first 15 chars of address (locality-level fuzzy matching)."""
    if not text or len(text) < 3:
        return set()
    no_space = "".join(text.split())[:15]
    return {no_space[i : i + 3] for i in range(len(no_space) - 2)}


def get_address_ngram_keys(addr_tokens_str: str) -> Set[str]:
    """Generate char 3-grams from normalized street/locality tokens (length >= 4, non-stopword).
    Catches street name fuzzy matches regardless of house/building prefix variations.
    """
    if not addr_tokens_str:
        return set()
    ngrams = set()
    tokens = [
        t for t in addr_tokens_str.split()
        if t not in ADDRESS_STOPWORDS and len(t) >= 4 and not t.isdigit()
    ]
    for token in tokens[:5]:
        for i in range(len(token) - 2):
            ngrams.add(token[i : i + 3])
    return ngrams


class BlockingIndex:
    """Inverted index data structure for multi-key candidate blocking.

    Signals (in priority order for max_candidates_per_s1 ranking):
      1. Name token overlap (freq-pruned)
      2a. Soundex phonetic (primary)
      2b. NYSIIS phonetic (secondary - better for complex/non-Anglo names)
      3. PIN code (high selectivity)
      4. Name trigrams (prefix, 12 chars)
      5. Address token overlap (independent signal - catches divergent-name same-entity)
      6. Address trigrams (prefix, 15 chars)

    Design constraints for scale (10M+ target records):
    - Corporate stopwords: pruned upfront from name tokens
    - Address stopwords: pruned upfront from address tokens
    - Token index: cap max token frequency at max_token_postings (default 5000)
    - Trigram index: prune any trigram key with posting list > max_trigram_postings (default 1500)
    - Phonetic index: capped at max_phonetic_postings per code (default 1000)
    - Pin index: retained as-is (high selectivity)
    """

    def __init__(
        self,
        max_token_freq_ratio: float = 0.001,
        max_token_postings: int = 5000,
        max_trigram_postings: int = 1500,
        max_phonetic_postings: int = 1000,
        max_addr_token_postings: int = 3000,
        use_nysiis: bool = True,
        use_address_blocking: bool = True,
    ):
        self.max_token_freq_ratio = max_token_freq_ratio
        self.max_token_postings = max_token_postings
        self.max_trigram_postings = max_trigram_postings
        self.max_phonetic_postings = max_phonetic_postings
        self.max_addr_token_postings = max_addr_token_postings
        self.use_nysiis = use_nysiis
        self.use_address_blocking = use_address_blocking

        self.token_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.phonetic_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.nysiis_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.pincode_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.trigram_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # Address-based indices (independent signal)
        self.addr_token_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.addr_trigram_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.addr_ngram_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        self.high_freq_tokens: Set[str] = set(GENERIC_STOPWORDS)
        self.high_freq_addr_tokens: Set[str] = set(ADDRESS_STOPWORDS)
        self.pruned_trigrams: Set[str] = set()
        self.pruned_addr_trigrams: Set[str] = set()
        self.pruned_addr_ngrams: Set[str] = set()

    def build_index(self, target_dfs: Dict[str, pd.DataFrame], show_progress: bool = True) -> float:
        """Build inverted indices across target DataFrames using fast tuple iteration."""
        start_t = time.perf_counter()
        token_doc_counts: Dict[str, int] = defaultdict(int)
        addr_token_doc_counts: Dict[str, int] = defaultdict(int)
        total_records = sum(len(df) for df in target_dfs.values())

        # Pass 1: count token doc frequencies for pruning
        for source_name, df in target_dfs.items():
            name_col = "name_tokens" if "name_tokens" in df.columns else "clean_name"
            for name_val in df[name_col].fillna("").astype(str):
                for t in set(name_val.split()):
                    if t and t not in GENERIC_STOPWORDS:
                        token_doc_counts[t] += 1

            if self.use_address_blocking and "address_tokens" in df.columns:
                for addr_val in df["address_tokens"].fillna("").astype(str):
                    for t in set(addr_val.split()):
                        if t and t not in ADDRESS_STOPWORDS and len(t) >= 3:
                            addr_token_doc_counts[t] += 1

        max_allowed_freq = min(
            self.max_token_postings,
            max(50, int(total_records * self.max_token_freq_ratio))
        )
        for t, c in token_doc_counts.items():
            if c > max_allowed_freq:
                self.high_freq_tokens.add(t)

        max_allowed_addr_freq = min(
            self.max_addr_token_postings,
            max(50, int(total_records * self.max_token_freq_ratio * 2))
        )
        for t, c in addr_token_doc_counts.items():
            if c > max_allowed_addr_freq:
                self.high_freq_addr_tokens.add(t)

        if show_progress:
            print(
                f"High-freq token pruning: {len(self.high_freq_tokens):,} name tokens pruned "
                f"(>{max_allowed_freq:,} occurrences or corporate stopwords)."
            )
            if self.use_address_blocking:
                print(
                    f"High-freq addr pruning : {len(self.high_freq_addr_tokens):,} address tokens pruned "
                    f"(>{max_allowed_addr_freq:,} occurrences or address stopwords)."
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
            addr_tokens_list = (
                df["address_tokens"].fillna("").astype(str).tolist()
                if "address_tokens" in df.columns else [""] * len(df)
            )
            addr_norms = (
                df["address_norm"].fillna("").astype(str).tolist()
                if "address_norm" in df.columns else [""] * len(df)
            )

            iterator = zip(e_ids, n_tokens_list, n_norms, pin_codes, addr_tokens_list, addr_norms)
            if show_progress and HAS_TQDM and len(df) > 10000:
                iterator = tqdm(
                    iterator, total=len(df),
                    desc=f"Indexing {source_name} ({len(df):,} records)"
                )

            for e_id, n_tokens_str, n_norm, pin_code, addr_tokens_str, addr_norm in iterator:
                item = (e_id, source_name)
                tokens = set(n_tokens_str.split())

                # 1. Token index (freq-pruned)
                for token in tokens:
                    if token and token not in self.high_freq_tokens:
                        lst = self.token_index[token]
                        if len(lst) < self.max_token_postings:
                            lst.append(item)

                # 2a. Soundex phonetic index (cap at max_phonetic_postings)
                significant_tokens = [t for t in tokens if t not in GENERIC_STOPWORDS]
                first_token = significant_tokens[0] if significant_tokens else next(iter(tokens), "")
                if first_token:
                    sdx_code = get_soundex_code(first_token)
                    if sdx_code and len(self.phonetic_index[sdx_code]) < self.max_phonetic_postings:
                        self.phonetic_index[sdx_code].append(item)

                # 2b. NYSIIS phonetic index (secondary - different error surface than Soundex)
                if self.use_nysiis and first_token:
                    nysiis_code = get_nysiis_code(first_token)
                    if nysiis_code and len(self.nysiis_index[nysiis_code]) < self.max_phonetic_postings:
                        self.nysiis_index[nysiis_code].append(item)

                # 3. PIN code index (high selectivity - no pruning needed)
                clean_pin = pin_code.strip()
                if clean_pin and clean_pin.lower() not in ("none", "nan", ""):
                    self.pincode_index[clean_pin].append(item)

                # 4. Name trigram index (prefix-only + hard posting-list cap)
                for tg in get_trigrams(n_norm):
                    if tg not in self.pruned_trigrams:
                        lst = self.trigram_index[tg]
                        if len(lst) < self.max_trigram_postings:
                            lst.append(item)
                        else:
                            self.pruned_trigrams.add(tg)
                            del self.trigram_index[tg]

                # 5. Address-token blocking (independent from name - catches divergent-name same-entity)
                if self.use_address_blocking and addr_tokens_str:
                    for token in set(addr_tokens_str.split()):
                        if token and token not in self.high_freq_addr_tokens and len(token) >= 3:
                            lst = self.addr_token_index[token]
                            if len(lst) < self.max_addr_token_postings:
                                lst.append(item)

                    # 6. Address trigrams (locality-level fuzzy matching)
                    for tg in get_address_trigrams(addr_norm):
                        if tg not in self.pruned_addr_trigrams:
                            lst = self.addr_trigram_index[tg]
                            if len(lst) < self.max_trigram_postings:
                                lst.append(item)
                            else:
                                self.pruned_addr_trigrams.add(tg)
                                del self.addr_trigram_index[tg]

                    # 7. Address token n-grams (shared 3-gram tokens on normalized street name)
                    for ng in get_address_ngram_keys(addr_tokens_str):
                        if ng not in self.pruned_addr_ngrams:
                            lst = self.addr_ngram_index[ng]
                            if len(lst) < self.max_trigram_postings:
                                lst.append(item)
                            else:
                                self.pruned_addr_ngrams.add(ng)
                                del self.addr_ngram_index[ng]

        elapsed = time.perf_counter() - start_t
        if show_progress:
            print(
                f"Built inverted index for {total_records:,} target records in {elapsed:.2f}s.\n"
                f"  Token keys       : {len(self.token_index):,}\n"
                f"  Phonetic (SDX)   : {len(self.phonetic_index):,}\n"
                f"  Phonetic (NYSIIS): {len(self.nysiis_index):,}\n"
                f"  PIN keys         : {len(self.pincode_index):,}\n"
                f"  Trigram keys     : {len(self.trigram_index):,}  "
                f"(pruned {len(self.pruned_trigrams):,} high-freq name trigrams)\n"
                f"  Addr-token keys  : {len(self.addr_token_index):,}\n"
                f"  Addr-tgram keys  : {len(self.addr_trigram_index):,}  "
                f"(pruned {len(self.pruned_addr_trigrams):,} high-freq addr trigrams)"
            )
        return elapsed


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    target_dfs: Dict[str, pd.DataFrame],
    max_token_freq_ratio: float = 0.001,
    max_token_postings: int = 5000,
    max_trigram_postings: int = 1500,
    max_phonetic_postings: int = 1000,
    max_candidates_per_s1: int = 300,
    max_addr_token_postings: int = 3000,
    use_nysiis: bool = True,
    use_address_blocking: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Generate candidate pairs for all Source 1 entities using union of blocking keys.

    Candidates are ranked by number of distinct blocking keys fired (descending) and
    capped at max_candidates_per_s1. This ensures the strongest candidates survive
    the cap while keeping total pairs manageable for ranking.

    Args:
        s1_df: Source 1 DataFrame (normalized).
        target_dfs: Dict mapping source name -> normalized target DataFrame.
        max_token_freq_ratio: Fraction of total records beyond which a name token is high-freq.
        max_token_postings: Hard cap on posting list size for name tokens.
        max_trigram_postings: Hard cap on posting list size for trigram keys.
        max_phonetic_postings: Hard cap on posting list size for phonetic codes.
        max_candidates_per_s1: Max candidates retained per S1 entity after union.
        max_addr_token_postings: Hard cap on posting list size for address tokens.
        use_nysiis: Whether to use NYSIIS as a secondary phonetic blocking signal.
        use_address_blocking: Whether to block on normalized address tokens.
        show_progress: Display tqdm progress bars and summary statistics.

    Returns:
        DataFrame of candidate pairs with columns: source1_entity_id, candidate_entity_id,
        source, blocking_keys_fired, num_keys_fired, entity_id_1, entity_id_2.
    """
    start_t = time.perf_counter()
    index = BlockingIndex(
        max_token_freq_ratio=max_token_freq_ratio,
        max_token_postings=max_token_postings,
        max_trigram_postings=max_trigram_postings,
        max_phonetic_postings=max_phonetic_postings,
        max_addr_token_postings=max_addr_token_postings,
        use_nysiis=use_nysiis,
        use_address_blocking=use_address_blocking,
    )
    index.build_index(target_dfs, show_progress=show_progress)

    name_tokens_col = "name_tokens" if "name_tokens" in s1_df.columns else "clean_name"
    name_norm_col = "name_norm" if "name_norm" in s1_df.columns else "clean_name"
    pin_col = "pin_code"

    e_ids = s1_df["entity_id"].astype(str).tolist()
    n_tokens_list = s1_df[name_tokens_col].fillna("").astype(str).tolist()
    n_norms = s1_df[name_norm_col].fillna("").astype(str).tolist()
    pin_codes = s1_df[pin_col].fillna("").astype(str).tolist() if pin_col in s1_df.columns else [""] * len(s1_df)
    addr_tokens_list = (
        s1_df["address_tokens"].fillna("").astype(str).tolist()
        if "address_tokens" in s1_df.columns else [""] * len(s1_df)
    )
    addr_norms = (
        s1_df["address_norm"].fillna("").astype(str).tolist()
        if "address_norm" in s1_df.columns else [""] * len(s1_df)
    )

    candidate_records = []

    iterator = zip(e_ids, n_tokens_list, n_norms, pin_codes, addr_tokens_list, addr_norms)
    if show_progress and HAS_TQDM:
        iterator = tqdm(
            iterator,
            total=len(s1_df),
            desc=f"Blocking candidates for {len(s1_df):,} S1 records",
        )

    for s1_id, n_tokens_str, name_norm, pin_code, addr_tokens_str, addr_norm in iterator:
        tokens = set(n_tokens_str.split())
        clean_pin = pin_code.strip()

        # matches maps (cand_id, source) -> set of blocking keys fired
        matches: Dict[Tuple[str, str], Set[str]] = {}

        def _add(cand_id, source, key):
            k = (cand_id, source)
            if k not in matches:
                matches[k] = set()
            matches[k].add(key)

        # 1. Token overlap blocking
        for token in tokens:
            if token and token not in index.high_freq_tokens:
                for cand_id, source in index.token_index.get(token, []):
                    _add(cand_id, source, "token_overlap")

        # 2a. Soundex phonetic blocking
        sig_tokens = [t for t in tokens if t not in GENERIC_STOPWORDS]
        first_token = sig_tokens[0] if sig_tokens else next(iter(tokens), "")
        if first_token:
            sdx_code = get_soundex_code(first_token)
            if sdx_code:
                for cand_id, source in index.phonetic_index.get(sdx_code, []):
                    _add(cand_id, source, "phonetic_soundex")

        # 2b. NYSIIS phonetic blocking (secondary - different error surface than Soundex)
        if use_nysiis and first_token:
            nysiis_code = get_nysiis_code(first_token)
            if nysiis_code:
                for cand_id, source in index.nysiis_index.get(nysiis_code, []):
                    _add(cand_id, source, "phonetic_nysiis")

        # 3. Pin code blocking
        if clean_pin and clean_pin.lower() not in ("none", "nan", ""):
            for cand_id, source in index.pincode_index.get(clean_pin, []):
                _add(cand_id, source, "pin_code")

        # 4. Trigram blocking on name (only retained non-pruned trigrams)
        for tg in get_trigrams(name_norm):
            if tg not in index.pruned_trigrams:
                for cand_id, source in index.trigram_index.get(tg, []):
                    _add(cand_id, source, "trigram_name")

        # 5. Address-token blocking (independent signal - catches divergent-name same-entity)
        if use_address_blocking and addr_tokens_str:
            for token in set(addr_tokens_str.split()):
                if token and token not in index.high_freq_addr_tokens and len(token) >= 3:
                    for cand_id, source in index.addr_token_index.get(token, []):
                        _add(cand_id, source, "addr_token")

            # 6. Address trigrams (locality-level fuzzy matching)
            for tg in get_address_trigrams(addr_norm):
                if tg not in index.pruned_addr_trigrams:
                    for cand_id, source in index.addr_trigram_index.get(tg, []):
                        _add(cand_id, source, "addr_trigram")

            # 7. Address token n-gram blocking (shared 3-grams on normalized street name)
            for ng in get_address_ngram_keys(addr_tokens_str):
                if ng not in index.pruned_addr_ngrams:
                    for cand_id, source in index.addr_ngram_index.get(ng, []):
                        _add(cand_id, source, "addr_token_ngram")

        # Rank by number of distinct blocking keys fired (more = stronger evidence)
        # and cap at max_candidates_per_s1
        ranked_matches = sorted(matches.items(), key=lambda x: len(x[1]), reverse=True)
        if max_candidates_per_s1 and len(ranked_matches) > max_candidates_per_s1:
            ranked_matches = ranked_matches[:max_candidates_per_s1]

        for (cand_id, source), keys_fired in ranked_matches:
            candidate_records.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                    "source": source,
                    "blocking_keys_fired": ",".join(sorted(keys_fired)),
                    "num_keys_fired": len(keys_fired),
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
            f"in {elapsed:.2f}s ({len(s1_df)/elapsed:,.0f} S1/s, avg {avg:.1f} cands/S1)."
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


def diagnose_missed_matches(
    candidate_pairs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    target_dfs: Dict[str, pd.DataFrame],
    n_samples: int = 15,
) -> pd.DataFrame:
    """Sample missed true-match pairs and classify failure reason.

    For each missed true-match pair, retrieves raw business_name and business_address
    strings from both sides so failure patterns can be inspected directly.

    Failure reason heuristics:
      NO_SHARED_TOKENS : name+addr token sets are disjoint -> alias or language variant
      NAME_TOKEN_GAP   : addr overlaps but name does not -> addr-only blocking needed
      ADDR_NOISE       : name overlaps but addr does not -> posting-cap or rare token
      POSTING_CAP      : both overlap -> dropped by max_token or per-S1 cap

    Args:
        candidate_pairs_df: Output of generate_candidate_pairs().
        ground_truth_df: Ground truth DataFrame.
        s1_df: Source 1 DataFrame (normalized).
        target_dfs: Dict of target DataFrames (source2, source3).
        n_samples: Max number of missed-match pairs to sample.

    Returns:
        DataFrame with columns: s1_id, target_id, target_source, s1_name, s1_address,
        target_name, target_address, shared_name_tokens, shared_address_tokens,
        likely_failure_reason.
    """
    gt_s1_col = "source1_entity_id" if "source1_entity_id" in ground_truth_df.columns else ground_truth_df.columns[0]
    gt_match_col = "matched_entity_ids" if "matched_entity_ids" in ground_truth_df.columns else ground_truth_df.columns[1]

    gt_pairs: Set[Tuple[str, str]] = set()
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

    missed = list(gt_pairs - cand_pairs)

    import random
    random.seed(42)
    sample = random.sample(missed, min(n_samples, len(missed)))

    sampled_s1_ids = {s1 for s1, _ in sample}
    sampled_target_ids = {t for _, t in sample}

    s1_lookup = {
        str(r["entity_id"]): r
        for _, r in s1_df[s1_df["entity_id"].astype(str).isin(sampled_s1_ids)].iterrows()
    }
    target_lookup: Dict[str, tuple] = {}
    for src_name, df in target_dfs.items():
        subset = df[df["entity_id"].astype(str).isin(sampled_target_ids)]
        for _, row in subset.iterrows():
            target_lookup[str(row["entity_id"])] = (row, src_name)

    rows = []
    for s1_id, target_id in sample:
        s1_row = s1_lookup.get(s1_id)
        target_info = target_lookup.get(target_id)
        if s1_row is None or target_info is None:
            continue
        target_row, target_src = target_info

        s1_name_toks = set(str(s1_row.get("name_tokens", "")).split())
        t_name_toks = set(str(target_row.get("name_tokens", "")).split())
        shared_name_toks = s1_name_toks & t_name_toks

        s1_addr_toks = set(str(s1_row.get("address_tokens", "")).split())
        t_addr_toks = set(str(target_row.get("address_tokens", "")).split())
        shared_addr_toks = s1_addr_toks & t_addr_toks

        if not shared_name_toks and not shared_addr_toks:
            reason = "NO_SHARED_TOKENS (name+addr disjoint: alias or language variant)"
        elif not shared_name_toks and shared_addr_toks:
            reason = "NAME_TOKEN_GAP (addr overlaps, name doesnt: addr-only blocking needed)"
        elif shared_name_toks and not shared_addr_toks:
            reason = "ADDR_NOISE (name overlaps, addr disjoint: posting-cap or rare token)"
        else:
            reason = "POSTING_CAP (both overlap: dropped by max_token or per-S1 cap)"

        rows.append({
            "s1_id": s1_id,
            "target_id": target_id,
            "target_source": target_src,
            "s1_name": str(s1_row.get("business_name", "")),
            "s1_address": str(s1_row.get("business_address", "")),
            "target_name": str(target_row.get("business_name", "")),
            "target_address": str(target_row.get("business_address", "")),
            "shared_name_tokens": " | ".join(sorted(shared_name_toks)),
            "shared_address_tokens": " | ".join(sorted(shared_addr_toks)),
            "likely_failure_reason": reason,
        })

    return pd.DataFrame(rows)

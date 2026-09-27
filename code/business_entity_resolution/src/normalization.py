import re
import time
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Extensible maps for legal entity suffixes
LEGAL_SUFFIX_MAP: Dict[str, str] = {
    # Common English / Global
    "inc": "incorporated",
    "incorporated": "incorporated",
    "corp": "corporation",
    "corporation": "corporation",
    "pvt": "private",
    "private": "private",
    "pvtd": "private",
    "ltd": "limited",
    "limited": "limited",
    "llc": "llc",
    "co": "company",
    "company": "company",
    "grp": "group",
    "group": "group",
    "tech": "technology",
    "technologies": "technology",
    "technology": "technology",
    "ent": "enterprise",
    "enterprises": "enterprise",
    "enterprise": "enterprise",
    "soln": "solutions",
    "solns": "solutions",
    "solutions": "solutions",
    "svc": "services",
    "svcs": "services",
    "services": "services",
    # German / European
    "gmbh": "gmbh",
    "ag": "ag",
    # French
    "sarl": "sarl",
    "sas": "sas",
    "sa": "sa",
    "ste": "societe",
    "societe": "societe",
    # Dutch / Spanish / Italian / UK
    "bv": "bv",
    "nv": "nv",
    "plc": "plc",
    "srl": "srl",
    "spa": "spa",
}

LEGAL_SUFFIX_TOKENS: Set[str] = set(LEGAL_SUFFIX_MAP.keys()).union(set(LEGAL_SUFFIX_MAP.values()))

ADDRESS_ABBREV_GLOBAL: Dict[str, str] = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "pkwy": "parkway",
    "pkway": "parkway",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "flr": "floor",
    "bldg": "building",
    "hwy": "highway",
}

ADDRESS_ABBREV_COUNTRY: Dict[str, Dict[str, str]] = {
    "france": {
        "bd": "boulevard",
        "av": "avenue",
        "r": "rue",
        "all": "allee",
        "imp": "impasse",
        "pl": "place",
        "res": "residence",
    },
    "fr": {
        "bd": "boulevard",
        "av": "avenue",
        "r": "rue",
        "all": "allee",
        "imp": "impasse",
        "pl": "place",
        "res": "residence",
    },
    "germany": {"str": "strasse", "str.": "strasse", "pl": "platz"},
    "de": {"str": "strasse", "str.": "strasse", "pl": "platz"},
}

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "of", "for", "in", "on", "at", "to", "with", "by", "no", "number", "num"
}

PIN_REGEX = re.compile(r"\b\d{5,6}\b")
PUNCT_REGEX = re.compile(r"[^\w\s]")
WHITESPACE_REGEX = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    """Lowercase string, strip punctuation, and collapse whitespace."""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    text = PUNCT_REGEX.sub(" ", text)
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    return text


def normalize_business_name(name: str) -> Tuple[str, str, str]:
    """Normalize business name, expanding legal suffixes and extracting core tokens."""
    cleaned = basic_clean(name)
    if not cleaned:
        return "", "", ""

    tokens = cleaned.split()
    norm_tokens = []
    stripped_tokens = []
    core_tokens = []

    for token in tokens:
        canonical_suffix = LEGAL_SUFFIX_MAP.get(token, token)
        norm_tokens.append(canonical_suffix)

        if token not in LEGAL_SUFFIX_TOKENS and canonical_suffix not in LEGAL_SUFFIX_TOKENS:
            stripped_tokens.append(token)
            if token not in STOPWORDS:
                core_tokens.append(token)

    return " ".join(norm_tokens), " ".join(stripped_tokens), " ".join(core_tokens)


def extract_pin_code(text: str) -> Optional[str]:
    """Extract 5 or 6 digit PIN / ZIP code using precompiled regex."""
    if not isinstance(text, str) or pd.isna(text):
        return None
    match = PIN_REGEX.search(text)
    return match.group(0) if match else None


def normalize_business_address(
    address: str, country: Optional[str] = None
) -> Tuple[str, str, Optional[str]]:
    """Normalize address expanding abbreviations, extracting pin code and core tokens."""
    cleaned = basic_clean(address)
    if not cleaned:
        return "", "", None

    abbrev_map = dict(ADDRESS_ABBREV_GLOBAL)
    if country and isinstance(country, str):
        country_key = country.strip().lower()
        if country_key in ADDRESS_ABBREV_COUNTRY:
            abbrev_map.update(ADDRESS_ABBREV_COUNTRY[country_key])

    pin_code = extract_pin_code(address)
    tokens = cleaned.split()
    norm_tokens = []
    core_tokens = []

    for token in tokens:
        expanded = abbrev_map.get(token, token)
        norm_tokens.append(expanded)
        if token not in STOPWORDS and expanded not in STOPWORDS and token != pin_code:
            core_tokens.append(expanded)

    return " ".join(norm_tokens), " ".join(core_tokens), pin_code


def normalize_source_df(df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    """Apply high-performance vectorized/fast-tuple text normalization to DataFrame.
    
    Args:
        df: Input DataFrame containing business_name, business_address, and optional country.
        show_progress: Whether to display a tqdm progress bar for large DataFrames.
        
    Returns:
        DataFrame with added normalized columns.
    """
    df = df.copy()
    start_t = time.perf_counter()

    has_name = "business_name" in df.columns
    has_addr = "business_address" in df.columns
    has_country = "country" in df.columns

    b_names = df["business_name"].fillna("").astype(str).tolist() if has_name else [""] * len(df)
    b_addrs = df["business_address"].fillna("").astype(str).tolist() if has_addr else [""] * len(df)
    b_countries = df["country"].fillna("").astype(str).tolist() if has_country else [None] * len(df)

    name_norms = []
    name_stripped_list = []
    name_tokens_list = []
    addr_norms = []
    addr_tokens_list = []
    pin_codes = []

    iterator = zip(b_names, b_addrs, b_countries)
    if show_progress and HAS_TQDM and len(df) > 5000:
        iterator = tqdm(iterator, total=len(df), desc=f"Normalizing {len(df):,} records")

    for b_name, b_addr, country_val in iterator:
        n_norm, n_stripped, n_tokens = normalize_business_name(b_name)
        a_norm, a_tokens, p_code = normalize_business_address(b_addr, country=country_val)

        name_norms.append(n_norm)
        name_stripped_list.append(n_stripped)
        name_tokens_list.append(n_tokens)
        addr_norms.append(a_norm)
        addr_tokens_list.append(a_tokens)
        pin_codes.append(p_code)

    df["name_norm"] = name_norms
    df["name_stripped"] = name_stripped_list
    df["name_tokens"] = name_tokens_list
    df["address_norm"] = addr_norms
    df["address_tokens"] = addr_tokens_list
    df["pin_code"] = pin_codes

    df["clean_name"] = df["name_norm"]
    df["clean_address"] = df["address_norm"]
    if has_country:
        df["clean_country"] = [basic_clean(c) if c else "" for c in b_countries]

    elapsed = time.perf_counter() - start_t
    if show_progress and len(df) > 5000:
        print(f"Normalized {len(df):,} records in {elapsed:.2f}s ({len(df)/elapsed:,.0f} rec/s).")

    return df

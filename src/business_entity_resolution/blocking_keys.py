"""Blocking key generation module.

Implements multi-channel blocking key generation for entity resolution,
including exact names, country-partitioned names, core names (stripped suffixes),
compact representations, token shingles, and address-assisted postal keys.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from business_entity_resolution.address_normalizer import (
    NormalizedBusinessAddress,
)
from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.text_normalizer import NormalizedBusinessName


def generate_blocking_keys(
    norm_name: NormalizedBusinessName,
    norm_addr: Optional[NormalizedBusinessAddress] = None,
    country: Optional[str] = None,
    config: Optional[BlockingConfig] = None,
) -> Dict[str, List[str]]:
    """Generate multi-channel blocking keys for a single business entity record.

    Channels Implemented:
    ---------------------
    - 'exact_name': Clean normalized name across all countries.
    - 'country_exact_name': Country code + normalized name.
    - 'country_core_name': Country code + core name (legal suffixes stripped).
    - 'country_compact': Country code + compact alphanumeric string.
    - 'name_token_shingles': Country code + 2-word combinations of significant name tokens.
    - 'postal_prefix': Country code + extracted postal/PIN code + first 4 chars of name.

    Parameters
    ----------
    norm_name : NormalizedBusinessName
        Normalized business name container.
    norm_addr : Optional[NormalizedBusinessAddress]
        Normalized business address container.
    country : Optional[str]
        Raw or clean country string.
    config : Optional[BlockingConfig]
        Blocking configuration settings.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of channel name to list of generated keys for this record.
    """
    cfg = config or BlockingConfig()
    ctry = (country or "UNKNOWN").upper().strip()
    keys: Dict[str, List[str]] = {}

    if norm_name.is_empty or not norm_name.normalized:
        return keys

    # Channel A: Exact normalized name
    if cfg.use_exact_name:
        keys["exact_name"] = [norm_name.normalized]

    # Channel B: Country + Exact normalized name
    if cfg.use_country_exact_name:
        keys["country_exact_name"] = [f"{ctry}::{norm_name.normalized}"]

    # Channel C: Country + Core name (stripped legal suffixes)
    if cfg.use_country_core_name and norm_name.core_name:
        if norm_name.core_name != norm_name.normalized:
            keys["country_core_name"] = [f"{ctry}::{norm_name.core_name}"]

    # Channel D: Country + Compact alphanumeric name
    if cfg.use_country_compact_name and norm_name.compact:
        keys["country_compact"] = [f"{ctry}::{norm_name.compact}"]

    # Channel E: Name Token Shingles (2-word sorted combinations)
    if cfg.use_name_token_shingles and len(norm_name.tokens) >= 2:
        shingles: List[str] = []
        # Filter tokens with minimum length
        sig_tokens = [
            t
            for t in norm_name.tokens
            if len(t) >= cfg.min_token_length and t not in {"and", "the", "for"}
        ][: cfg.max_tokens_per_name]

        if len(sig_tokens) >= 2:
            # Adjacent 2-grams
            for i in range(len(sig_tokens) - 1):
                pair = sorted([sig_tokens[i], sig_tokens[i + 1]])
                shingles.append(f"{ctry}::shingle::{pair[0]}_{pair[1]}")
            # First and last token shingle (catches long name variants)
            if cfg.use_first_last_shingle and len(sig_tokens) >= 3:
                pair = sorted([sig_tokens[0], sig_tokens[-1]])
                shingles.append(f"{ctry}::shingle::{pair[0]}_{pair[1]}")

            if shingles:
                keys["name_token_shingles"] = list(dict.fromkeys(shingles))

    # Channel F: Postal + Name Prefix
    if cfg.use_postal_prefix and norm_addr and not norm_addr.is_empty:
        if norm_addr.postal_code and len(norm_name.compact) >= 3:
            name_prefix = norm_name.compact[:4]
            keys["postal_prefix"] = [
                f"{ctry}::postal::{norm_addr.postal_code}::{name_prefix}"
            ]

    # Channel G: Address-Assisted (Street Number + Street Stem)
    # Recovers matches where business names are acronyms, DBAs, or transliterated
    if cfg.use_addr_num_stem and norm_addr and not norm_addr.is_empty:
        if norm_addr.street_number and norm_addr.street_stem:
            keys["addr_num_stem"] = [
                f"{ctry}::addr::{norm_addr.street_number}_{norm_addr.street_stem}"
            ]

    # Channel H: Postal Code + Street Number
    if cfg.use_postal_num and norm_addr and not norm_addr.is_empty:
        if norm_addr.postal_code and norm_addr.street_number:
            keys["postal_num"] = [
                f"{ctry}::pnum::{norm_addr.postal_code}_{norm_addr.street_number}"
            ]

    # Channel I: Street Stem + Name Prefix (Phase 5)
    # Recovers matches where street numbers or suffixes diverge, but street name and name prefix match
    if cfg.use_stem_prefix and norm_addr and not norm_addr.is_empty:
        if norm_addr.street_stem and len(norm_name.compact) >= 3:
            stem_key = norm_addr.street_stem[:15]
            keys["stem_prefix"] = [
                f"{ctry}::sp::{stem_key}::{norm_name.compact[:3]}"
            ]

    # Channel J: Script Transliteration for Non-Latin Names (Phase 5)
    if cfg.use_transliteration and norm_name and norm_name.raw:
        if any(ord(c) > 0x024F and c.isalnum() for c in norm_name.raw):
            import text_unidecode
            tl_str = text_unidecode.unidecode(norm_name.raw)
            tl_clean = "".join(c.lower() for c in tl_str if c.isalnum())
            if len(tl_clean) >= 4:
                keys["tl_country_compact"] = [f"{ctry}::tlcp::{tl_clean}"]

    # Channel K: Rare Single Name Token (Phase 5)
    if cfg.use_rare_name_token and norm_name and norm_name.tokens:
        stop_tokens = {
            "pvt", "ltd", "corp", "inc", "llc", "company", "limited", "private",
            "services", "enterprises", "holdings", "group", "solutions", "technologies",
            "international", "national", "global", "center", "centre", "general",
            "medical", "health", "care", "auto", "repair", "store", "shop", "restaurant"
        }
        rare = [t for t in norm_name.tokens if len(t) >= 5 and t not in stop_tokens][:2]
        if rare:
            keys["rare_token"] = [f"{ctry}::rt::{t}" for t in rare]

    # Channel L: Address Numbers
    # Matches on the largest numeric token and the first numeric token extracted from the address.
    if getattr(cfg, "use_addr_numbers", False) and norm_addr and not norm_addr.is_empty:
        import re
        nums = [int(x) for x in re.findall(r'\d+', norm_addr.raw)]
        if nums:
            keys["addr_max_num"] = [f"{ctry}::amn::{max(nums)}"]
            keys["addr_first_num"] = [f"{ctry}::afn::{nums[0]}"]

    # Channel M: Name without Spaces
    if getattr(cfg, "use_name_no_space", False) and norm_name and norm_name.compact:
        # Use first 8 chars of compact name, which has no spaces or punctuation
        if len(norm_name.compact) >= 8:
            keys["name_no_space"] = [f"{ctry}::nns::{norm_name.compact[:8]}"]

    # Channel N: Longest Address Word
    # Matches on the longest alphabetical token in the address (>=7 chars)
    if getattr(cfg, "use_addr_alpha", True) and norm_addr and not norm_addr.is_empty:
        import re
        words = [w for w in re.findall(r'[a-z]+', norm_addr.raw) if len(w) >= 7]
        if words:
            longest_word = max(words, key=len)
            keys["addr_longest_word"] = [f"{ctry}::alw::{longest_word}"]

    # Channel O: Consonant Skeleton
    # Strips vowels and takes first 8 chars. Highly robust to typos and phonetic variations.
    if getattr(cfg, "use_consonant_skeleton", True) and norm_name and norm_name.compact:
        vowels = set("aeiouy")
        skeleton = "".join(c for c in norm_name.compact if c not in vowels)
        if len(skeleton) >= 4:
            keys["consonant_skeleton"] = [f"{ctry}::cs::{skeleton[:8]}"]

    # Channel P: Address Word (>=5) + Address Number
    # Highly robust for cross-lingual matches where translated names fail but addresses share a city and a number.
    if getattr(cfg, "use_addr_word_num", True) and norm_addr and not norm_addr.is_empty:
        import re
        words = set([w for w in re.findall(r'[a-z]+', norm_addr.raw) if len(w) >= 5])
        nums = set(re.findall(r'\d+', norm_addr.raw))
        if words and nums:
            cross_keys = []
            for w in words:
                for n in nums:
                    cross_keys.append(f"{ctry}::awn::{w}_{n}")
            if cross_keys:
                keys["addr_word_num"] = cross_keys

    return keys

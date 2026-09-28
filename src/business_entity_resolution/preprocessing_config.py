"""Configuration dataclasses and defaults for data preprocessing, normalization,
and candidate generation blocking strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class TextNormalizerConfig:
    """Configuration for business name normalization."""

    unicode_form: str = "NFKC"
    lowercase: bool = True
    normalize_whitespace: bool = True
    remove_punctuation: bool = True
    normalize_legal_suffixes: bool = True
    generate_compact: bool = True
    generate_core_name: bool = True
    strip_diacritics: bool = True
    # Standard legal entity suffixes across US, India, and France
    legal_suffixes: Set[str] = field(
        default_factory=lambda: {
            # US / International
            "inc",
            "incorporated",
            "corp",
            "corporation",
            "llc",
            "l.l.c.",
            "llp",
            "l.l.p.",
            "co",
            "company",
            "ltd",
            "limited",
            "holdings",
            "group",
            "enterprise",
            "enterprises",
            "services",
            "solutions",
            "technologies",
            # India specific
            "pvt",
            "private",
            "pvt ltd",
            "private limited",
            "pltd",
            # France specific
            "sarl",
            "s.a.r.l.",
            "sa",
            "s.a.",
            "sas",
            "s.a.s.",
            "sasu",
            "eurl",
            "snc",
            "sci",
        }
    )


@dataclass
class AddressNormalizerConfig:
    """Configuration for business address normalization."""

    unicode_form: str = "NFKC"
    lowercase: bool = True
    normalize_whitespace: bool = True
    normalize_street_abbreviations: bool = True
    extract_postal_code: bool = True
    missing_address_placeholder: str = ""


@dataclass
class BlockingConfig:
    """Configuration for multi-channel candidate generation blocking."""

    # Channels to activate
    use_exact_name: bool = True
    use_country_exact_name: bool = True
    use_country_core_name: bool = True
    use_country_compact_name: bool = True
    use_name_token_shingles: bool = True
    use_first_last_shingle: bool = True
    use_postal_prefix: bool = True
    use_addr_num_stem: bool = True
    use_postal_num: bool = True

    # Phase 5 Blocking Enhancements (default False to preserve baseline behavior)
    use_stem_prefix: bool = True
    use_transliteration: bool = True
    use_rare_name_token: bool = True
    use_addr_numbers: bool = True
    use_name_no_space: bool = True
    use_addr_alpha: bool = True
    use_consonant_skeleton: bool = True

    # Shingle & token tuning
    min_token_length: int = 3
    shingle_size: int = 2
    max_tokens_per_name: int = 6

    # Frequency cap to prevent explosive Cartesian products on ultra-generic keys
    max_key_frequency: int = 1500

    # Maximum candidate pairs per Source 1 entity (safety cap for downstream matching)
    max_candidates_per_entity: Optional[int] = 100

    # Channel priority weights for candidate ranking when capping
    channel_weights: Dict[str, int] = field(
        default_factory=lambda: {
            "country_exact_name": 12,
            "exact_name": 12,
            "country_core_name": 10,
            "country_compact": 8,
            "tl_country_compact": 8,
            "name_no_space": 7,
            "addr_num_stem": 7,
            "postal_num": 6,
            "stem_prefix": 6,
            "postal_prefix": 5,
            "addr_max_num": 5,
            "addr_first_num": 5,
            "name_token_shingles": 4,
            "addr_longest_word": 4,
            "rare_token": 3,
        }
    )


@dataclass
class DevSplitConfig:
    """Configuration for reproducible development sampling and evaluation."""

    random_seed: int = 42
    sample_size: Optional[int] = 50_000  # Default evaluation sample for rapid benchmarking
    stratify_by_country: bool = True

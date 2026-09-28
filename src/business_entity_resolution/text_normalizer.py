"""Business name normalization module.

Implements conservative, configurable, and non-destructive normalization for
multi-source business entities, retaining raw data and producing multiple
representations (clean, core, compact, and tokens).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Set

from business_entity_resolution.preprocessing_config import TextNormalizerConfig


@dataclass
class NormalizedBusinessName:
    """Container holding raw and multi-level normalized representations of a business name."""

    raw: str
    normalized: str
    core_name: str
    compact: str
    tokens: List[str] = field(default_factory=list)
    is_empty: bool = False

    def __post_init__(self):
        if self.tokens is None:
            self.tokens = []


class BusinessNameNormalizer:
    """Normalizes business names into clean, standardized representations while

    preserving the raw input.
    """

    def __init__(self, config: Optional[TextNormalizerConfig] = None):
        self.config = config or TextNormalizerConfig()
        # Compile regex patterns
        # Whitespace collapsing
        self._ws_re = re.compile(r"\s+")
        # Replace non-alphanumeric (except basic word characters and ampersand)
        # Note: replace '&' with ' and ' or keep safe
        self._ampersand_re = re.compile(r"&")
        # Punctuation to space
        self._punct_re = re.compile(r"[^\w\s]")
        # Alphanumeric only (for compact)
        self._alphanumeric_re = re.compile(r"[^\w]")

        # Sort suffixes by length descending so multi-word suffixes (e.g., 'private limited') match first
        sorted_suffixes = sorted(
            self.config.legal_suffixes, key=lambda s: len(s), reverse=True
        )
        escaped_suffixes = [re.escape(s) for s in sorted_suffixes]
        # Match legal suffix at end of string preceded by word boundary or space
        if escaped_suffixes:
            self._legal_suffix_re = re.compile(
                r"(?:\b|\s)(?:" + "|".join(escaped_suffixes) + r")\b\.?$",
                re.IGNORECASE,
            )
        else:
            self._legal_suffix_re = None

    def normalize(self, name: Optional[str]) -> NormalizedBusinessName:
        """Perform multi-level normalization on a single business name string.

        Parameters
        ----------
        name : Optional[str]
            Raw business name.

        Returns
        -------
        NormalizedBusinessName
            Structured object with raw and normalized fields.
        """
        raw_val = "" if name is None else str(name)
        if not raw_val or raw_val.strip() == "" or raw_val.strip().lower() in {"nan", "null", "none"}:
            return NormalizedBusinessName(
                raw=raw_val,
                normalized="",
                core_name="",
                compact="",
                tokens=[],
                is_empty=True,
            )

        # 1. Unicode normalization & diacritics stripping
        text = unicodedata.normalize(self.config.unicode_form, raw_val)
        if self.config.strip_diacritics:
            text = "".join(
                c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
            )

        # 2. Casing
        if self.config.lowercase:
            text = text.lower()

        # 3. Replace '&' with ' and ' for consistency
        text = self._ampersand_re.sub(" and ", text)

        # 4. Remove leading/trailing non-alphanumerics like dashes, dots, quotes
        text = self._punct_re.sub(" ", text)

        # 5. Whitespace normalization
        if self.config.normalize_whitespace:
            text = self._ws_re.sub(" ", text).strip()

        # 6. Generate Core Name (strip trailing legal suffixes if enabled)
        core_name = text
        if self.config.generate_core_name and self._legal_suffix_re:
            stripped = self._legal_suffix_re.sub("", core_name).strip()
            # If removing the suffix empties the name (e.g. name was literally just 'Inc'), keep text
            if stripped:
                core_name = stripped

        # 7. Generate Compact (alphanumeric only, no spaces)
        compact = ""
        if self.config.generate_compact:
            compact = self._alphanumeric_re.sub("", text)

        # 8. Tokenize
        tokens = [t for t in text.split(" ") if t]

        return NormalizedBusinessName(
            raw=raw_val,
            normalized=text,
            core_name=core_name,
            compact=compact,
            tokens=tokens,
            is_empty=False,
        )

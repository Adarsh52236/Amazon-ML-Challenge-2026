"""Business address normalization module.

Implements conservative, open-set address normalization, handling missing
addresses, standardizing whitespace and punctuation, and safely extracting
country-aware postal codes without external dependencies.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Set

from business_entity_resolution.preprocessing_config import AddressNormalizerConfig


@dataclass
class NormalizedBusinessAddress:
    """Container holding raw and normalized representations of a business address."""

    raw: str
    normalized: str
    postal_code: str = ""
    street_number: str = ""
    street_stem: str = ""
    tokens: List[str] = field(default_factory=list)
    is_empty: bool = False

    def __post_init__(self):
        if self.tokens is None:
            self.tokens = []


class BusinessAddressNormalizer:
    """Normalizes business addresses, extracts postal codes, and handles missing

    address indicators while preserving the raw input.
    """

    def __init__(self, config: Optional[AddressNormalizerConfig] = None):
        self.config = config or AddressNormalizerConfig()
        self._ws_re = re.compile(r"\s+")
        self._punct_re = re.compile(r"[^\w\s]")

        # Postal code / PIN code patterns:
        # India: 6 digits (optionally with space, e.g. 560001, 110 001)
        self._india_pin_re = re.compile(r"\b([1-9][0-9]{2}\s?[0-9]{3})\b")
        # US: 5 digits (optionally 5+4, e.g. 90210, 90210-1234)
        self._us_zip_re = re.compile(r"\b([0-9]{5})(?:-[0-9]{4})?\b")
        # France: 5 digits starting with 01-98 or 2A/2B (e.g. 75008, 13001)
        self._france_postal_re = re.compile(r"\b(0[1-9]|[1-8][0-9]|9[0-8])[0-9]{3}\b")

        # Standard street abbreviation normalizations (conservative)
        self._street_abbrev_map = {
            "street": "st",
            "road": "rd",
            "avenue": "ave",
            "boulevard": "blvd",
            "drive": "dr",
            "lane": "ln",
            "highway": "hwy",
            "parkway": "pkwy",
            "suite": "ste",
            "apartment": "apt",
            "building": "bldg",
            "floor": "fl",
            "nagar": "ngr",
            "marg": "mrg",
            "rue": "r",
        }

    def extract_postal_code(self, text: str, country: Optional[str] = None) -> str:
        """Extract candidate postal code based on country or general pattern matching."""
        if not text:
            return ""

        ctry = (country or "").upper().strip()
        if ctry == "INDIA":
            m = self._india_pin_re.search(text)
            if m:
                return m.group(0).replace(" ", "")
        elif ctry == "US":
            m = self._us_zip_re.search(text)
            if m:
                # Return only the 5 digit zip
                return m.group(1)
        elif ctry == "FRANCE":
            m = self._france_postal_re.search(text)
            if m:
                return m.group(0)

        # Fallback open-set check
        m = self._india_pin_re.search(text)
        if m:
            return m.group(0).replace(" ", "")
        m = self._us_zip_re.search(text)
        if m:
            return m.group(1)

        return ""

    def normalize(
        self, address: Optional[str], country: Optional[str] = None
    ) -> NormalizedBusinessAddress:
        """Perform conservative normalization on a business address string.

        Parameters
        ----------
        address : Optional[str]
            Raw business address text.
        country : Optional[str]
            Associated country label for postal code extraction.

        Returns
        -------
        NormalizedBusinessAddress
            Structured container with raw and normalized properties.
        """
        raw_val = "" if address is None else str(address)
        if not raw_val or raw_val.strip() == "" or raw_val.strip().lower() in {"nan", "null", "none"}:
            return NormalizedBusinessAddress(
                raw=raw_val,
                normalized="",
                postal_code="",
                tokens=[],
                is_empty=True,
            )

        # 1. Unicode normalization
        text = unicodedata.normalize(self.config.unicode_form, raw_val)

        # 2. Extract postal code before punctuation removal
        postal_code = ""
        if self.config.extract_postal_code:
            postal_code = self.extract_postal_code(text, country)

        # 3. Extract street / building number (first numeric token 1-6 digits)
        m_num = re.search(r"\b([0-9]{1,6})\b", text)
        street_number = m_num.group(1) if m_num else ""

        # 4. Casing
        if self.config.lowercase:
            text = text.lower()

        # 5. Remove punctuation
        text = self._punct_re.sub(" ", text)

        # 6. Collapse whitespace
        if self.config.normalize_whitespace:
            text = self._ws_re.sub(" ", text).strip()

        # 7. Normalize conservative street abbreviations and extract street stem
        tokens = text.split(" ")
        normalized_tokens: List[str] = []
        street_stem = ""
        skip_stem_words = {
            "no", "plot", "house", "flat", "door", "building", "floor", "unit",
            "suite", "ste", "sector", "road", "rd", "street", "st", "avenue",
            "ave", "lane", "ln", "drive", "dr", "highway", "hwy", "blvd",
            "near", "opp", "opposite", "behind", "beside", "null"
        }

        for t in tokens:
            if not t:
                continue
            if self.config.normalize_street_abbreviations and t in self._street_abbrev_map:
                norm_t = self._street_abbrev_map[t]
                normalized_tokens.append(norm_t)
            else:
                norm_t = t
                normalized_tokens.append(norm_t)

            if not street_stem and len(t) >= 4 and t not in skip_stem_words and not t.isdigit():
                street_stem = t

        normalized_text = " ".join(normalized_tokens)

        return NormalizedBusinessAddress(
            raw=raw_val,
            normalized=normalized_text,
            postal_code=postal_code,
            street_number=street_number,
            street_stem=street_stem,
            tokens=normalized_tokens,
            is_empty=False,
        )

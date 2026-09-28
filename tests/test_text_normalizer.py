"""Unit tests for business name normalization."""

import pytest
from business_entity_resolution.text_normalizer import (
    BusinessNameNormalizer,
    NormalizedBusinessName,
)
from business_entity_resolution.preprocessing_config import TextNormalizerConfig


def test_mixed_casing_and_whitespace():
    normalizer = BusinessNameNormalizer()
    res = normalizer.normalize("   AcMe   CORPorAtioN   ")
    assert res.normalized == "acme corporation"
    assert res.core_name == "acme"
    assert res.compact == "acmecorporation"
    assert res.raw == "   AcMe   CORPorAtioN   "
    assert not res.is_empty


def test_punctuation_and_ampersand():
    normalizer = BusinessNameNormalizer()
    res = normalizer.normalize("Barnes & Noble, Inc.")
    assert "and" in res.normalized
    assert res.core_name == "barnes and noble"
    assert "inc" not in res.core_name


def test_unicode_accents():
    normalizer = BusinessNameNormalizer()
    res = normalizer.normalize("Café Société Électrique S.A.")
    assert "café" in res.normalized or "cafe" in res.normalized
    assert res.raw == "Café Société Électrique S.A."


def test_empty_and_null_values():
    normalizer = BusinessNameNormalizer()
    empty_res = normalizer.normalize("")
    assert empty_res.is_empty
    assert empty_res.normalized == ""

    none_res = normalizer.normalize(None)
    assert none_res.is_empty
    assert none_res.normalized == ""

    ws_res = normalizer.normalize("   \t  ")
    assert ws_res.is_empty
    assert ws_res.normalized == ""


def test_legal_suffixes_various():
    normalizer = BusinessNameNormalizer()
    # US / International
    assert normalizer.normalize("Tesla Motors, LLC").core_name == "tesla motors"
    assert normalizer.normalize("Apple Computer Inc.").core_name == "apple computer"
    # India
    assert normalizer.normalize("Tata Consultancy Services Private Limited").core_name == "tata consultancy services"
    assert normalizer.normalize("Infosys Pvt Ltd").core_name == "infosys"
    # France
    assert normalizer.normalize("Renault SAS").core_name == "renault"
    assert normalizer.normalize("TotalEnergies SE SARL").core_name == "totalenergies se"


def test_names_with_numbers_and_hyphens():
    normalizer = BusinessNameNormalizer()
    res = normalizer.normalize("7-Eleven Stores, Inc.")
    assert "7" in res.normalized
    assert "eleven" in res.normalized
    assert res.compact == "7elevenstoresinc"
    assert res.core_name == "7 eleven stores"


def test_short_names_not_wiped():
    normalizer = BusinessNameNormalizer()
    # If the company is literally named "Inc" or "LLC"
    res = normalizer.normalize("Inc")
    assert res.core_name == "inc"
    assert res.normalized == "inc"


def test_transliteration_and_diacritics_edge_cases():
    normalizer = BusinessNameNormalizer()
    # Accents stripped: Pólytechnic -> polytechnic
    res1 = normalizer.normalize("Shri Kripa Pólytechnic")
    assert "polytechnic" in res1.tokens

    # Spanish / Scandinavian diacritics
    res2 = normalizer.normalize("Mölnlycke Health Care AB")
    assert "molnlycke" in res2.tokens


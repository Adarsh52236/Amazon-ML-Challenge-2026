"""Unit tests for business address normalization."""

import pytest
from business_entity_resolution.address_normalizer import (
    BusinessAddressNormalizer,
    NormalizedBusinessAddress,
)


def test_empty_and_null_address():
    normalizer = BusinessAddressNormalizer()
    res_empty = normalizer.normalize("")
    assert res_empty.is_empty
    assert res_empty.normalized == ""
    assert res_empty.postal_code == ""

    res_none = normalizer.normalize(None)
    assert res_none.is_empty
    assert res_none.normalized == ""

    res_ws = normalizer.normalize("   \n\t  ")
    assert res_ws.is_empty


def test_us_address_zip_extraction():
    normalizer = BusinessAddressNormalizer()
    addr = "123 Main Street, Suite 400, Springfield, IL 62701-1234"
    res = normalizer.normalize(addr, country="US")
    assert not res.is_empty
    assert res.postal_code == "62701"
    assert "st" in res.tokens
    assert "ste" in res.tokens
    assert res.raw == addr


def test_india_address_pin_extraction():
    normalizer = BusinessAddressNormalizer()
    addr = "Plot No. 45, MG Road, Near SBI ATM, Bangalore, Karnataka 560001"
    res = normalizer.normalize(addr, country="India")
    assert not res.is_empty
    assert res.postal_code == "560001"
    assert "rd" in res.tokens
    assert "bangalore" in res.tokens


def test_french_address_postal_extraction():
    normalizer = BusinessAddressNormalizer()
    addr = "63 Rue de Dieppe, 59000 Lille"
    res = normalizer.normalize(addr, country="France")
    assert not res.is_empty
    assert res.postal_code == "59000"
    assert "lille" in res.tokens


def test_street_abbreviations():
    normalizer = BusinessAddressNormalizer()
    addr = "North Avenue and South Highway, Floor 3"
    res = normalizer.normalize(addr)
    assert "ave" in res.tokens
    assert "hwy" in res.tokens
    assert "fl" in res.tokens


def test_numbers_and_special_chars_preserved():
    normalizer = BusinessAddressNormalizer()
    addr = "Building #4, Sector-18, PIN: 122001"
    res = normalizer.normalize(addr, country="India")
    assert "4" in res.tokens
    assert "18" in res.tokens
    assert res.postal_code == "122001"


def test_street_number_and_stem_extraction():
    normalizer = BusinessAddressNormalizer()
    res_us = normalizer.normalize("304 Stonehouse Road, James City County, VA", country="US")
    assert res_us.street_number == "304"
    assert res_us.street_stem == "stonehouse"

    res_in = normalizer.normalize("Plot No. 45, MG Road, Bangalore", country="India")
    assert res_in.street_number == "45"
    assert res_in.street_stem == "bangalore"

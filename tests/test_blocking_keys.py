"""Unit tests for blocking keys generation."""

import pytest
from business_entity_resolution.text_normalizer import BusinessNameNormalizer
from business_entity_resolution.address_normalizer import BusinessAddressNormalizer
from business_entity_resolution.blocking_keys import generate_blocking_keys
from business_entity_resolution.preprocessing_config import BlockingConfig


def test_blocking_keys_channels():
    name_norm = BusinessNameNormalizer()
    addr_norm = BusinessAddressNormalizer()

    n = name_norm.normalize("Apex Data Systems, LLC")
    a = addr_norm.normalize("100 Market St, San Francisco, CA 94105", country="US")

    keys = generate_blocking_keys(n, a, country="US")

    assert "exact_name" in keys
    assert keys["exact_name"] == ["apex data systems llc"]

    assert "country_exact_name" in keys
    assert keys["country_exact_name"] == ["US::apex data systems llc"]

    assert "country_core_name" in keys
    assert keys["country_core_name"] == ["US::apex data systems"]

    assert "country_compact" in keys
    assert keys["country_compact"] == ["US::apexdatasystemsllc"]

    assert "name_token_shingles" in keys
    # Tokens: apex, data, systems, llc
    assert any("apex_data" in s or "data_systems" in s for s in keys["name_token_shingles"])

    assert "postal_prefix" in keys
    assert keys["postal_prefix"] == ["US::postal::94105::apex"]

    assert "addr_num_stem" in keys
    assert keys["addr_num_stem"] == ["US::addr::100_market"]

    assert "postal_num" in keys
    assert keys["postal_num"] == ["US::pnum::94105_100"]


def test_missing_address_fallback():
    name_norm = BusinessNameNormalizer()
    addr_norm = BusinessAddressNormalizer()

    n = name_norm.normalize("Omega Diagnostics Ltd")
    # Empty address (simulating S2/S3 missing address records)
    a = addr_norm.normalize("", country="India")

    keys = generate_blocking_keys(n, a, country="India")

    # Name-based channels should all be present
    assert "country_exact_name" in keys
    assert "country_core_name" in keys
    assert "country_compact" in keys
    assert "name_token_shingles" in keys
    # Postal channel should gracefully be omitted without errors
    assert "postal_prefix" not in keys

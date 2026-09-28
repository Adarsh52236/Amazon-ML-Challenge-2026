#!/usr/bin/env python3
"""CLI Script to demonstrate and test business name and address normalization."""

import argparse
import sys
from pathlib import Path

# Add src to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.address_normalizer import BusinessAddressNormalizer
from business_entity_resolution.config import configure_console_encoding
from business_entity_resolution.data_loader import load_source_file
from business_entity_resolution.text_normalizer import BusinessNameNormalizer


def main():
    configure_console_encoding()
    parser = argparse.ArgumentParser(
        description="Preprocess and inspect normalized business records."
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "test"]
    )
    parser.add_argument(
        "--source", type=int, default=1, choices=[1, 2, 3]
    )
    parser.add_argument(
        "--nrows", type=int, default=10, help="Number of sample records to inspect."
    )
    args = parser.parse_args()

    print(f"Loading {args.nrows} sample records from {args.split.upper()} Source {args.source}...")
    df = load_source_file(split=args.split, source_num=args.source, nrows=args.nrows)

    name_norm = BusinessNameNormalizer()
    addr_norm = BusinessAddressNormalizer()

    print("\n" + "=" * 90)
    print(f"NORMALIZATION INSPECTION — {args.split.upper()} SOURCE {args.source}")
    print("=" * 90)

    for i, row in df.iterrows():
        eid = row["entity_id"]
        raw_name = row.get("business_name", "")
        raw_addr = row.get("business_address", "")
        country = row.get("country", "")

        n_name = name_norm.normalize(raw_name)
        n_addr = addr_norm.normalize(raw_addr, country)

        print(f"\n[Record #{i+1}] ID: {eid} | Country: {country}")
        print(f"  Raw Name:        {raw_name!r}")
        print(f"  Normalized Name: {n_name.normalized!r}")
        print(f"  Core Name:       {n_name.core_name!r}")
        print(f"  Compact Name:    {n_name.compact!r}")
        print(f"  Raw Address:     {raw_addr!r}")
        print(f"  Norm Address:    {n_addr.normalized!r}")
        print(f"  Postal / PIN:    {n_addr.postal_code!r} (missing: {n_addr.is_empty})")

    print("\n" + "=" * 90)
    print("Inspection complete. Raw dataset files remain 100% unmodified.")
    print("=" * 90)


if __name__ == "__main__":
    main()

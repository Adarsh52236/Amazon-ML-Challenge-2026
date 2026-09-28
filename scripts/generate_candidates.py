#!/usr/bin/env python3
"""CLI Script to generate candidate pairs for a dataset split and export them in

official format or internal tabular format.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.candidate_generator import (
    CandidateGenerator,
    CandidatePair,
)
from business_entity_resolution.config import (
    configure_console_encoding,
    get_source_path,
)
from business_entity_resolution.data_loader import (
    iter_source_records,
    load_source_file,
)
from business_entity_resolution.preprocessing_config import BlockingConfig


def main():
    configure_console_encoding()
    parser = argparse.ArgumentParser(
        description="Generate candidate pairs for business entity resolution."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to generate candidates for.",
    )
    parser.add_argument(
        "--s1-limit",
        type=int,
        default=None,
        help="Limit number of Source 1 records (for fast test/dev runs).",
    )
    parser.add_argument(
        "--target-limit",
        type=int,
        default=None,
        help="Limit number of secondary source records to index.",
    )
    parser.add_argument(
        "--output-official",
        type=str,
        default="output/candidate_pairs.tsv",
        help="Path to export official format candidate_pairs.tsv.",
    )
    parser.add_argument(
        "--output-tabular",
        type=str,
        default=None,
        help="Optional path to export detailed tabular candidate pairs with channel provenance.",
    )

    args = parser.parse_args()
    print("=" * 80)
    print(f"CANDIDATE GENERATION PIPELINE — {args.split.upper()} SPLIT")
    print("=" * 80)

    # 1. Load Source 1 records
    print(f"\n[1/3] Loading Source 1 {args.split} records...")
    t0 = time.time()
    s1_df = load_source_file(split=args.split, source_num=1, nrows=args.s1_limit)
    print(f"Loaded {len(s1_df):,} Source 1 records in {time.time() - t0:.2f}s")

    # 2. Build target index from Source 2 and Source 3
    print(f"\n[2/3] Building target index from Source 2 and Source 3...")
    t0 = time.time()
    generator = CandidateGenerator(config=BlockingConfig())

    s2_path = get_source_path(args.split, 2)
    s3_path = get_source_path(args.split, 3)

    count_s2 = 0
    for r in iter_source_records(s2_path):
        if args.target_limit and count_s2 >= args.target_limit:
            break
        generator.index_target_record(
            entity_id=r["entity_id"],
            name=r.get("business_name", ""),
            address=r.get("business_address", ""),
            country=r.get("country", ""),
        )
        count_s2 += 1

    count_s3 = 0
    for r in iter_source_records(s3_path):
        if args.target_limit and count_s3 >= args.target_limit:
            break
        generator.index_target_record(
            entity_id=r["entity_id"],
            name=r.get("business_name", ""),
            address=r.get("business_address", ""),
            country=r.get("country", ""),
        )
        count_s3 += 1

    print(
        f"Indexed {generator.indexed_target_count:,} target records (S2: {count_s2:,}, S3: {count_s3:,}) in {time.time() - t0:.2f}s"
    )

    pruned = generator.prune_high_frequency_keys()
    print(f"Pruned high-frequency keys: {pruned}")

    # 3. Query and generate candidate pairs
    print(f"\n[3/3] Generating candidate pairs...")
    t0 = time.time()
    candidates = generator.generate_candidates_for_dataframe(s1_df)
    elapsed = time.time() - t0
    print(
        f"Generated {len(candidates):,} candidate pairs in {elapsed:.2f}s ({len(candidates)/max(1, elapsed):.1f} pairs/sec)"
    )

    # 4. Export official candidate_pairs.tsv
    s1_ids = list(s1_df["entity_id"])
    out_file = generator.export_to_official_candidate_pairs(
        candidates, s1_ids, args.output_official
    )
    print(f"\n[SUCCESS] Exported official candidate pairs file: '{out_file}'")

    if args.output_tabular:
        tab_df = generator.candidates_to_dataframe(candidates)
        tab_path = Path(args.output_tabular).resolve()
        tab_path.parent.mkdir(parents=True, exist_ok=True)
        tab_df.to_csv(tab_path, sep="\t", index=False)
        print(f"[SUCCESS] Exported detailed tabular candidates: '{tab_path}'")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI Script to evaluate candidate generation recall and reduction ratios on a

deterministic development split against training ground truth.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

# Add src to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from business_entity_resolution.candidate_evaluator import (
    CandidateEvaluationReport,
    evaluate_candidates,
)
from business_entity_resolution.candidate_generator import (
    CandidateGenerator,
    CandidatePair,
)
from business_entity_resolution.config import (
    configure_console_encoding,
    get_dataset_root,
    get_ground_truth_path,
    get_source_path,
)
from business_entity_resolution.data_loader import (
    iter_source_records,
    load_ground_truth,
    load_source_file,
)
from business_entity_resolution.preprocessing_config import (
    BlockingConfig,
    DevSplitConfig,
)
from business_entity_resolution.utils import parse_matched_ids


def run_evaluation(
    dev_sample_size: int = 10_000,
    random_seed: int = 42,
    target_scan_limit: int = 500_000,
    incremental_channels: bool = True,
    output_report_path: str = "output/candidate_recall_baseline.md",
) -> CandidateEvaluationReport:
    configure_console_encoding()
    print("=" * 80)
    print("BUSINESS ENTITY RESOLUTION — CANDIDATE GENERATION BASELINE EVALUATION")
    print("=" * 80)

    # 1. Load Ground Truth mapping
    print("\n[Step 1/5] Loading training ground truth...")
    t0 = time.time()
    gt_mapping: Dict[str, List[str]] = load_ground_truth(as_dict=True)
    print(
        f"Loaded {len(gt_mapping):,} S1 ground truth records in {time.time() - t0:.2f}s"
    )

    # 2. Select Deterministic Development Split of Source 1
    print(
        f"\n[Step 2/5] Creating deterministic dev split (seed={random_seed}, sample_size={dev_sample_size:,})..."
    )
    # Read S1 records
    s1_all_df = load_source_file(split="train", source_num=1)
    # Filter to entities with at least 1 match for precise recall benchmarking (or full sample)
    # We sample deterministically across countries
    rng = np.random.default_rng(random_seed)
    sampled_indices = rng.choice(
        len(s1_all_df),
        size=min(dev_sample_size, len(s1_all_df)),
        replace=False,
    )
    s1_dev_df = s1_all_df.iloc[sampled_indices].copy().reset_index(drop=True)
    dev_s1_ids = set(s1_dev_df["entity_id"])

    dev_gt = {
        s1_id: gt_mapping.get(s1_id, [])
        for s1_id in dev_s1_ids
        if s1_id in gt_mapping
    }
    dev_gt_pair_count = sum(len(m) for m in dev_gt.values())
    dev_singletons = sum(1 for m in dev_gt.values() if len(m) == 0)

    s1_metadata = {
        row["entity_id"]: {
            "country": str(row.get("country", "")),
            "name": str(row.get("business_name", "")),
            "address": str(row.get("business_address", "")),
        }
        for _, row in s1_dev_df.iterrows()
    }

    print(f"Dev split: {len(s1_dev_df):,} S1 entities")
    print(f"  - Ground truth positive pairs: {dev_gt_pair_count:,}")
    print(f"  - Singletons (0 true matches): {dev_singletons:,}")

    # 3. Streamingly Index Target Records from Source 2 and Source 3
    # For fast, memory-safe execution, we can index a representative target window or full target stream.
    # To measure authentic recall, we must ensure true matches for our dev split are indexed!
    # So we collect the set of all true target IDs needed by our dev split, plus a background corpus of N records.
    target_ids_needed = set()
    for targets in dev_gt.values():
        target_ids_needed.update(targets)

    print(
        f"\n[Step 3/5] Indexing target records (needed true matches: {len(target_ids_needed):,}, background limit: {target_scan_limit:,})..."
    )
    t0 = time.time()
    generator = CandidateGenerator(config=BlockingConfig())

    s2_path = get_source_path("train", 2)
    s3_path = get_source_path("train", 3)

    indexed_s2 = 0
    indexed_s3 = 0
    target_metadata: Dict[str, Dict[str, Any]] = {}

    # Index Source 2
    for r in iter_source_records(s2_path):
        eid = r["entity_id"]
        # Index if needed by dev ground truth OR within background limit
        if eid in target_ids_needed or indexed_s2 < target_scan_limit:
            generator.index_target_record(
                entity_id=eid,
                name=r.get("business_name", ""),
                address=r.get("business_address", ""),
                country=r.get("country", ""),
            )
            target_metadata[eid] = {
                "is_address_empty": (r.get("business_address", "").strip() == "")
            }
            indexed_s2 += 1

    # Index Source 3
    for r in iter_source_records(s3_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s3 < target_scan_limit:
            generator.index_target_record(
                entity_id=eid,
                name=r.get("business_name", ""),
                address=r.get("business_address", ""),
                country=r.get("country", ""),
            )
            target_metadata[eid] = {
                "is_address_empty": (r.get("business_address", "").strip() == "")
            }
            indexed_s3 += 1

    print(
        f"Indexed {generator.indexed_target_count:,} target records (S2: {indexed_s2:,}, S3: {indexed_s3:,}) in {time.time() - t0:.2f}s"
    )

    # Prune extreme high-frequency keys
    pruned = generator.prune_high_frequency_keys()
    print(f"Pruned high-frequency keys: {pruned}")

    # 4. Generate Candidate Pairs for Dev Source 1 Records
    print(
        f"\n[Step 4/5] Generating candidate pairs across multi-channel blocking..."
    )
    t0 = time.time()
    candidates: List[CandidatePair] = (
        generator.generate_candidates_for_dataframe(s1_dev_df)
    )
    print(
        f"Generated {len(candidates):,} candidate pairs in {time.time() - t0:.2f}s"
    )

    # 5. Evaluate Candidate Recall & Metrics
    print(f"\n[Step 5/5] Evaluating recall ceiling against ground truth...")
    report = evaluate_candidates(
        candidates=candidates,
        ground_truth=dev_gt,
        s1_metadata=s1_metadata,
        target_metadata=target_metadata,
        total_target_records=generator.indexed_target_count,
    )

    # Optional incremental channel breakdown
    channel_steps = []
    if incremental_channels:
        print("\n--- INCREMENTAL CHANNEL RECALL PROGRESSION ---")
        channel_configs = [
            ("Channel A: Exact Name", BlockingConfig(use_country_exact_name=False, use_country_core_name=False, use_country_compact_name=False, use_name_token_shingles=False, use_first_last_shingle=False, use_postal_prefix=False, use_addr_num_stem=False, use_postal_num=False)),
            ("+ Channel B: Country + Name", BlockingConfig(use_country_core_name=False, use_country_compact_name=False, use_name_token_shingles=False, use_first_last_shingle=False, use_postal_prefix=False, use_addr_num_stem=False, use_postal_num=False)),
            ("+ Channel C: Core Name (Suffix Stripped)", BlockingConfig(use_country_compact_name=False, use_name_token_shingles=False, use_first_last_shingle=False, use_postal_prefix=False, use_addr_num_stem=False, use_postal_num=False)),
            ("+ Channel D: Compact Name", BlockingConfig(use_name_token_shingles=False, use_first_last_shingle=False, use_postal_prefix=False, use_addr_num_stem=False, use_postal_num=False)),
            ("+ Channel E: Adjacent Token Shingles", BlockingConfig(use_first_last_shingle=False, use_postal_prefix=False, use_addr_num_stem=False, use_postal_num=False)),
            ("+ Channel F: First-Last Shingles", BlockingConfig(use_postal_prefix=False, use_addr_num_stem=False, use_postal_num=False)),
            ("+ Channel G: Postal Prefix", BlockingConfig(use_addr_num_stem=False, use_postal_num=False)),
            ("+ Channel H: Address Street Number + Stem", BlockingConfig(use_postal_num=False)),
            ("+ Channel I: Postal Code + Street Number (Full)", BlockingConfig()),
        ]

        for label, cfg in channel_configs:
            sub_gen = CandidateGenerator(config=cfg)
            # Share the indexed target keys for enabled channels
            for ch in cfg.__dict__:
                if ch.startswith("use_") and getattr(cfg, ch):
                    ch_name = ch[4:]
                    # Map to internal channel key
                    sub_gen.index[ch_name] = generator.index.get(ch_name, {})
            sub_cands = sub_gen.generate_candidates_for_dataframe(s1_dev_df)
            sub_rep = evaluate_candidates(
                candidates=sub_cands,
                ground_truth=dev_gt,
                s1_metadata=s1_metadata,
                target_metadata=target_metadata,
                total_target_records=generator.indexed_target_count,
            )
            print(
                f"{label:<45}: Recall = {sub_rep.pair_recall*100:6.2f}% | Candidates = {len(sub_cands):,}"
            )
            channel_steps.append((label, sub_rep.pair_recall, len(sub_cands)))

    # Print summary
    print("\n" + report.to_markdown())

    # Save report to output
    if output_report_path:
        out_p = Path(output_report_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        report_text = (
            "# Candidate Generation Baseline Evaluation Report\n\n"
            f"**Evaluation Split Size:** {dev_sample_size:,} Source 1 records  \n"
            f"**Random Seed:** {random_seed}  \n"
            f"**Search Space:** {generator.indexed_target_count:,} target records (S2 + S3)  \n\n"
            + report.to_markdown()
        )
        if channel_steps:
            report_text += "\n\n### Incremental Channel Ablation\n\n"
            report_text += "| Channel Addition | Candidate Recall | Candidate Volume |\n|---|---|---|\n"
            for lbl, rec, c_vol in channel_steps:
                report_text += f"| {lbl} | {rec*100:.2f}% | {c_vol:,} |\n"

        out_p.write_text(report_text, encoding="utf-8")
        print(f"\nSaved evaluation report to {out_p}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate candidate generation baseline."
    )
    parser.add_argument(
        "--dev-sample",
        type=int,
        default=5000,
        help="Number of Source 1 entities for dev split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic dev split.",
    )
    parser.add_argument(
        "--target-scan-limit",
        type=int,
        default=100_000,
        help="Limit on background target records to index for quick evaluation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/candidate_recall_baseline.md",
        help="Path to save markdown report.",
    )

    args = parser.parse_args()
    run_evaluation(
        dev_sample_size=args.dev_sample,
        random_seed=args.seed,
        target_scan_limit=args.target_scan_limit,
        output_report_path=args.output,
    )


if __name__ == "__main__":
    main()

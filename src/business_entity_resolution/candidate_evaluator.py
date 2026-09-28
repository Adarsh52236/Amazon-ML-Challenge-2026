"""Candidate evaluation module for Business Entity Resolution.

Evaluates candidate pair quality, candidate recall ceiling against ground truth,
reduction ratios, candidate count distributions, country breakdowns, and channel
contributions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from business_entity_resolution.candidate_generator import CandidatePair

logger = logging.getLogger(__name__)


@dataclass
class CandidateEvaluationReport:
    """Comprehensive evaluation metrics for candidate generation blocking."""

    total_gt_pairs: int
    total_candidate_pairs: int
    found_gt_pairs: int
    missing_gt_pairs: int
    pair_recall: float
    reduction_ratio: float

    total_s1_entities: int
    s1_with_zero_candidates: int
    s1_with_at_least_one_candidate: int

    candidates_per_entity_mean: float
    candidates_per_entity_median: float
    candidates_per_entity_min: int
    candidates_per_entity_max: int
    candidates_per_entity_p90: float
    candidates_per_entity_p99: float

    recall_by_country: Dict[str, float] = field(default_factory=dict)
    recall_by_address_status: Dict[str, float] = field(default_factory=dict)
    channel_contribution: Dict[str, int] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Format metrics into a clean Markdown summary table."""
        lines = [
            "### Candidate Generation Recall & Performance Summary",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| **Total Reference Entities (S1)** | {self.total_s1_entities:,} |",
            f"| **Total Ground Truth Positive Pairs** | {self.total_gt_pairs:,} |",
            f"| **Total Generated Candidate Pairs** | {self.total_candidate_pairs:,} |",
            f"| **Ground Truth Pairs Recovered (TP)** | {self.found_gt_pairs:,} |",
            f"| **Ground Truth Pairs Missed (FN)** | {self.missing_gt_pairs:,} |",
            f"| **Candidate Pair Recall Ceiling** | **{self.pair_recall * 100:.2f}%** |",
            f"| **Search Space Reduction Ratio** | **{self.reduction_ratio * 100:.4f}%** |",
            f"| **S1 Entities with 0 Candidates** | {self.s1_with_zero_candidates:,} |",
            "",
            "#### Candidate Count per S1 Entity Distribution",
            "| Statistic | Value |",
            "|---|---|",
            f"| **Mean** | {self.candidates_per_entity_mean:.2f} |",
            f"| **Median** | {self.candidates_per_entity_median:.1f} |",
            f"| **Min / Max** | {self.candidates_per_entity_min} / {self.candidates_per_entity_max} |",
            f"| **90th Percentile** | {self.candidates_per_entity_p90:.1f} |",
            f"| **99th Percentile** | {self.candidates_per_entity_p99:.1f} |",
            "",
        ]

        if self.recall_by_country:
            lines.append("#### Recall by Country")
            lines.append("| Country | Recall |")
            lines.append("|---|---|")
            for ctry, rec in sorted(self.recall_by_country.items()):
                lines.append(f"| `{ctry}` | {rec * 100:.2f}% |")
            lines.append("")

        if self.recall_by_address_status:
            lines.append("#### Recall by Address Status in Secondary Match")
            lines.append("| Target Address Status | Recall |")
            lines.append("|---|---|")
            for status, rec in sorted(self.recall_by_address_status.items()):
                lines.append(f"| {status} | {rec * 100:.2f}% |")
            lines.append("")

        if self.channel_contribution:
            lines.append("#### True Positive Pairs Found by Channel")
            lines.append("| Blocking Channel | True Positives Retrieved |")
            lines.append("|---|---|")
            for chan, cnt in sorted(
                self.channel_contribution.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                lines.append(f"| `{chan}` | {cnt:,} |")
            lines.append("")

        return "\n".join(lines)


def evaluate_candidates(
    candidates: Sequence[CandidatePair],
    ground_truth: Dict[str, List[str]],
    s1_metadata: Optional[Dict[str, Dict[str, str]]] = None,
    target_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    total_target_records: int = 1,
) -> CandidateEvaluationReport:
    """Evaluate candidate pairs against ground truth matching labels.

    Parameters
    ----------
    candidates : Sequence[CandidatePair]
        Generated candidate pairs.
    ground_truth : Dict[str, List[str]]
        Ground truth mapping from `source1_entity_id -> list of true matching IDs`.
    s1_metadata : Optional[Dict[str, Dict[str, str]]]
        Optional metadata for S1 entities containing 'country' etc.
    target_metadata : Optional[Dict[str, Dict[str, Any]]]
        Optional metadata for target entities (e.g. 'is_address_empty').
    total_target_records : int
        Total number of target records in the search space (for reduction ratio calculation).

    Returns
    -------
    CandidateEvaluationReport
        Detailed evaluation metrics.
    """
    # Build candidate lookup: (s1_id, target_id) -> channels
    cand_lookup: Dict[Tuple[str, str], List[str]] = {}
    cands_by_s1: Dict[str, Set[str]] = defaultdict(set)

    for c in candidates:
        pair = (c.source1_entity_id, c.matched_entity_id)
        cand_lookup[pair] = c.channels
        cands_by_s1[c.source1_entity_id].add(c.matched_entity_id)

    # Count ground truth pairs
    total_gt_pairs = 0
    found_gt_pairs = 0
    missing_gt_pairs = 0

    country_gt_total = defaultdict(int)
    country_gt_found = defaultdict(int)

    addr_status_total = defaultdict(int)
    addr_status_found = defaultdict(int)

    channel_hits: Dict[str, int] = defaultdict(int)

    for s1_id, true_targets in ground_truth.items():
        ctry = (
            s1_metadata.get(s1_id, {}).get("country", "UNKNOWN")
            if s1_metadata
            else "ALL"
        )
        for target_id in true_targets:
            total_gt_pairs += 1
            country_gt_total[ctry] += 1

            # Check address status if target metadata provided
            if isinstance(target_metadata, dict) and target_id in target_metadata:
                is_empty = target_metadata[target_id].get("is_address_empty", False)
                status_label = "Empty Target Address" if is_empty else "Present Target Address"
                addr_status_total[status_label] += 1
            else:
                status_label = None

            pair = (s1_id, target_id)
            if pair in cand_lookup:
                found_gt_pairs += 1
                country_gt_found[ctry] += 1
                if status_label:
                    addr_status_found[status_label] += 1
                for ch in cand_lookup[pair]:
                    channel_hits[ch] += 1
            else:
                missing_gt_pairs += 1

    pair_recall = (
        (found_gt_pairs / total_gt_pairs) if total_gt_pairs > 0 else 0.0
    )

    # Reduction ratio
    total_s1 = len(ground_truth)
    total_possible_pairs = max(1, total_s1 * total_target_records)
    reduction_ratio = max(
        0.0, 1.0 - (len(candidates) / total_possible_pairs)
    )

    # Distribution of candidates per S1 entity
    cand_counts = [len(cands_by_s1.get(s1_id, set())) for s1_id in ground_truth]
    zero_cands = sum(1 for cnt in cand_counts if cnt == 0)

    if cand_counts:
        mean_c = float(np.mean(cand_counts))
        med_c = float(np.median(cand_counts))
        min_c = int(np.min(cand_counts))
        max_c = int(np.max(cand_counts))
        p90_c = float(np.percentile(cand_counts, 90))
        p99_c = float(np.percentile(cand_counts, 99))
    else:
        mean_c = med_c = p90_c = p99_c = 0.0
        min_c = max_c = 0

    # Recall by country
    recall_by_country: Dict[str, float] = {}
    for ctry, total_c in country_gt_total.items():
        found_c = country_gt_found[ctry]
        recall_by_country[ctry] = (found_c / total_c) if total_c > 0 else 0.0

    # Recall by address status
    recall_by_address: Dict[str, float] = {}
    for status, total_s in addr_status_total.items():
        found_s = addr_status_found[status]
        recall_by_address[status] = (found_s / total_s) if total_s > 0 else 0.0

    return CandidateEvaluationReport(
        total_gt_pairs=total_gt_pairs,
        total_candidate_pairs=len(candidates),
        found_gt_pairs=found_gt_pairs,
        missing_gt_pairs=missing_gt_pairs,
        pair_recall=round(pair_recall, 4),
        reduction_ratio=round(reduction_ratio, 6),
        total_s1_entities=total_s1,
        s1_with_zero_candidates=zero_cands,
        s1_with_at_least_one_candidate=total_s1 - zero_cands,
        candidates_per_entity_mean=round(mean_c, 2),
        candidates_per_entity_median=round(med_c, 2),
        candidates_per_entity_min=min_c,
        candidates_per_entity_max=max_c,
        candidates_per_entity_p90=round(p90_c, 2),
        candidates_per_entity_p99=round(p99_c, 2),
        recall_by_country=recall_by_country,
        recall_by_address_status=recall_by_address,
        channel_contribution=dict(channel_hits),
    )

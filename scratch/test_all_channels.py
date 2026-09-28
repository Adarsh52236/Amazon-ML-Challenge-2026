import os
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.candidate_generator import CandidateGenerator

def evaluate_candidates():
    # Load sample 5,000 S1 entities
    s1_df = pd.read_csv(ROOT / "output" / "dev_source1_sample.csv")
    s2_df = pd.read_csv(ROOT / "student_resource" / "dataset" / "train" / "train_source2.tsv", sep="\t")
    s3_df = pd.read_csv(ROOT / "student_resource" / "dataset" / "train" / "train_source3.tsv", sep="\t")
    gt_df = pd.read_csv(ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv", sep="\t")

    # Filter targets to countries present in S1
    s1_countries = s1_df["country"].unique()
    s2_df = s2_df[s2_df["country"].isin(s1_countries)]
    s3_df = s3_df[s3_df["country"].isin(s1_countries)]
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)

    # Filter ground truth
    gt_df = gt_df[gt_df["source1_entity_id"].isin(s1_df["entity_id"])]
    ground_truth_pairs = set(zip(gt_df["source1_entity_id"], gt_df["target_entity_id"]))
    print(f"Total ground-truth positive pairs for sample: {len(ground_truth_pairs)}")

    # Test config
    config = BlockingConfig(
        use_stem_prefix=True,
        use_transliteration=True,
        use_rare_name_token=True,
        max_candidates_per_entity=200, # Bumping up
        max_key_frequency=2500 # Bumping up
    )

    generator = CandidateGenerator(config=config)
    print("Building target index...")
    generator.build_index_from_dataframe(targets_df)
    
    pruned = generator.prune_high_frequency_keys()
    print(f"Pruned keys: {pruned}")
    
    print("Generating candidates...")
    candidates = generator.generate_candidates_for_dataframe(s1_df)
    
    print(f"Total candidate pairs generated: {len(candidates)}")
    print(f"Average candidates per S1 entity: {len(candidates) / len(s1_df):.2f}")
    
    candidate_set = set((c.source1_entity_id, c.matched_entity_id) for c in candidates)
    
    found = candidate_set.intersection(ground_truth_pairs)
    missed = ground_truth_pairs - candidate_set
    
    recall = len(found) / len(ground_truth_pairs) if ground_truth_pairs else 0
    print(f"Ground-truth pairs found: {len(found)}")
    print(f"Ground-truth pairs missed: {len(missed)}")
    print(f"Candidate Recall: {recall:.4f} ({recall*100:.2f}%)")

if __name__ == "__main__":
    evaluate_candidates()

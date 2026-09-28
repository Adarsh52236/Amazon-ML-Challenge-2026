import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from business_entity_resolution.data_loader import load_source_file, iter_source_records
from business_entity_resolution.config import get_source_path
from business_entity_resolution.candidate_generator import CandidateGenerator
from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.features.pairwise_extractor import PairwiseFeatureExtractor

def main():
    print("Loading S1...", flush=True)
    s1_all = load_source_file("train", 1)
    
    # Target index
    print("Building Target Index...", flush=True)
    b_cfg = BlockingConfig(max_candidates_per_entity=100)
    generator = CandidateGenerator(config=b_cfg)
    
    # We will just index all
    s2_path = get_source_path("train", 2)
    s3_path = get_source_path("train", 3)
    
    raw_targets = {}
    for r in iter_source_records(s2_path):
        eid = r["entity_id"]
        generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
        raw_targets[eid] = {"entity_id": eid, "name": r.get("business_name", ""), "address": r.get("business_address", ""), "country": r.get("country", ""), "source": "S2"}
        
    for r in iter_source_records(s3_path):
        eid = r["entity_id"]
        generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
        raw_targets[eid] = {"entity_id": eid, "name": r.get("business_name", ""), "address": r.get("business_address", ""), "country": r.get("country", ""), "source": "S3"}

    generator.prune_high_frequency_keys()
    
    n_total = len(s1_all)
    chunk_size = n_total // 20
    print(f"Total S1: {n_total}. Chunk size for 5%: {chunk_size}", flush=True)
    
    extractor = PairwiseFeatureExtractor()
    
    for pct in range(1, 21):
        start = (pct - 1) * chunk_size
        end = pct * chunk_size if pct < 20 else n_total
        
        batch = s1_all.iloc[start:end].copy()
        cands = generator.generate_candidates_for_dataframe(batch)
        
        # Preprocess targets in this batch
        cand_tids = {c.matched_entity_id for c in cands}
        target_preprocessed = {}
        for tid in cand_tids:
            r = raw_targets[tid]
            target_preprocessed[tid] = extractor.preprocess_entity(tid, r["name"], r["address"], r["country"], r["source"])
            
        s1_prep = {}
        for _, row in batch.iterrows():
            eid = str(row["entity_id"])
            s1_prep[eid] = extractor.preprocess_entity(eid, row.get("business_name", ""), row.get("business_address", ""), row.get("country", ""), "S1")
            
        # extract
        extractor.extract_features_for_candidates(cands, s1_prep, target_preprocessed)
        
        print(f"PROGRESS: {pct * 5}%", flush=True)

if __name__ == "__main__":
    main()

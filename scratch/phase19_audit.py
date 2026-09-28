import os
import pandas as pd
from pathlib import Path

def main():
    root = Path('c:/Users/adars/Downloads/business_entity_resolution')
    
    test_s1 = pd.read_csv(root / 'student_resource' / 'dataset' / 'test' / 'test_source1.tsv', sep='\t')
    test_s2 = pd.read_csv(root / 'student_resource' / 'dataset' / 'test' / 'test_source2.tsv', sep='\t')
    test_s3 = pd.read_csv(root / 'student_resource' / 'dataset' / 'test' / 'test_source3.tsv', sep='\t')
    
    valid_s1_ids = set(test_s1['entity_id'])
    valid_target_ids = set(test_s2['entity_id']).union(set(test_s3['entity_id']))
    
    print(f"Total valid S1: {len(valid_s1_ids)}")
    print(f"Total valid targets: {len(valid_target_ids)}")
    
    # Read output
    try:
        results_df = pd.read_csv(root / 'output' / 'submission_candidate_075.tsv', sep='\t', dtype=str)
        cands_df = pd.read_csv(root / 'output' / 'candidate_pairs.tsv', sep='\t', dtype=str)
    except FileNotFoundError:
        print("Output files not found yet.")
        return
        
    s1_in_results = results_df['source1_entity_id'].tolist()
    
    missing_s1 = valid_s1_ids - set(s1_in_results)
    extra_s1 = set(s1_in_results) - valid_s1_ids
    
    duplicate_rows = len(s1_in_results) - len(set(s1_in_results))
    
    total_links = 0
    total_cands = 0
    zero_match = 0
    singleton = 0
    multi_match = 0
    max_links = 0
    
    duplicate_ids = 0
    invalid_ids = 0
    
    dist = []
    
    for _, row in results_df.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) else ""
        if not matches.strip():
            count = 0
        else:
            ids = matches.split(',')
            count = len(ids)
            if len(set(ids)) < len(ids):
                duplicate_ids += 1
            for i in ids:
                if i not in valid_target_ids:
                    invalid_ids += 1
        
        dist.append(count)
        total_links += count
        if count == 0: zero_match += 1
        elif count == 1: singleton += 1
        else: multi_match += 1
        
        if count > max_links: max_links = count
        
    for _, row in cands_df.iterrows():
        cands = str(row['candidate_entity_ids']) if pd.notna(row['candidate_entity_ids']) else ""
        if cands.strip():
            total_cands += len(cands.split(','))
            
    print(f"Test S1 Count: {len(valid_s1_ids)}")
    print(f"Candidates generated: {total_cands}")
    print(f"Predicted links: {total_links}")
    print(f"Zero-match S1: {zero_match}")
    print(f"Singleton S1: {singleton}")
    print(f"Multi-match S1: {multi_match}")
    print(f"Max links/S1: {max_links}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Duplicate IDs within row: {duplicate_ids}")
    print(f"Invalid IDs: {invalid_ids}")
    print(f"Missing S1: {len(missing_s1)}")
    print(f"Extra S1: {len(extra_s1)}")
    
if __name__ == "__main__":
    main()

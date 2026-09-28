import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.candidate_generator import CandidateGenerator
from business_entity_resolution.data_loader import load_ground_truth, load_source_file, iter_source_records, get_source_path

def main():
    seed = 42
    sample_size = 5000
    target_scan_limit = 100_000

    gt_mapping = load_ground_truth(as_dict=True)
    s1_all = load_source_file("train", 1)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(s1_all), size=sample_size, replace=False)
    s1_dev = s1_all.iloc[sampled].copy().reset_index(drop=True)
    dev_s1_ids = set(s1_dev["entity_id"])

    dev_gt = {s1_id: gt_mapping.get(s1_id, []) for s1_id in dev_s1_ids if s1_id in gt_mapping}
    all_gt_pairs = set()
    for s1_id, targets in dev_gt.items():
        for t in targets:
            all_gt_pairs.add((s1_id, t))
    target_ids_needed = {t for _, t in all_gt_pairs}

    # Use max cap 10000 for "uncapped"
    config = BlockingConfig(
        use_stem_prefix=True, use_transliteration=True, use_rare_name_token=True,
        use_addr_alpha=True, max_candidates_per_entity=10000, max_key_frequency=5000
    )
    config.channel_weights = {} # Baseline behavior
    
    generator = CandidateGenerator(config=config)
    
    s2_path = get_source_path("train", 2)
    s3_path = get_source_path("train", 3)

    raw_targets = {}
    indexed_s2 = 0
    for r in iter_source_records(s2_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s2 < target_scan_limit:
            generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
            raw_targets[eid] = {"entity_id": eid, "country": r.get("country", "")}
            indexed_s2 += 1
            
    indexed_s3 = 0
    for r in iter_source_records(s3_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s3 < target_scan_limit:
            generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
            raw_targets[eid] = {"entity_id": eid, "country": r.get("country", "")}
            indexed_s3 += 1
            
    generator.prune_high_frequency_keys()
    
    print("Generating Uncapped candidates...")
    uncapped_cands = generator.generate_candidates_for_dataframe(s1_dev)
    
    # Sort them EXACTLY as CandidateGenerator would if limit was reached
    # Since config.channel_weights is {}, total_weight == len(channels)
    
    s1_to_cands = defaultdict(list)
    for c in uncapped_cands:
        s1_to_cands[c.source1_entity_id].append(c)
        
    for s1_id in s1_to_cands:
        cands = s1_to_cands[s1_id]
        # Sort by len(channels) descending. Python's sort is stable, which preserves insertion order (same as generator)
        cands.sort(key=lambda c: len(c.channels), reverse=True)
        s1_to_cands[s1_id] = cands

    s1_country = dict(zip(s1_dev['entity_id'], s1_dev['country']))
    
    lost_gt_pairs = []
    
    for s1_id, targets in dev_gt.items():
        if not targets: continue
            
        cands = s1_to_cands.get(s1_id, [])
        cand_t_ids = [c.matched_entity_id for c in cands]
        
        for t_id in targets:
            if t_id in cand_t_ids:
                rank = cand_t_ids.index(t_id) + 1
                cand_obj = cands[rank - 1]
                
                if rank > 100:
                    lost_gt_pairs.append({
                        'source1_entity_id': s1_id,
                        'target_entity_id': t_id,
                        'country': s1_country.get(s1_id, 'UNKNOWN'),
                        'uncapped_rank': rank,
                        'channel_count': len(cand_obj.channels),
                        'channels': ",".join(sorted(cand_obj.channels)),
                        'score': len(cand_obj.channels),
                        'cand_obj': cand_obj,
                        'all_cands': cands
                    })
                    
    print(f"Found {len(lost_gt_pairs)} GT pairs lost to cap=100")
    
    ranks = [p['uncapped_rank'] for p in lost_gt_pairs]
    p75, p90, p95, p99 = np.percentile(ranks, [75, 90, 95, 99])
    
    bins = {
        '101-125': 0, '126-150': 0, '151-200': 0, '201-300': 0, '301-500': 0, '500+': 0
    }
    for r in ranks:
        if 101 <= r <= 125: bins['101-125'] += 1
        elif 126 <= r <= 150: bins['126-150'] += 1
        elif 151 <= r <= 200: bins['151-200'] += 1
        elif 201 <= r <= 300: bins['201-300'] += 1
        elif 301 <= r <= 500: bins['301-500'] += 1
        else: bins['500+'] += 1
        
    lost_channel_counts = Counter()
    for p in lost_gt_pairs:
        for ch in p['cand_obj'].channels:
            lost_channel_counts[ch] += 1
            
    retained_gt_pairs = []
    for s1_id, targets in dev_gt.items():
        cands = s1_to_cands.get(s1_id, [])[:100]
        cand_t_ids = [c.matched_entity_id for c in cands]
        for t_id in targets:
            if t_id in cand_t_ids:
                rank = cand_t_ids.index(t_id) + 1
                retained_gt_pairs.append(cands[rank - 1])
                
    retained_channel_counts = Counter()
    for c in retained_gt_pairs:
        for ch in c.channels:
            retained_channel_counts[ch] += 1
            
    combinations = Counter()
    len_counts = Counter()
    for p in lost_gt_pairs:
        combo = ",".join(sorted(p['cand_obj'].channels))
        combinations[combo] += 1
        len_counts[len(p['cand_obj'].channels)] += 1
        
    top_combos = combinations.most_common(20)
    
    country_stats = {}
    for ctry in ['India', 'US']:
        c_lost = [p for p in lost_gt_pairs if p['country'] == ctry]
        total_gt = sum(1 for s1, t in all_gt_pairs if s1_country.get(s1) == ctry)
        country_stats[ctry] = {
            'lost': len(c_lost),
            'total': total_gt,
            'pct': len(c_lost) / max(1, total_gt),
            'median_rank': np.median([p['uncapped_rank'] for p in c_lost]) if c_lost else 0,
            'mean_rank': np.mean([p['uncapped_rank'] for p in c_lost]) if c_lost else 0,
        }
        
    tie_cutoff_losses = 0
    weak_channel_losses = 0
    deep_losses = 0
    
    for p in lost_gt_pairs:
        s1_id = p['source1_entity_id']
        cands = p['all_cands']
        score = p['score']
        
        top_100 = cands[:100]
        competing_scores = [len(c.channels) for c in top_100]
        
        cutoff_score = competing_scores[99] if len(competing_scores) == 100 else 0
        
        if score == cutoff_score:
            tie_cutoff_losses += 1
        elif score < cutoff_score:
            if score == 1:
                weak_channel_losses += 1
            else:
                deep_losses += 1
                
    def simulate_ranking(sort_fn, cap=100):
        found = 0
        in_found = 0
        us_found = 0
        for s1_id, targets in dev_gt.items():
            if not targets: continue
            ctry = s1_country.get(s1_id)
            cands = list(s1_to_cands.get(s1_id, []))
            
            cands.sort(key=sort_fn, reverse=True)
            
            capped = cands[:cap]
            capped_tids = {c.matched_entity_id for c in capped}
            
            for t in targets:
                if t in capped_tids:
                    found += 1
                    if ctry == 'India': in_found += 1
                    if ctry == 'US': us_found += 1
        return found, in_found, us_found

    base_found, base_in, base_us = simulate_ranking(lambda c: len(c.channels))
    
    def sort_exact_name(c):
        has_exact = 'exact_name' in c.channels or 'country_exact_name' in c.channels
        return (len(c.channels), 1 if has_exact else 0)
    en_found, en_in, en_us = simulate_ranking(sort_exact_name)
    
    def sort_tid_asc(c):
        return (len(c.channels), -hash(c.matched_entity_id))
    tid_found, tid_in, tid_us = simulate_ranking(sort_tid_asc)

    tsv_rows = []
    for p in lost_gt_pairs:
        tsv_rows.append({
            'source1_entity_id': p['source1_entity_id'],
            'target_entity_id': p['target_entity_id'],
            'country': p['country'],
            'uncapped_rank': p['uncapped_rank'],
            'channel_count': p['channel_count'],
            'channels': p['channels'],
            'rank_bucket': '101-125' if 101<=p['uncapped_rank']<=125 else ('126-150' if 126<=p['uncapped_rank']<=150 else ('500+' if p['uncapped_rank']>500 else str(p['uncapped_rank']))),
            'channel_combination': p['channels']
        })
    pd.DataFrame(tsv_rows).to_csv(ROOT / "output/phase_12_ranking_failure_forensics.tsv", sep='\t', index=False)
    
    md = f"""# Phase 12: Forensic Analysis of Ranking/Cap Failures

## 1. IDENTIFY ALL 557 PAIRS
Found {len(lost_gt_pairs)} GT pairs lost to cap=100.
(Details in `output/phase_12_ranking_failure_forensics.tsv`)

## 2. RANK DISTRIBUTION
- Median rank: {np.median(ranks):.1f}
- Mean rank: {np.mean(ranks):.1f}
- P75: {p75:.1f}
- P90: {p90:.1f}
- P95: {p95:.1f}
- P99: {p99:.1f}
- Maximum rank: {np.max(ranks)}

Bins:
- 101–125: {bins['101-125']}
- 126–150: {bins['126-150']}
- 151–200: {bins['151-200']}
- 201–300: {bins['201-300']}
- 301–500: {bins['301-500']}
- 500+: {bins['500+']}

## 3. CHANNEL DISTRIBUTION
Lost GT Channel Distribution:
"""
    for ch, count in lost_channel_counts.most_common():
        md += f"- {ch}: {count} ({(count/len(lost_gt_pairs))*100:.1f}%)\n"
        
    md += "\nRetained GT Channel Distribution:\n"
    for ch, count in retained_channel_counts.most_common():
        md += f"- {ch}: {count} ({(count/len(retained_gt_pairs))*100:.1f}%)\n"
        
    md += f"""
## 4. CHANNEL COMBINATION ANALYSIS
Length counts:
- 1 channel: {len_counts[1]}
- 2 channels: {len_counts[2]}
- 3 channels: {len_counts[3]}
- 4+ channels: {sum(v for k,v in len_counts.items() if k >= 4)}

Top 20 Combinations:
"""
    for combo, count in top_combos:
        md += f"- {combo}: {count}\n"
        
    md += f"""
## 5. INDIA VS US
- India: {country_stats['India']['lost']} lost ({country_stats['India']['pct']*100:.1f}% of India GT). Median rank: {country_stats['India']['median_rank']:.1f}
- US: {country_stats['US']['lost']} lost ({country_stats['US']['pct']*100:.1f}% of US GT). Median rank: {country_stats['US']['median_rank']:.1f}

## 6. COMPETING CANDIDATES
- Tie cutoff losses (score == cutoff_score): {tie_cutoff_losses}
- Weak channel losses (score=1 < cutoff_score): {weak_channel_losses}
- Deep losses (score>1 < cutoff_score): {deep_losses}

## 8. COUNTERFACTUAL RANKING SIMULATION
- Current (len channels): {base_found}
- Exact Name tie-break: {en_found}
- Target ID tie-break: {tid_found}

## 10. IDENTIFY THE REAL OPPORTUNITY
A. Easily recoverable near-cap tie-break candidates: {tie_cutoff_losses} ({(tie_cutoff_losses/len(lost_gt_pairs))*100:.1f}%)
B. Weak-channel collision losses: {weak_channel_losses} ({(weak_channel_losses/len(lost_gt_pairs))*100:.1f}%)

Strongest Hypothesis: A large portion of lost candidates are tied at the score cutoff, or are generated by a single weak channel that gets buried by noisy candidates.
"""
    with open(ROOT / "documentation/daily/phase_12_ranking_failure_forensics.md", "w") as f:
        f.write(md)

if __name__ == "__main__":
    main()

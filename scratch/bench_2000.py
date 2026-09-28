import time
import numpy as np
import pandas as pd
from business_entity_resolution.candidate_generator import CandidateGenerator
from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.features.pairwise_extractor import PairwiseFeatureExtractor

# Test direct numpy extraction speed
print("Testing batch of 2000 S1 entities...")
s2 = pd.read_csv('student_resource/dataset/test/test_source2.tsv', sep='\t')
s2_fr = s2[s2['country'] == 'France']
del s2
s3 = pd.read_csv('student_resource/dataset/test/test_source3.tsv', sep='\t')
s3_fr = s3[s3['country'] == 'France']
del s3

bconfig = BlockingConfig(use_stem_prefix=True, max_candidates_per_entity=100)
gen = CandidateGenerator(config=bconfig)
for row in s2_fr.itertuples(index=False):
    gen.index_target_record(row[0], str(row[1]), str(row[2]), str(row[3]))
for row in s3_fr.itertuples(index=False):
    gen.index_target_record(row[0], str(row[1]), str(row[2]), str(row[3]))
gen.prune_high_frequency_keys()

s1 = pd.read_csv('student_resource/dataset/test/test_source1.tsv', sep='\t')
s1_fr = s1[s1['country'] == 'France'].iloc[:2000]

# Lookup for targets
fr_targets = pd.concat([s2_fr, s3_fr], ignore_index=True)
target_name_map = dict(zip(fr_targets['entity_id'], fr_targets['business_name'].fillna('').astype(str)))
target_addr_map = dict(zip(fr_targets['entity_id'], fr_targets['business_address'].fillna('').astype(str)))
target_country_map = dict(zip(fr_targets['entity_id'], fr_targets['country'].fillna('').astype(str)))
del fr_targets

extractor = PairwiseFeatureExtractor()
feat_names = extractor.feature_names

t_gen_0 = time.time()
cands = []
for row in s1_fr.itertuples(index=False):
    c = gen.generate_candidates_for_record(row[0], str(row[1]), str(row[2]), str(row[3]))
    cands.extend(c)
t_gen = time.time() - t_gen_0
print(f"Generated {len(cands)} candidates in {t_gen:.2f}s")

# Preprocess
t_pre_0 = time.time()
pre_s1 = {row[0]: extractor.preprocess_entity(row[0], str(row[1]), str(row[2]), str(row[3]), 'S1')
          for row in s1_fr.itertuples(index=False)}
unique_targets = list({p.matched_entity_id for p in cands})
pre_targets = {tid: extractor.preprocess_entity(tid, target_name_map[tid], target_addr_map[tid], target_country_map[tid], tid[:2])
               for tid in unique_targets}
t_pre = time.time() - t_pre_0
print(f"Preprocessed {len(pre_s1)} S1 and {len(pre_targets)} targets in {t_pre:.2f}s")

# Direct extraction
t_feat_0 = time.time()
N = len(cands)
X = np.empty((N, len(feat_names)), dtype=np.float32)
for i, p in enumerate(cands):
    e1 = pre_s1[p.source1_entity_id]
    e2 = pre_targets[p.matched_entity_id]
    feats = extractor.extract_features_for_pair(e1, e2, p.channels)
    for j, fn in enumerate(feat_names):
        val = feats.get(fn, -1.0)
        X[i, j] = val if not (np.isnan(val) or np.isinf(val)) else -1.0
t_feat = time.time() - t_feat_0
print(f"Extracted {N} feature vectors in {t_feat:.2f}s ({N/t_feat:.1f} pairs/sec)")

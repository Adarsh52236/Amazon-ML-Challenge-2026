import time
import psutil
import pandas as pd
from business_entity_resolution.candidate_generator import CandidateGenerator
from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.features.pairwise_extractor import PairwiseFeatureExtractor
import xgboost as xgb
import lightgbm as lgb
import numpy as np

print("Testing France 1000 S1 batch...")
t0 = time.time()
s2_df = pd.read_csv('student_resource/dataset/test/test_source2.tsv', sep='\t')
s2_fr = s2_df[s2_df['country'] == 'France']
del s2_df
s3_df = pd.read_csv('student_resource/dataset/test/test_source3.tsv', sep='\t')
s3_fr = s3_df[s3_df['country'] == 'France']
del s3_df
print(f"Loaded France targets: S2={len(s2_fr)}, S3={len(s3_fr)} in {time.time()-t0:.2f}s")

# Index targets using itertuples
t1 = time.time()
bconfig = BlockingConfig(use_stem_prefix=True, max_candidates_per_entity=100)
gen = CandidateGenerator(config=bconfig)

# Index S2 and S3
for row in s2_fr.itertuples(index=False):
    gen.index_target_record(row[0], str(row[1]), str(row[2]), str(row[3]))
for row in s3_fr.itertuples(index=False):
    gen.index_target_record(row[0], str(row[1]), str(row[2]), str(row[3]))

print(f"Indexed {gen.indexed_target_count} France targets in {time.time()-t1:.2f}s, RAM: {psutil.Process().memory_info().rss / 1e6:.1f} MB")

# Test 1000 S1 records
s1_df = pd.read_csv('student_resource/dataset/test/test_source1.tsv', sep='\t')
s1_fr = s1_df[s1_df['country'] == 'France'].iloc[:1000]

t2 = time.time()
cands = []
for row in s1_fr.itertuples(index=False):
    c = gen.generate_candidates_for_record(row[0], str(row[1]), str(row[2]), str(row[3]))
    cands.extend(c)

print(f"Generated {len(cands)} candidates for 1000 S1 in {time.time()-t2:.2f}s (avg {len(cands)/1000:.1f} cands/entity)")

# Target lookup dict
t3 = time.time()
fr_targets = pd.concat([s2_fr, s3_fr], ignore_index=True)
target_dict = fr_targets.set_index('entity_id').to_dict('index')
print(f"Target dict created in {time.time()-t3:.2f}s")

# Feature extraction
t4 = time.time()
extractor = PairwiseFeatureExtractor()
pre_s1 = {row[0]: extractor.preprocess_entity(row[0], str(row[1]), str(row[2]), str(row[3]), 'S1')
          for row in s1_fr.itertuples(index=False)}
target_ids = list({p.matched_entity_id for p in cands})
pre_targets = {tid: extractor.preprocess_entity(tid, str(target_dict[tid]['business_name']), str(target_dict[tid]['business_address']), str(target_dict[tid]['country']), tid[:2])
               for tid in target_ids}
print(f"Preprocessed {len(pre_s1)} S1 and {len(pre_targets)} targets in {time.time()-t4:.2f}s")

t5 = time.time()
feat_df = extractor.extract_features_for_candidates(cands, pre_s1, pre_targets)
print(f"Extracted features for {len(feat_df)} candidate pairs in {time.time()-t5:.2f}s")

# Model inference
t6 = time.time()
model_xgb = xgb.Booster()
model_xgb.load_model('models/model_xgb_prod.json')
model_lgb = lgb.Booster(model_file='models/model_lgb_prod.txt')

X = feat_df[extractor.feature_names].values
dmat = xgb.DMatrix(X, feature_names=extractor.feature_names)
p_xgb = model_xgb.predict(dmat)
p_lgb = model_lgb.predict(X)
p_blend = 0.25 * p_lgb + 0.75 * p_xgb

matches_xgb = np.sum(p_xgb >= 0.45)
matches_lgb = np.sum(p_lgb >= 0.30)
matches_blend = np.sum(p_blend >= 0.45)
print(f"Predictions in {time.time()-t6:.2f}s: XGB matches={matches_xgb}, LGB matches={matches_lgb}, Blend matches={matches_blend}")

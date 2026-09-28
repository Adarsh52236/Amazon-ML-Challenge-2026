import time
import os
import psutil
from concurrent.futures import ProcessPoolExecutor
from business_entity_resolution.features.pairwise_extractor import PairwiseFeatureExtractor, PreprocessedEntity
from business_entity_resolution.text_normalizer import BusinessNameNormalizer
from business_entity_resolution.address_normalizer import BusinessAddressNormalizer

# Worker function at top level for Windows pickle compatibility
def extract_batch_worker(batch):
    extractor = PairwiseFeatureExtractor()
    results = []
    for item in batch:
        # item: (s1_id, t_id, s1_name, s1_addr, s1_country, t_name, t_addr, t_country, t_src, channels)
        s1_id, t_id, s1_name, s1_addr, s1_country, t_name, t_addr, t_country, t_src, channels = item
        e1 = extractor.preprocess_entity(s1_id, s1_name, s1_addr, s1_country, 'S1')
        e2 = extractor.preprocess_entity(t_id, t_name, t_addr, t_country, t_src)
        feats = extractor.extract_features_for_pair(e1, e2, channels)
        # Convert to ordered float array directly
        vec = [feats.get(fn, -1.0) for fn in extractor.feature_names]
        results.append((s1_id, t_id, vec))
    return results

if __name__ == '__main__':
    print("Testing multi-process feature worker pool...")
    test_batch = [
        ('S1-1', 'S2-2', 'Apple Inc', '1 Infinite Loop Cupertino CA', 'US',
         'Apple Computer Inc', '1 Infinite Loop Cupertino CA', 'US', 'S2', ['country_exact_name'])
    ] * 2000

    t0 = time.time()
    # 4 workers
    chunks = [test_batch[i:i+500] for i in range(0, len(test_batch), 500)]
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(extract_batch_worker, chunk) for chunk in chunks]
        res = []
        for f in futures:
            res.extend(f.result())

    elapsed = time.time() - t0
    print(f"Processed {len(res)} pairs in {elapsed:.2f}s ({len(res)/elapsed:.1f} pairs/sec)")

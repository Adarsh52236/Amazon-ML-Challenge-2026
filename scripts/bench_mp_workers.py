"""Benchmark parallel feature extraction to find optimal worker count."""
import sys, time, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src'))

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Must import from module (not define locally) for Windows spawn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_worker import extract_features_worker

from business_entity_resolution.features.pairwise_extractor import PairwiseFeatureExtractor
extractor = PairwiseFeatureExtractor()
feat_names = extractor.feature_names

N = 375000
raw = [
    ('S1-1','Societe Generale','1 Rue de la Paix Paris','France',
     'S2-1','Societe Generale SA','1 Rue de la Paix Paris','France','S2',
     ['country_exact_name','country_core_name'])
] * N

if __name__ == '__main__':
    print(f"Total pairs: {N:,}")
    print(f"CPUs: {os.cpu_count()}")

    for n_workers in [1, 2, 4, 6, 8]:
        chunks = []
        chunk_size = N // n_workers
        for i in range(n_workers):
            start = i * chunk_size
            end = start + chunk_size if i < n_workers - 1 else N
            chunks.append((raw[start:end], feat_names))

        t0 = time.time()
        if n_workers == 1:
            results = [extract_features_worker(chunks[0])]
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(extract_features_worker, chunks))
        elapsed = time.time() - t0
        total = sum(len(r) for r in results)
        print(f"Workers={n_workers}: {total:,} pairs in {elapsed:.2f}s = {total/elapsed:,.0f} pairs/sec  (batch_equiv={375000/(total/elapsed):.1f}s)")

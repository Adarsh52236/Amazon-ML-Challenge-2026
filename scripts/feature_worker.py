"""
Multiprocessing worker module for parallel feature extraction.

Must be a top-level importable module (not defined inside __main__)
so that Windows Process spawning can pickle the function reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def extract_features_worker(args):
    """
    Worker function called in a subprocess.

    args: (raw_pairs, feat_names)
    raw_pairs: list of (s1_id, s1_name, s1_addr, s1_ctry, t_id, t_name, t_addr, t_ctry, t_src, channels)

    Returns numpy array of shape (N, n_feats) as float32.
    """
    raw_pairs, feat_names = args
    from business_entity_resolution.features.pairwise_extractor import PairwiseFeatureExtractor
    extractor = PairwiseFeatureExtractor()

    N = len(raw_pairs)
    n_feats = len(feat_names)
    X = np.empty((N, n_feats), dtype=np.float32)

    feat_idx = {fn: j for j, fn in enumerate(feat_names)}
    
    e1_cache = {}
    e2_cache = {}

    for i, (s1_id, s1_name, s1_addr, s1_ctry, t_id, t_name, t_addr, t_ctry, t_src, channels) in enumerate(raw_pairs):
        if s1_id not in e1_cache:
            e1_cache[s1_id] = extractor.preprocess_entity(s1_id, s1_name, s1_addr, s1_ctry, "S1")
        e1 = e1_cache[s1_id]
        
        if t_id not in e2_cache:
            e2_cache[t_id] = extractor.preprocess_entity(t_id, t_name, t_addr, t_ctry, t_src)
        e2 = e2_cache[t_id]
        
        feats = extractor.extract_features_for_pair(e1, e2, channels)
        for fn, j in feat_idx.items():
            v = feats.get(fn, -1.0)
            X[i, j] = v if (v == v and v != float("inf") and v != float("-inf")) else -1.0

    return X

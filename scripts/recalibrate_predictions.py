"""
Fast recalibration: take existing matching_results_xgb.tsv (9.5M pairs at threshold=0.55),
re-extract features and re-predict, then apply a much higher threshold to kill false positives.

Key insight: competition uses F_0.5 (precision-heavy). With 9.5M matches for 1.7M entities
(avg 5.5 per entity), we have massive FPs. We need to be MUCH more conservative.

Strategy: threshold=0.85 + max 3 matches per S1 entity.
"""
from __future__ import annotations
import gc
import sys
import time
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from feature_worker import extract_features_worker
from business_entity_resolution.features.pairwise_extractor import PairwiseFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(ROOT / "output" / "recalibrate.log"), mode="w"),
    ]
)
log = logging.getLogger(__name__)

THRESHOLD = 0.95   # Very aggressive threshold for precision
MAX_MATCHES = 1    # Only 1 match per entity - most businesses have 0 or 1 match
N_WORKERS = 6      # Slightly fewer workers to save RAM after restart


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("RECALIBRATION PIPELINE")
    log.info(f"Threshold: {THRESHOLD} | Max matches per entity: {MAX_MATCHES}")
    log.info("=" * 70)

    # 1. Load target lookup maps
    log.info("Loading test source data...")
    DATA_DIR = ROOT / "student_resource" / "dataset" / "test"
    
    s2 = pd.read_csv(DATA_DIR / "test_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(DATA_DIR / "test_source3.tsv", sep="\t", dtype=str).fillna("")
    targets = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    name_map = targets.set_index("entity_id")["business_name"].to_dict()
    addr_map = targets.set_index("entity_id")["business_address"].to_dict()
    country_map = targets.set_index("entity_id")["country"].to_dict()
    del targets
    gc.collect()

    s1 = pd.read_csv(DATA_DIR / "test_source1.tsv", sep="\t", dtype=str).fillna("")
    s1_dict = {
        row.entity_id: (row.business_name, row.business_address, row.country)
        for row in s1.itertuples(index=False)
    }
    s1_order = s1["entity_id"].tolist()
    del s1
    gc.collect()
    log.info(f"Loaded {len(s1_order):,} S1 entities and target lookup maps.")

    # 2. Read existing XGB predictions (these are S1->target pairs that scored >=0.55)
    log.info("Reading existing matching_results_xgb.tsv...")
    xgb_preds = pd.read_csv(ROOT / "output" / "matching_results_xgb.tsv", sep="\t", dtype=str).fillna("")

    # Expand to raw pairs
    log.info("Expanding pairs...")
    all_pairs = []
    for row in xgb_preds.itertuples(index=False):
        s1_id = row.source1_entity_id
        if not row.matched_entity_ids:
            continue
        s1_name, s1_addr, s1_ctry = s1_dict.get(s1_id, ("", "", ""))
        for t_id in row.matched_entity_ids.split(","):
            t_id = t_id.strip()
            if not t_id:
                continue
            all_pairs.append((
                s1_id, s1_name, s1_addr, s1_ctry,
                t_id, name_map.get(t_id, ""), addr_map.get(t_id, ""), country_map.get(t_id, ""),
                t_id[:2], ["recalib"]
            ))

    total_pairs = len(all_pairs)
    log.info(f"Total pairs to re-score: {total_pairs:,}")

    del xgb_preds, name_map, addr_map, country_map
    gc.collect()

    # 3. Load model
    extractor = PairwiseFeatureExtractor()
    feat_names = extractor.feature_names
    log.info(f"Feature schema: {len(feat_names)} features")

    model_xgb = xgb.Booster()
    model_xgb.load_model(str(ROOT / "models" / "model_xgb_prod.json"))
    log.info("XGBoost model loaded.")

    # 4. Re-score in batches
    BATCH_SIZE = 400_000
    results_by_s1 = {}  # s1_id -> list of (prob, t_id)

    for b0 in range(0, total_pairs, BATCH_SIZE):
        b1 = min(b0 + BATCH_SIZE, total_pairs)
        batch = all_pairs[b0:b1]
        t_b = time.time()

        # Use np.array_split for correct even chunking (no off-by-one)
        indices = np.array_split(np.arange(len(batch)), N_WORKERS)
        chunks = []
        for idx_group in indices:
            if len(idx_group) > 0:
                chunks.append(([batch[i] for i in idx_group], feat_names))

        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            feats = list(ex.map(extract_features_worker, chunks))

        X = np.vstack(feats)
        del feats
        assert len(X) == len(batch), f"Feature count mismatch: {len(X)} vs {len(batch)}"

        dmat = xgb.DMatrix(X, feature_names=feat_names)
        del X
        probs = model_xgb.predict(dmat)
        del dmat

        for i, pair in enumerate(batch):
            s1_id = pair[0]
            t_id = pair[4]
            prob = float(probs[i])
            if prob >= THRESHOLD:
                if s1_id not in results_by_s1:
                    results_by_s1[s1_id] = []
                results_by_s1[s1_id].append((prob, t_id))

        del probs, batch
        gc.collect()

        elapsed = time.time() - t_b
        log.info(f"Batch {b0//BATCH_SIZE + 1}: processed {b1:,}/{total_pairs:,} | "
                 f"accepted above thresh: {sum(len(v) for v in results_by_s1.values()):,} | "
                 f"{elapsed:.1f}s")

    del all_pairs
    gc.collect()

    # 5. Apply top-K cap and write output
    log.info(f"Applying top-{MAX_MATCHES} cap per entity and writing output...")
    
    out_path = ROOT / "output" / "matching_results_recalibrated.tsv"
    total_matches = 0
    singleton_count = 0

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_order:
            matches = results_by_s1.get(s1_id, [])
            if matches:
                # Sort by prob desc, take top K, deduplicate
                matches.sort(key=lambda x: -x[0])
                top_ids = list(dict.fromkeys(t_id for _, t_id in matches[:MAX_MATCHES]))
                f.write(f"{s1_id}\t{','.join(top_ids)}\n")
                total_matches += len(top_ids)
            else:
                f.write(f"{s1_id}\t\n")
                singleton_count += 1

    log.info(f"Written {len(s1_order):,} rows")
    log.info(f"Total matches: {total_matches:,}")
    log.info(f"Predicted singletons: {singleton_count:,} ({100*singleton_count/len(s1_order):.1f}%)")
    log.info(f"Avg matches per entity: {total_matches/len(s1_order):.2f}")

    # 6. Also copy to matching_results.tsv (the champion file)
    import shutil
    shutil.copy(out_path, ROOT / "output" / "matching_results.tsv")
    log.info("Copied to matching_results.tsv")

    # 7. Rebuild zip
    log.info("Rebuilding submission zip...")
    zip_path = ROOT / "business_entity_resolution_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(ROOT / "output" / "matching_results.tsv", "output/matching_results.tsv")
        zf.write(ROOT / "output" / "candidate_pairs.tsv", "output/candidate_pairs.tsv")
        for root, dirs, files in (ROOT / "src").walk() if hasattr((ROOT / "src"), "walk") else __import__("os").walk(str(ROOT / "src")):
            for file in files:
                if ".pyc" in file or "__pycache__" in str(root):
                    continue
                fpath = Path(root) / file
                arcname = "code/business_entity_resolution/" + str(fpath.relative_to(ROOT))
                zf.write(fpath, arcname)
        zf.write(ROOT / "README.md", "code/business_entity_resolution/README.md")
        zf.write(ROOT / "requirements.txt", "code/business_entity_resolution/requirements.txt")
        zf.write(ROOT / "documentation" / "Documentation_template.md", "Documentation_template.md")

    elapsed_total = time.time() - t0
    log.info(f"DONE in {elapsed_total/60:.1f} min. Zip at: {zip_path}")
    log.info("Upload output/matching_results.tsv to the leaderboard!")


if __name__ == "__main__":
    main()

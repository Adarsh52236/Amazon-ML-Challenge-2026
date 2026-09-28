"""
Production inference pipeline for Business Entity Resolution — Phase 6.

Processes test data country-by-country (France, US, India) to stay within RAM.
Generates candidates -> extracts 54 pairwise features -> predicts with XGB, LGB, Blend.

Output files:
  output/candidate_pairs.tsv
  output/matching_results_xgb.tsv
  output/matching_results_lgb.tsv
  output/matching_results_blend.tsv
  output/matching_results.tsv   (champion = XGBoost @0.45)

Constraints:
  - No ground-truth labels used (test-only inference).
  - No one-to-one / argmax constraint; multi-match preserved.
  - No external APIs or synthetic data.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import logging
import hashlib
import datetime
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from concurrent.futures import ProcessPoolExecutor
from business_entity_resolution.candidate_generator import CandidateGenerator
from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.features.pairwise_extractor import PairwiseFeatureExtractor

sys.path.insert(0, str(ROOT / "scripts"))
from feature_worker import extract_features_worker

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "phase_6_pipeline.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

DATA_DIR = ROOT / "student_resource" / "dataset" / "test"
MODEL_XGB = ROOT / "models" / "model_xgb_prod.json"
MODEL_LGB = ROOT / "models" / "model_lgb_prod.txt"

THRESHOLD_XGB = 0.75
THRESHOLD_LGB = 0.75
THRESHOLD_BLEND = 0.75
BLEND_LGB_WEIGHT = 0.25
BLEND_XGB_WEIGHT = 0.75

COUNTRIES = ["France", "US", "India"]
BATCH_SIZE_S1 = 5000


def peak_ram_mb():
    return psutil.Process().memory_info().rss / 1e6


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_models():
    log.info("Loading production models...")
    model_xgb = xgb.Booster()
    model_xgb.load_model(str(MODEL_XGB))
    model_lgb = lgb.Booster(model_file=str(MODEL_LGB))
    xgb_kb = MODEL_XGB.stat().st_size / 1024
    lgb_kb = MODEL_LGB.stat().st_size / 1024
    log.info(f"Models loaded. XGB: {xgb_kb:.0f} KB, LGB: {lgb_kb:.0f} KB")
    return model_xgb, model_lgb


def build_country_index(country):
    log.info(f"[{country}] Loading S2/S3 targets...")
    t0 = time.time()

    s2 = pd.read_csv(DATA_DIR / "test_source2.tsv", sep="\t", dtype=str)
    s2_c = s2[s2["country"] == country].copy()
    del s2
    s3 = pd.read_csv(DATA_DIR / "test_source3.tsv", sep="\t", dtype=str)
    s3_c = s3[s3["country"] == country].copy()
    del s3

    n_s2, n_s3 = len(s2_c), len(s3_c)
    n_total = n_s2 + n_s3
    log.info(f"[{country}] S2={n_s2:,}, S3={n_s3:,}, total={n_total:,}")

    bconfig = BlockingConfig(
        use_stem_prefix=True,
        use_transliteration=True,
        use_rare_name_token=True,
        use_addr_alpha=True,
        use_addr_numbers=True,
        use_name_no_space=True,
        max_candidates_per_entity=100,
        max_key_frequency=5000
    )
    bconfig.channel_weights = {}
    gen = CandidateGenerator(config=bconfig)

    log.info(f"[{country}] Indexing...")
    for row in s2_c.itertuples(index=False):
        gen.index_target_record(
            row[0],
            str(row[1]) if str(row[1]) != "nan" else "",
            str(row[2]) if str(row[2]) != "nan" else "",
            str(row[3]) if str(row[3]) != "nan" else "",
        )
    for row in s3_c.itertuples(index=False):
        gen.index_target_record(
            row[0],
            str(row[1]) if str(row[1]) != "nan" else "",
            str(row[2]) if str(row[2]) != "nan" else "",
            str(row[3]) if str(row[3]) != "nan" else "",
        )

    pruned = gen.prune_high_frequency_keys()
    log.info(
        f"[{country}] Indexed {n_total:,} targets in {time.time()-t0:.1f}s | "
        f"RAM={peak_ram_mb():.0f} MB | pruned={pruned}"
    )

    all_targets = pd.concat([s2_c, s3_c], ignore_index=True)
    name_map = dict(zip(
        all_targets["entity_id"],
        all_targets["business_name"].fillna("").astype(str)
    ))
    addr_map = dict(zip(
        all_targets["entity_id"],
        all_targets["business_address"].fillna("").astype(str)
    ))
    country_map = dict(zip(
        all_targets["entity_id"],
        all_targets["country"].fillna("").astype(str)
    ))
    del all_targets, s2_c, s3_c
    gc.collect()

    return gen, name_map, addr_map, country_map, n_total


def run_country(
    country, s1_df, model_xgb, model_lgb, extractor, feat_names,
    cand_dict, match_xgb_dict, match_lgb_dict, match_blend_dict,
):
    log.info(f"\n{'='*60}")
    log.info(f"[{country}] Starting: {len(s1_df):,} S1 entities")
    t_country = time.time()

    gen, name_map, addr_map, country_map, n_targets = build_country_index(country)

    cstat = {
        "n_s1": len(s1_df),
        "n_targets": n_targets,
        "n_candidates": 0,
        "n_matches_xgb": 0,
        "n_matches_lgb": 0,
        "n_matches_blend": 0,
        "s1_zero_cand": 0,
        "runtime_s": 0.0,
        "channel_counts": {},
    }

    n_s1 = len(s1_df)
    n_batches = (n_s1 + BATCH_SIZE_S1 - 1) // BATCH_SIZE_S1
    log.info(f"[{country}] {n_batches} batches of {BATCH_SIZE_S1}")

    total_cands = 0
    channel_counts = {}

    for bi in range(n_batches):
        b0 = bi * BATCH_SIZE_S1
        b1 = min(b0 + BATCH_SIZE_S1, n_s1)
        batch_df = s1_df.iloc[b0:b1]
        t_b = time.time()

        # 1. Candidate generation
        batch_cands = []
        for row in batch_df.itertuples(index=False):
            s1_id = row[0]
            name = str(row[1]) if str(row[1]) != "nan" else ""
            addr = str(row[2]) if str(row[2]) != "nan" else ""
            ctry = str(row[3]) if str(row[3]) != "nan" else ""
            c = gen.generate_candidates_for_record(s1_id, name, addr, ctry)
            
            # Sort by len(channels) and cap to 100
            c.sort(key=lambda x: len(x.channels), reverse=True)
            c = c[:100]
            
            batch_cands.extend(c)
            if not c:
                cstat["s1_zero_cand"] += 1
                cand_dict[s1_id] = ""
                match_xgb_dict[s1_id] = ""
                match_lgb_dict[s1_id] = ""
                match_blend_dict[s1_id] = ""
            for pair in c:
                for ch in pair.channels:
                    channel_counts[ch] = channel_counts.get(ch, 0) + 1

        n_cands = len(batch_cands)
        total_cands += n_cands

        if n_cands == 0:
            log.info(
                f"[{country}] Batch {bi+1}/{n_batches} ({b0}-{b1}): "
                f"0 cands | {time.time()-t_b:.1f}s"
            )
            continue

        # 2. Preprocess / Build raw pairs for MP
        s1_dict = {
            row[0]: (
                str(row[1]) if str(row[1]) != "nan" else "",
                str(row[2]) if str(row[2]) != "nan" else "",
                str(row[3]) if str(row[3]) != "nan" else ""
            )
            for row in batch_df.itertuples(index=False)
        }

        pair_s1_ids = []
        pair_t_ids = []
        raw_pairs = []
        
        for pair in batch_cands:
            s1_id = pair.source1_entity_id
            t_id = pair.matched_entity_id
            s1_name, s1_addr, s1_ctry = s1_dict[s1_id]
            
            t_name = name_map.get(t_id, "")
            t_addr = addr_map.get(t_id, "")
            t_ctry = country_map.get(t_id, "")
            
            raw_pairs.append((s1_id, s1_name, s1_addr, s1_ctry, t_id, t_name, t_addr, t_ctry, t_id[:2], pair.channels))
            pair_s1_ids.append(s1_id)
            pair_t_ids.append(t_id)

        # 3. Feature extraction -> numpy via multiprocessing
        n_workers = 8
        chunks = []
        chunk_size = (n_cands + n_workers - 1) // n_workers
        for i in range(n_workers):
            start = i * chunk_size
            end = min(start + chunk_size, n_cands)
            if start < end:
                chunks.append((raw_pairs[start:end], feat_names))

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(extract_features_worker, chunks))
        
        X = np.vstack(results)

        # 4. Predict
        dmat = xgb.DMatrix(X, feature_names=feat_names)
        p_xgb = model_xgb.predict(dmat)
        p_lgb = model_lgb.predict(X)
        p_blend = BLEND_LGB_WEIGHT * p_lgb + BLEND_XGB_WEIGHT * p_xgb

        # 5. Group into per-entity dicts
        s1_cands_agg = {}
        s1_xgb_agg = {}
        s1_lgb_agg = {}
        s1_blend_agg = {}

        for i, (s1_id, t_id) in enumerate(zip(pair_s1_ids, pair_t_ids)):
            if s1_id not in s1_cands_agg:
                s1_cands_agg[s1_id] = []
                s1_xgb_agg[s1_id] = []
                s1_lgb_agg[s1_id] = []
                s1_blend_agg[s1_id] = []
            s1_cands_agg[s1_id].append(t_id)
            if p_xgb[i] >= THRESHOLD_XGB:
                s1_xgb_agg[s1_id].append(t_id)
            if p_lgb[i] >= THRESHOLD_LGB:
                s1_lgb_agg[s1_id].append(t_id)
            if p_blend[i] >= THRESHOLD_BLEND:
                s1_blend_agg[s1_id].append(t_id)

        for s1_id in s1_cands_agg:
            cand_dict[s1_id] = ",".join(sorted(set(s1_cands_agg[s1_id])))
            match_xgb_dict[s1_id] = ",".join(sorted(set(s1_xgb_agg[s1_id])))
            match_lgb_dict[s1_id] = ",".join(sorted(set(s1_lgb_agg[s1_id])))
            match_blend_dict[s1_id] = ",".join(sorted(set(s1_blend_agg[s1_id])))

        cstat["n_matches_xgb"] += int(np.sum(p_xgb >= THRESHOLD_XGB))
        cstat["n_matches_lgb"] += int(np.sum(p_lgb >= THRESHOLD_LGB))
        cstat["n_matches_blend"] += int(np.sum(p_blend >= THRESHOLD_BLEND))

        elapsed_b = time.time() - t_b
        log.info(
            f"[{country}] Batch {bi+1}/{n_batches} ({b0}-{b1}): "
            f"{n_cands:,} cands | XGB+={int(np.sum(p_xgb>=THRESHOLD_XGB))} | "
            f"RAM={peak_ram_mb():.0f} MB | {elapsed_b:.1f}s"
        )

        del X, dmat, p_xgb, p_lgb, p_blend
        del batch_cands, raw_pairs, pair_s1_ids, pair_t_ids
        gc.collect()

    cstat["n_candidates"] = total_cands
    cstat["runtime_s"] = time.time() - t_country
    cstat["channel_counts"] = channel_counts

    log.info(
        f"[{country}] DONE: {total_cands:,} cands | "
        f"XGB={cstat['n_matches_xgb']:,} LGB={cstat['n_matches_lgb']:,} "
        f"Blend={cstat['n_matches_blend']:,} | {cstat['runtime_s']/60:.1f} min"
    )

    del gen, name_map, addr_map, country_map
    gc.collect()
    return cstat


def write_submission_files(s1_order, cand_dict, match_xgb_dict, match_lgb_dict, match_blend_dict):
    log.info("\nWriting submission files...")
    paths = {
        "candidates": OUTPUT_DIR / "candidate_pairs.tsv",
        "xgb": OUTPUT_DIR / "matching_results_xgb.tsv",
        "lgb": OUTPUT_DIR / "matching_results_lgb.tsv",
        "blend": OUTPUT_DIR / "matching_results_blend.tsv",
        "champion": OUTPUT_DIR / "submission_candidate_075.tsv",
    }

    with (
        open(paths["candidates"], "w", encoding="utf-8", newline="") as fc,
        open(paths["xgb"], "w", encoding="utf-8", newline="") as fx,
        open(paths["lgb"], "w", encoding="utf-8", newline="") as fl,
        open(paths["blend"], "w", encoding="utf-8", newline="") as fb,
        open(paths["champion"], "w", encoding="utf-8", newline="") as fch,
    ):
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fx.write("source1_entity_id\tmatched_entity_ids\n")
        fl.write("source1_entity_id\tmatched_entity_ids\n")
        fb.write("source1_entity_id\tmatched_entity_ids\n")
        fch.write("source1_entity_id\tmatched_entity_ids\n")

        for s1_id in s1_order:
            fc.write(f"{s1_id}\t{cand_dict.get(s1_id, '')}\n")
            fx.write(f"{s1_id}\t{match_xgb_dict.get(s1_id, '')}\n")
            fl.write(f"{s1_id}\t{match_lgb_dict.get(s1_id, '')}\n")
            fb.write(f"{s1_id}\t{match_blend_dict.get(s1_id, '')}\n")
            fch.write(f"{s1_id}\t{match_blend_dict.get(s1_id, '')}\n")

    checksums = {}
    for name, path in paths.items():
        sz = path.stat().st_size
        sha = sha256_file(path)
        checksums[name] = sha
        log.info(f"  {path.name}: {sz:,} bytes | sha256={sha}")
    return checksums


def main():
    log_path = OUTPUT_DIR / "phase_6_pipeline.log"
    if log_path.exists():
        open(log_path, 'w').close()
        
    t_start = time.time()
    log.info("=" * 70)
    log.info("PHASE 6 PRODUCTION PIPELINE")
    log.info(f"Started: {datetime.datetime.now().isoformat()}")
    log.info(f"Thresholds: XGB={THRESHOLD_XGB} LGB={THRESHOLD_LGB} Blend={THRESHOLD_BLEND}")
    log.info(f"Blend weights: LGB={BLEND_LGB_WEIGHT} XGB={BLEND_XGB_WEIGHT}")
    log.info("=" * 70)

    model_xgb, model_lgb = load_models()
    extractor = PairwiseFeatureExtractor()
    feat_names = extractor.feature_names
    log.info(f"Feature schema: {len(feat_names)} features")

    log.info("\nLoading test_source1.tsv...")
    t0 = time.time()
    s1_df = pd.read_csv(DATA_DIR / "test_source1.tsv", sep="\t", dtype=str)
    s1_order = s1_df["entity_id"].tolist()
    log.info(f"S1: {len(s1_order):,} entities in {time.time()-t0:.1f}s")

    cand_dict = {}
    match_xgb_dict = {}
    match_lgb_dict = {}
    match_blend_dict = {}
    country_stats = {}
    total_targets = 0
    total_cands = 0
    total_xgb = 0
    total_lgb = 0
    total_blend = 0

    for country in COUNTRIES:
        s1_c = s1_df[s1_df["country"] == country].copy()
        if len(s1_c) == 0:
            log.warning(f"No S1 entities for '{country}', skipping.")
            continue
        try:
            cstat = run_country(
                country, s1_c, model_xgb, model_lgb, extractor, feat_names,
                cand_dict, match_xgb_dict, match_lgb_dict, match_blend_dict,
            )
            country_stats[country] = cstat
            total_targets += cstat["n_targets"]
            total_cands += cstat["n_candidates"]
            total_xgb += cstat["n_matches_xgb"]
            total_lgb += cstat["n_matches_lgb"]
            total_blend += cstat["n_matches_blend"]
        except Exception:
            log.error(f"[{country}] FATAL ERROR:")
            traceback.print_exc()
            sys.exit(1)

    missing = [x for x in s1_order if x not in cand_dict]
    if missing:
        log.warning(f"{len(missing):,} S1 IDs missing from dict; setting empty.")
        for x in missing:
            cand_dict[x] = ""
            match_xgb_dict[x] = ""
            match_lgb_dict[x] = ""
            match_blend_dict[x] = ""

    checksums = write_submission_files(
        s1_order, cand_dict, match_xgb_dict, match_lgb_dict, match_blend_dict
    )

    elapsed = time.time() - t_start
    log.info("\n" + "=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info(f"Runtime: {elapsed/60:.1f} min")
    log.info(f"S1 entities: {len(s1_order):,}")
    log.info(f"Target entities: {total_targets:,}")
    log.info(f"Candidate pairs: {total_cands:,}")
    log.info(f"XGBoost @{THRESHOLD_XGB}: {total_xgb:,} matches")
    log.info(f"LightGBM @{THRESHOLD_LGB}: {total_lgb:,} matches")
    log.info(f"Blend @{THRESHOLD_BLEND}: {total_blend:,} matches")
    log.info(f"Peak RAM: {peak_ram_mb():.0f} MB")
    log.info("=" * 70)

    out_stats = {
        "total_s1": len(s1_order),
        "total_targets": total_targets,
        "total_candidates": total_cands,
        "total_matches_xgb": total_xgb,
        "total_matches_lgb": total_lgb,
        "total_matches_blend": total_blend,
        "elapsed_total_s": elapsed,
        "country_stats": {
            k: {kk: vv for kk, vv in v.items() if kk != "channel_counts"}
            for k, v in country_stats.items()
        },
        "checksums": checksums,
        "thresholds": {
            "xgb": THRESHOLD_XGB,
            "lgb": THRESHOLD_LGB,
            "blend": THRESHOLD_BLEND,
        },
    }
    stats_path = OUTPUT_DIR / "phase_6_pipeline_stats.json"
    with open(stats_path, "w") as f:
        json.dump(out_stats, f, indent=2)
    log.info(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()

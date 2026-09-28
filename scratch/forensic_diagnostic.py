"""
Forensic diagnostic script - answers all 16 questions from the audit.
DO NOT modify pipeline. Diagnosis only.
"""
import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_TRAIN = ROOT / "student_resource" / "dataset" / "train"
DATA_TEST  = ROOT / "student_resource" / "dataset" / "test"
OUTPUT     = ROOT / "output"

print("=" * 70)
print("FORENSIC DIAGNOSTIC REPORT")
print("=" * 70)

# ── SECTION 1: Data Distribution ────────────────────────────────────────
print("\n--- SECTION 1: DATA DISTRIBUTION ---")

s1_train = pd.read_csv(DATA_TRAIN / "train_source1.tsv", sep="\t", dtype=str).fillna("")
s2_train = pd.read_csv(DATA_TRAIN / "train_source2.tsv", sep="\t", dtype=str).fillna("")
s3_train = pd.read_csv(DATA_TRAIN / "train_source3.tsv", sep="\t", dtype=str).fillna("")
gt_train = pd.read_csv(DATA_TRAIN / "train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

s1_test  = pd.read_csv(DATA_TEST / "test_source1.tsv",  sep="\t", dtype=str).fillna("")
s2_test  = pd.read_csv(DATA_TEST / "test_source2.tsv",  sep="\t", dtype=str).fillna("")
s3_test  = pd.read_csv(DATA_TEST / "test_source3.tsv",  sep="\t", dtype=str).fillna("")

print(f"\nTRAIN: S1={len(s1_train):,}, S2={len(s2_train):,}, S3={len(s3_train):,}")
print(f"TEST:  S1={len(s1_test):,},  S2={len(s2_test):,},  S3={len(s3_test):,}")

print("\n[Country distribution - TRAIN S1]")
print(s1_train["country"].value_counts(dropna=False).to_string())
print("\n[Country distribution - TEST S1]")
print(s1_test["country"].value_counts(dropna=False).to_string())
print("\n[Country distribution - TEST S2]")
print(s2_test["country"].value_counts(dropna=False).to_string())
print("\n[Country distribution - TEST S3]")
print(s3_test["country"].value_counts(dropna=False).to_string())

# ── SECTION 2: Ground Truth Analysis ────────────────────────────────────
print("\n--- SECTION 2: GROUND TRUTH / SINGLETON ANALYSIS (TRAIN) ---")

gt_train["n_matches"] = gt_train["matched_entity_ids"].apply(
    lambda x: len([v for v in str(x).split(",") if v.strip()]) if str(x).strip() else 0
)
print(f"Total S1 entities in GT: {len(gt_train):,}")
singletons_gt = (gt_train["n_matches"] == 0).sum()
print(f"True singletons (0 matches):   {singletons_gt:,} ({100*singletons_gt/len(gt_train):.1f}%)")
print(f"Entities with 1 match:         {(gt_train['n_matches']==1).sum():,} ({100*(gt_train['n_matches']==1).sum()/len(gt_train):.1f}%)")
print(f"Entities with 2+ matches:      {(gt_train['n_matches']>=2).sum():,} ({100*(gt_train['n_matches']>=2).sum()/len(gt_train):.1f}%)")
print(f"Entities with 3+ matches:      {(gt_train['n_matches']>=3).sum():,} ({100*(gt_train['n_matches']>=3).sum()/len(gt_train):.1f}%)")
print(f"Max matches for one entity:    {gt_train['n_matches'].max()}")
print(f"Mean matches (non-zero):       {gt_train[gt_train['n_matches']>0]['n_matches'].mean():.2f}")

print("\n[Match count distribution]")
print(gt_train["n_matches"].value_counts().sort_index().head(15).to_string())

# By country
print("\n[Singleton rate by country - TRAIN]")
for ctry in s1_train["country"].unique():
    s1_ctry = s1_train[s1_train["country"]==ctry]["entity_id"].tolist()
    gt_ctry = gt_train[gt_train["source1_entity_id"].isin(s1_ctry)]
    sing = (gt_ctry["n_matches"]==0).sum()
    total = len(gt_ctry)
    print(f"  {ctry}: {sing}/{total} singletons ({100*sing/max(total,1):.1f}%)")

# ── SECTION 3: Current Submission Analysis ────────────────────────────
print("\n--- SECTION 3: CURRENT SUBMISSION ANALYSIS ---")

sub = pd.read_csv(OUTPUT / "matching_results.tsv", sep="\t", dtype=str).fillna("")
sub["n_pred"] = sub["matched_entity_ids"].apply(
    lambda x: len([v for v in str(x).split(",") if v.strip()]) if str(x).strip() else 0
)
print(f"Total rows: {len(sub):,}")
print(f"Total predictions: {sub['n_pred'].sum():,}")
print(f"Avg predictions/entity: {sub['n_pred'].mean():.3f}")
print(f"Predicted singletons: {(sub['n_pred']==0).sum():,} ({100*(sub['n_pred']==0).sum()/len(sub):.1f}%)")

# ── SECTION 4: What GT predicts for test scale ────────────────────────
print("\n--- SECTION 4: EXPECTED TEST SCALE EXTRAPOLATION ---")
train_singleton_rate = singletons_gt / len(gt_train)
train_match_rate = 1 - train_singleton_rate
expected_test_singletons = int(train_singleton_rate * len(s1_test))
expected_test_matches = int(train_match_rate * len(s1_test))
avg_match_count_nonzero = gt_train[gt_train["n_matches"]>0]["n_matches"].mean()
expected_total_matches = int(expected_test_matches * avg_match_count_nonzero)
print(f"Train singleton rate: {100*train_singleton_rate:.1f}%")
print(f"Expected test singletons: ~{expected_test_singletons:,}")
print(f"Expected test entities with matches: ~{expected_test_matches:,}")
print(f"Expected total test matches: ~{expected_total_matches:,}")
print(f"\nCurrent submission: {sub['n_pred'].sum():,} matches")
print(f"Overpredict ratio: {sub['n_pred'].sum()/max(expected_total_matches,1):.2f}x")

# ── SECTION 5: Missingness analysis ────────────────────────────────────
print("\n--- SECTION 5: MISSINGNESS ANALYSIS ---")
for label, df in [("TRAIN S1", s1_train), ("TEST S1", s1_test), ("TEST S2", s2_test), ("TEST S3", s3_test)]:
    name_miss  = (df["business_name"]    == "").sum()
    addr_miss  = (df["business_address"] == "").sum()
    n = len(df)
    print(f"{label}: name_missing={name_miss} ({100*name_miss/n:.1f}%), addr_missing={addr_miss} ({100*addr_miss/n:.1f}%)")

# ── SECTION 6: France-specific analysis ───────────────────────────────
print("\n--- SECTION 6: FRANCE ANALYSIS ---")
s1_fr = s1_test[s1_test["country"]=="France"]
s2_fr = s2_test[s2_test["country"]=="France"]
s3_fr = s3_test[s3_test["country"]=="France"]
print(f"France S1: {len(s1_fr):,} ({100*len(s1_fr)/len(s1_test):.1f}% of test S1)")
print(f"France S2: {len(s2_fr):,}")
print(f"France S3: {len(s3_fr):,}")

fr_sub = sub[sub["source1_entity_id"].isin(s1_fr["entity_id"])]
fr_name_miss = (s1_fr["business_name"]=="").sum()
fr_addr_miss = (s1_fr["business_address"]=="").sum()
print(f"France S1 name missing: {fr_name_miss} ({100*fr_name_miss/max(len(s1_fr),1):.1f}%)")
print(f"France S1 addr missing: {fr_addr_miss} ({100*fr_addr_miss/max(len(s1_fr),1):.1f}%)")
print(f"France sub predictions: {fr_sub['n_pred'].sum():,} total, avg={fr_sub['n_pred'].mean():.2f}/entity")
print(f"France predicted singletons: {(fr_sub['n_pred']==0).sum():,} ({100*(fr_sub['n_pred']==0).sum()/max(len(fr_sub),1):.1f}%)")

# ── SECTION 7: Sample address/name examples France ────────────────────
print("\n[France S1 sample names/addresses]")
print(s1_fr[["entity_id","business_name","business_address"]].head(5).to_string())

# ── SECTION 8: US/India by country ────────────────────────────────────
print("\n--- SECTION 8: PREDICTION BREAKDOWN BY COUNTRY ---")
s1_test_c = s1_test.set_index("entity_id")["country"].to_dict()
sub["country"] = sub["source1_entity_id"].map(s1_test_c)
for ctry in ["France", "US", "India"]:
    sub_c = sub[sub["country"]==ctry]
    print(f"  {ctry}: {len(sub_c):,} entities, {sub_c['n_pred'].sum():,} predictions, "
          f"avg={sub_c['n_pred'].mean():.2f}, singletons={100*(sub_c['n_pred']==0).sum()/max(len(sub_c),1):.1f}%")

# ── SECTION 9: Token/name length distribution ─────────────────────────
print("\n--- SECTION 9: NAME/ADDRESS LENGTH COMPARISON ---")
for label, df in [("TRAIN S1", s1_train), ("TEST S1", s1_test)]:
    names = df["business_name"][df["business_name"]!=""]
    addrs = df["business_address"][df["business_address"]!=""]
    print(f"{label}: name_len_mean={names.str.len().mean():.1f}, addr_len_mean={addrs.str.len().mean():.1f}")
    
# ── SECTION 10: Check for Unicode/diacritics in France ────────────────
print("\n--- SECTION 10: UNICODE/DIACRITICS IN FRANCE ---")
def has_diacritic(s):
    return any(ord(c) > 0x007F for c in str(s))

fr_names = s1_fr["business_name"]
diac_count = fr_names.apply(has_diacritic).sum()
print(f"France S1 names with non-ASCII chars: {diac_count}/{len(fr_names)} ({100*diac_count/max(len(fr_names),1):.1f}%)")

# Sample of French names with diacritics
diac_examples = s1_fr[s1_fr["business_name"].apply(has_diacritic)]["business_name"].head(5).tolist()
print(f"Examples: {diac_examples}")

# ── SECTION 11: Postal code availability ─────────────────────────────
print("\n--- SECTION 11: POSTAL CODE AVAILABILITY ---")
def extract_postal(addr, country):
    if country == "India":
        m = re.search(r'\b[1-9]\d{5}\b', str(addr))
    elif country == "US":
        m = re.search(r'\b\d{5}(?:-\d{4})?\b', str(addr))
    elif country == "France":
        m = re.search(r'\b\d{5}\b', str(addr))
    else:
        m = re.search(r'\b\d{4,6}\b', str(addr))
    return m.group() if m else ""

for ctry in ["France", "US", "India"]:
    df_c = s1_test[s1_test["country"]==ctry]
    postal_count = df_c.apply(lambda r: bool(extract_postal(r["business_address"], ctry)), axis=1).sum()
    print(f"  {ctry}: {postal_count}/{len(df_c)} have extractable postal ({100*postal_count/max(len(df_c),1):.1f}%)")

print("\n" + "=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)

import pandas as pd
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    gs_path = ROOT / "data/official/wdc/80pair/wdcproducts80cc20rnd000un_gs.json.gz"
    train_path = ROOT / "data/official/wdc/80pair/wdcproducts80cc20rnd000un_train_large.json.gz"
    val_path = ROOT / "data/official/wdc/80pair/wdcproducts80cc20rnd000un_valid_large.json.gz"
    
    gs_df = pd.read_json(gs_path, lines=True)
    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)
    
    print("Dataset Version: WDC Products (80pair, 20cc, 000un)")
    print(f"GS Pairs: {len(gs_df)}")
    print(f"Train Pairs: {len(train_df)}")
    print(f"Val Pairs: {len(val_df)}")
    
    # Calculate unique entities/records
    def get_unique_records(df):
        left_cols = ['id_left', 'title_left', 'brand_left', 'description_left', 'price_left', 'priceCurrency_left', 'cluster_id_left']
        right_cols = ['id_right', 'title_right', 'brand_right', 'description_right', 'price_right', 'priceCurrency_right', 'cluster_id_right']
        
        left_df = df[left_cols].rename(columns=lambda x: x.replace('_left', ''))
        right_df = df[right_cols].rename(columns=lambda x: x.replace('_right', ''))
        
        return pd.concat([left_df, right_df]).drop_duplicates(subset=['id'])
        
    gs_records = get_unique_records(gs_df)
    train_records = get_unique_records(train_df)
    val_records = get_unique_records(val_df)
    
    print(f"GS Unique Records: {len(gs_records)}")
    print(f"GS Unique Clusters: {gs_records['cluster_id'].nunique()}")
    print(f"GS Positives: {gs_df['label'].sum()} / {len(gs_df)}")
    
    # Check overlap
    gs_ids = set(gs_records['id'])
    train_ids = set(train_records['id'])
    val_ids = set(val_records['id'])
    
    gs_clusters = set(gs_records['cluster_id'])
    train_clusters = set(train_records['cluster_id'])
    val_clusters = set(val_records['cluster_id'])
    
    print(f"Train-Val Record Overlap: {len(train_ids & val_ids)}")
    print(f"Train-GS Record Overlap: {len(train_ids & gs_ids)}")
    print(f"Train-Val Cluster Overlap: {len(train_clusters & val_clusters)}")
    print(f"Train-GS Cluster Overlap: {len(train_clusters & gs_clusters)}")
    
    # Missingness in GS
    print("GS Missingness:")
    print(gs_records.isna().sum() / len(gs_records))
    
    # Schema check
    print("Schema:")
    for col in gs_records.columns:
        print(f"- {col}: {gs_records[col].dtype}")
        
    # Matches per source record
    # Actually, how many times does an id appear in pairs?
    left_counts = gs_df.groupby('id_left')['label'].sum()
    right_counts = gs_df.groupby('id_right')['label'].sum()
    
    # Combine counts (number of positive matches per record)
    all_pos_counts = pd.concat([left_counts, right_counts]).groupby(level=0).sum()
    print("Positive matches per record distribution in GS:")
    print(all_pos_counts.value_counts().sort_index())
    
    # Candidate space
    # Total possible pairs in GS:
    total_possible = len(gs_records) * (len(gs_records) - 1) // 2
    print(f"Total possible GS pairs: {total_possible}")
    print(f"Provided GS candidate pairs: {len(gs_df)}")
    print(f"Reduction Ratio: {total_possible / len(gs_df) if len(gs_df) > 0 else 0}")
    
    
if __name__ == "__main__":
    main()

import pandas as pd
import json
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent

def main():
    multi_dir = ROOT / "data/official/wdc/80multi"
    
    gs_path = list(multi_dir.glob("**/*_gs.json.gz"))[0]
    train_path = list(multi_dir.glob("**/*_train_large.json.gz"))[0]
    val_path = list(multi_dir.glob("**/*_valid_large.json.gz"))[0]
    
    gs_df = pd.read_json(gs_path, lines=True)
    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)
    
    print("Dataset Version: WDC Products (80multi)")
    print(f"GS Records: {len(gs_df)}")
    print(f"Train Records: {len(train_df)}")
    print(f"Val Records: {len(val_df)}")
    
    print(f"GS Unique Clusters: {gs_df['cluster_id'].nunique()}")
    print(f"Train Unique Clusters: {train_df['cluster_id'].nunique()}")
    print(f"Val Unique Clusters: {val_df['cluster_id'].nunique()}")
    
    # Overlap
    train_clusters = set(train_df['cluster_id'])
    val_clusters = set(val_df['cluster_id'])
    gs_clusters = set(gs_df['cluster_id'])
    
    print(f"Train-Val Cluster Overlap: {len(train_clusters & val_clusters)}")
    print(f"Train-GS Cluster Overlap: {len(train_clusters & gs_clusters)}")
    print(f"Val-GS Cluster Overlap: {len(val_clusters & gs_clusters)}")
    
    # Unseen labels
    if 'unseen' in gs_df.columns:
        print(f"GS Unseen records (entities not in train): {gs_df['unseen'].sum()}")
        
    print("GS Missingness:")
    print(gs_df.isna().sum() / len(gs_df))
    
    print("Schema:")
    for col in gs_df.columns:
        print(f"- {col}: {gs_df[col].dtype}")
        
    print("Matches per entity (cluster) in GS:")
    cluster_counts = gs_df['cluster_id'].value_counts()
    print(cluster_counts.value_counts().sort_index().head(15))
    
    print("Max matches in a cluster:", cluster_counts.max())
    
    # Ambiguity check
    print("Missingness overall:")
    print(pd.concat([train_df, val_df, gs_df]).isna().sum() / (len(train_df)+len(val_df)+len(gs_df)))
    
    print("Done")

if __name__ == "__main__":
    main()

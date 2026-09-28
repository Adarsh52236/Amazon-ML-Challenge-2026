import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    out_dir = ROOT / "output/benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data = [
        {"Metric": "wdc_dataset_version", "Value": "WDC Products (80pair, 20cc, 000un) & 80multi"},
        {"Metric": "gs_unique_records", "Value": 1000},
        {"Metric": "gs_unique_clusters", "Value": 500},
        {"Metric": "train_unique_records_multi", "Value": 2841},
        {"Metric": "val_unique_records_multi", "Value": 1000},
        {"Metric": "gs_pairs", "Value": 4500},
        {"Metric": "train_pairs", "Value": 19835},
        {"Metric": "val_pairs", "Value": 4500},
        {"Metric": "gs_positive_pairs", "Value": 500},
        {"Metric": "train_positive_pairs", "Value": "Various (multi)"},
        {"Metric": "gs_max_matches_per_record", "Value": 1},
        {"Metric": "train_max_matches_per_cluster", "Value": 11},
    ]
    
    pd.DataFrame(data).to_csv(out_dir / "wdc_products_dataset_audit.tsv", sep="\t", index=False)
    
if __name__ == "__main__":
    main()

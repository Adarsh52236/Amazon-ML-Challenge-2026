import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    multi_dir = ROOT / "data/official/wdc/80multi"
    
    variants = [
        "wdcproductsmulti80cc20rnd000un",
        "wdcproductsmulti80cc20rnd050un",
        "wdcproductsmulti80cc20rnd100un"
    ]
    
    results = []
    
    for var in variants:
        try:
            gs_path = list(multi_dir.glob(f"*{var}_gs.json.gz"))[0]
            gs_df = pd.read_json(gs_path, lines=True)
            
            # Since 050un and 100un don't have their own train sets in the zip, they use the 000un train set or don't have one?
            # Actually they are test sets for the same model!
            
            rec = {
                "variant": var,
                "pair_or_multi": "multi",
                "unseen_rate": var[-5:-2],
                "test_records": len(gs_df),
                "test_entities": gs_df['cluster_id'].nunique(),
                "mean_cluster_size": gs_df.groupby('cluster_id').size().mean(),
                "max_cluster_size": gs_df.groupby('cluster_id').size().max(),
            }
            results.append(rec)
            
        except Exception as e:
            print(f"Error processing {var}: {e}")
            
    print(pd.DataFrame(results).to_string())

if __name__ == "__main__":
    main()

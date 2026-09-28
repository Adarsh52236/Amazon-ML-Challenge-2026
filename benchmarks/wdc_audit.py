import os
import sys
import json
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def download_and_extract(url, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "temp.zip"
    if not zip_path.exists():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
        
def main():
    val_pair_url = 'https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80pair.zip'
    
    out_dir = ROOT / "data" / "official" / "wdc"
    download_and_extract(val_pair_url, out_dir / "80pair")
    
    # Audit files
    print("Files downloaded.")
    for p in (out_dir / "val_pair").glob("**/*"):
        if p.is_file():
            print(f"File: {p.relative_to(out_dir)}, Size: {p.stat().st_size} bytes")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI Script to inspect the official Business Entity Resolution dataset."""

import sys
from pathlib import Path

# Add src to python path if run directly from repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.dataset_inspector import main_cli

if __name__ == "__main__":
    main_cli()

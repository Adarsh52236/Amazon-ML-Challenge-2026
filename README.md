# Business Entity Resolution — ML Challenge 2026

An enterprise-grade, reproducible machine learning system for multi-source Business Entity Resolution (ER).

The objective is to resolve noisy, heterogeneous business records from **Source 2** and **Source 3** to a deduplicated reference collection (**Source 1**) across the US, India, and France.

---

## 1. Challenge Overview

In commercial platforms, business identity data originates from multiple independent data sources with noisy and inconsistent fields. These fragments share no common natural keys or identifiers.

- **Reference Source:** **Source 1** (deduplicated reference collection).
- **Secondary Sources:** **Source 2** and **Source 3**.
- **Matching Cardinality:** A Source 1 entity may link to **0, 1, or multiple** records across Source 2 and Source 3.
- **Evaluation Metric:** Macro-averaged **$F_{0.5}$ score** across all Source 1 entities:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  $F_{0.5}$ weights precision $2\times$ over recall to heavily penalize false merges (combining two distinct businesses).
- **Singletons:** Source 1 entities with zero true matches score $1.0$ when correctly predicted with an empty match list and $0.0$ if any match is predicted.

---

## 2. Official Dataset & Setup

> **IMPORTANT:** The official challenge dataset is **NEVER** committed to GitHub or version control. Each collaborator must configure their own local path to the raw dataset. No synthetic competition datasets are generated.

### Dataset Directory Layout

The official dataset directory (`student_resource/`) contains:

```
student_resource/
├── README.md
├── Documentation_template.md
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv       # 2,206,821 rows (Reference source)
│   │   ├── train_source2.tsv       # 5,034,616 rows
│   │   ├── train_source3.tsv       # 5,285,603 rows
│   │   └── train_ground_truth.tsv  # 2,206,821 rows (Matches S1 -> S2/S3)
│   └── test/
│       ├── test_source1.tsv        # 1,732,544 rows (Evaluation S1 entities)
│       ├── test_source2.tsv        # 4,887,273 rows
│       └── test_source3.tsv        # 5,082,316 rows
└── utils/
    └── validate_submission.py      # Official stdlib submission format checker
```

### Configurable Dataset Location

Path resolution follows a strict priority:
1. Explicit function argument (`override_path`)
2. Environment variable: `BUSINESS_ENTITY_DATASET_ROOT`
3. `.env` file configuration
4. Default developer path:
   `C:\Users\adars\Downloads\business_entity_resolution\student_resource`
5. Repository fallback: `data/official/`

To set your local path, copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

And update `BUSINESS_ENTITY_DATASET_ROOT`:

```ini
BUSINESS_ENTITY_DATASET_ROOT=C:\path\to\your\student_resource
```

---

## 3. Project Structure

```
business_entity_resolution/
│
├── README.md                          # Main project guide
├── requirements.txt                   # Pinned core dependencies
├── pyproject.toml                     # Package definition & pytest config
├── .gitignore                         # Strict exclusion of raw/processed data
├── .env.example                       # Environment template
├── CONTRIBUTING.md                    # Git workflow & team guidelines
│
├── data/
│   ├── README.md                      # Data usage guidelines
│   ├── official/
│   │   └── README.md                  # Instructions for local dataset placement
│   └── processed/
│       └── .gitkeep                   # Intermediate artifacts (ignored by git)
│
├── notebooks/
│   ├── README.md                      # Exploratory guidelines
│   └── 01_dataset_inspection.ipynb    # Interactive dataset inspection & EDA
│
├── src/
│   └── business_entity_resolution/
│       ├── __init__.py                # Package exports
│       ├── config.py                  # Centralized path & environment config
│       ├── utils.py                   # String parsing, tree, & formatting utils
│       ├── data_loader.py             # Memory-efficient tabular & GT loaders
│       ├── dataset_inspector.py       # Comprehensive streaming inspector
│       └── data_validator.py          # Strict data quality & reference validator
│
├── scripts/
│   ├── inspect_dataset.py             # CLI runner for dataset inspection
│   └── validate_dataset.py            # CLI runner for dataset validation
│
├── tests/
│   ├── __init__.py
│   ├── test_config.py                 # Configuration unit tests
│   ├── test_data_loader.py            # Data loader unit tests
│   ├── test_data_validator.py         # Validator unit tests
│   └── test_dataset_inspector.py      # Inspector unit tests
│
├── experiments/
│   └── README.md                      # Experiment tracking log
│
├── documentation/
│   ├── README.md
│   ├── Documentation_template.md      # Official challenge methodology template
│   └── daily/
│       └── day_1.md                   # Day 1 engineering record & findings
│
└── output/
    ├── .gitkeep
    └── dataset_inspection_report.md   # Generated dataset inspection report
```

---

## 4. Getting Started & Installation

### Requirements
- Python 3.10+ (tested with Python 3.11)
- Windows / Linux / macOS compatible

### Environment Setup

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install project in editable mode
pip install -e .
```

---

## 5. CLI Commands

### Dataset Inspection
Inspect file dimensions, column schemas, ID uniqueness, string lengths, and singletons:

```bash
python scripts/inspect_dataset.py --output output/dataset_inspection_report.md
```

### Dataset Validation
Verify file integrity, required columns, string ID retention, and cross-source ground truth references:

```bash
python scripts/validate_dataset.py
```

Options:
- `--strict`: Treats warnings as non-zero exit errors.
- `--skip-id-cross-refs`: Skips memory-heavy full ID set cross-referencing for quick checks.

### Running Automated Tests
Run unit tests across configuration, loading, validation, and inspection using temporary fixtures:

```bash
pytest -v
```

---

## 6. Official Dataset Verified Schemas

### Source Files (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`)
| Column | Type | Preserved Format | Description |
|---|---|---|---|
| `entity_id` | String | String (`S1-`, `S2-`, `S3-`) | Unique entity record identifier |
| `business_name` | String | Raw string | Business trade or legal name |
| `business_address` | String | Raw string | Address text (empty for ~3% in S2/S3) |
| `country` | String | String (`US`, `India`, `France`) | Country code |

### Ground Truth File (`train_ground_truth.tsv`)
| Column | Type | Description |
|---|---|---|
| `source1_entity_id` | String | Reference Source 1 entity ID |
| `matched_entity_ids` | String | Comma-separated matched IDs from S2 and S3 (empty string for singletons) |

---

## 7. What Has Been Implemented (Phase 1)
- [x] Full inspection of official raw dataset (all 7 TSV files, 26.4M+ total rows).
- [x] Centralized, environment-aware configuration with Windows path support (`config.py`).
- [x] Memory-efficient streaming data loading with string ID preservation (`data_loader.py`).
- [x] Deep streaming dataset inspector producing markdown and JSON reports (`dataset_inspector.py`).
- [x] Rigorous schema, ID uniqueness, and cross-table reference validator (`data_validator.py`).
- [x] CLI entry points (`scripts/inspect_dataset.py`, `scripts/validate_dataset.py`).
- [x] Exploratory Jupyter notebook (`notebooks/01_dataset_inspection.ipynb`).
- [x] 15 comprehensive unit tests (all passing).
- [x] Documentation, Git collaboration guidelines, and Day 1 log.

---

## 8. Upcoming Phases
- **Phase 2:** Text normalization, address component extraction, and country-specific preprocessing (US, India, France).
- **Phase 3:** Candidate generation & blocking architecture (TF-IDF, phonetic blocking, recall ceiling analysis).
- **Phase 4:** Feature engineering & similarity scoring (Jaccard, Levenshtein, token overlap).
- **Phase 5:** Classification model training & $F_{0.5}$ threshold optimization.
- **Phase 6:** End-to-end test inference and official validation via `utils/validate_submission.py`.

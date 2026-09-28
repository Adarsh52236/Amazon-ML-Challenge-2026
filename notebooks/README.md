# Notebooks Directory

This directory contains Jupyter notebooks for exploratory data analysis, prototype validation, and visual data inspection.

## Principles
1. **Exploration Only**: Essential logic, schemas, and production code belong in `src/business_entity_resolution/`, not in notebooks.
2. **Reproducibility**: Notebooks should import package modules (`from business_entity_resolution import ...`) rather than copy-pasting raw scripts.
3. **No Bloat**: Notebooks should not commit multi-gigabyte outputs or raw data tables.

## Available Notebooks
- `01_dataset_inspection.ipynb`: Initial inspection of official raw datasets, schemas, singletons, and distributions.

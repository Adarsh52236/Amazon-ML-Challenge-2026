# Data Directory

This directory is intended for local datasets used in the Business Entity Resolution project.

## Directory Structure

```
data/
├── official/     # Symlink or location for the official competition dataset (ignored by git)
└── processed/    # Preprocessed, cached, or intermediate feature data (ignored by git)
```

## Official Dataset Policy
- **Never commit raw or processed datasets to Git**. All data files are ignored in `.gitignore`.
- Set the `BUSINESS_ENTITY_DATASET_ROOT` environment variable in your `.env` file to point to your local copy of `student_resource` or `dataset/`.
- Default local path configured in `src/business_entity_resolution/config.py`:
  `C:\Users\adars\Downloads\business_entity_resolution\student_resource`

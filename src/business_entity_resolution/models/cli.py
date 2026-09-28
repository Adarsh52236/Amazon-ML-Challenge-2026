"""CLI entrypoint for training and evaluating matching models."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from business_entity_resolution.models.label_builder import LabelBuilder
from business_entity_resolution.models.model_config import ModelConfig
from business_entity_resolution.models.train import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    features_path: str = "data/processed/features_dev_cap100.parquet",
    output_dir: str = "output",
    model_choice: str = "all",
) -> None:
    """Run model training, cross-validation, and metric reporting."""
    p_feat = Path(features_path)
    if not p_feat.is_file():
        logger.error(f"Features file not found at: {features_path}")
        sys.exit(1)

    logger.info(f"Loading features from: {features_path}")
    df_features = pd.read_parquet(p_feat)
    logger.info(f"Loaded features with shape: {df_features.shape}")

    builder = LabelBuilder()
    labeled_dataset = builder.build_labeled_dataset(df_features, strict_audit=True)
    logger.info(
        f"Labeled dataset: {labeled_dataset.n_samples:,} rows "
        f"({labeled_dataset.n_positives:,} pos, {labeled_dataset.n_negatives:,} neg, "
        f"{labeled_dataset.n_features} features)"
    )

    trainer = ModelTrainer()
    results = {}

    if model_choice in ("lr", "all"):
        logger.info("Training Logistic Regression...")
        lr_res, lr_meta = trainer.train_logistic_regression(labeled_dataset)
        results["logistic_regression"] = {
            "pr_auc": lr_res.pr_auc,
            "roc_auc": lr_res.roc_auc,
            "runtime_s": lr_res.training_runtime_seconds,
            "best_cand_thresh": lr_res.best_candidate_threshold,
            "best_cand_f1": lr_res.best_candidate_f1,
            "best_comp_thresh": lr_res.best_competition_threshold,
            "best_comp_f1": lr_res.best_competition_f1,
        }
        logger.info(
            f"Logistic Regression: Cand F1={lr_res.best_candidate_f1*100:.2f}%, "
            f"Comp F1={lr_res.best_competition_f1*100:.2f}%, PR-AUC={lr_res.pr_auc:.4f}"
        )

    if model_choice in ("lgb", "all"):
        logger.info("Training LightGBM...")
        lgb_res, lgb_meta = trainer.train_lightgbm(labeled_dataset)
        results["lightgbm"] = {
            "pr_auc": lgb_res.pr_auc,
            "roc_auc": lgb_res.roc_auc,
            "runtime_s": lgb_res.training_runtime_seconds,
            "best_cand_thresh": lgb_res.best_candidate_threshold,
            "best_cand_f1": lgb_res.best_candidate_f1,
            "best_comp_thresh": lgb_res.best_competition_threshold,
            "best_comp_f1": lgb_res.best_competition_f1,
        }
        logger.info(
            f"LightGBM: Cand F1={lgb_res.best_candidate_f1*100:.2f}%, "
            f"Comp F1={lgb_res.best_competition_f1*100:.2f}%, PR-AUC={lgb_res.pr_auc:.4f}"
        )

    if model_choice in ("xgb", "all"):
        logger.info("Training XGBoost...")
        xgb_res, xgb_meta = trainer.train_xgboost(labeled_dataset)
        results["xgboost"] = {
            "pr_auc": xgb_res.pr_auc,
            "roc_auc": xgb_res.roc_auc,
            "runtime_s": xgb_res.training_runtime_seconds,
            "best_cand_thresh": xgb_res.best_candidate_threshold,
            "best_cand_f1": xgb_res.best_candidate_f1,
            "best_comp_thresh": xgb_res.best_competition_threshold,
            "best_comp_f1": xgb_res.best_competition_f1,
        }
        logger.info(
            f"XGBoost: Cand F1={xgb_res.best_candidate_f1*100:.2f}%, "
            f"Comp F1={xgb_res.best_competition_f1*100:.2f}%, PR-AUC={xgb_res.pr_auc:.4f}"
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(output_dir) / "phase_4_cli_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 Matching Model Training & Validation CLI"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/features_dev_cap100.parquet",
        help="Path to input features Parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to store evaluation outputs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["lr", "lgb", "xgb", "all"],
        help="Which model(s) to train and evaluate",
    )

    args = parser.parse_args()
    run_pipeline(
        features_path=args.features,
        output_dir=args.output_dir,
        model_choice=args.model,
    )


if __name__ == "__main__":
    main()

"""Business Entity Resolution Package.

A clean, modular, and competition-ready framework for multi-source
business entity resolution.
"""

__version__ = "0.1.0"

from business_entity_resolution.config import (
    get_dataset_root,
    get_train_dir,
    get_test_dir,
    get_source_path,
    get_ground_truth_path,
)
from business_entity_resolution.data_loader import (
    load_tabular_file,
    load_source_file,
    load_ground_truth,
    discover_dataset_files,
    load_training_data,
    load_test_data,
)
from business_entity_resolution.dataset_inspector import (
    inspect_file,
    inspect_dataset,
    generate_inspection_report,
)
from business_entity_resolution.data_validator import (
    validate_dataset,
    ValidationResult,
    ValidationStatus,
)
from business_entity_resolution.preprocessing_config import (
    TextNormalizerConfig,
    AddressNormalizerConfig,
    BlockingConfig,
    DevSplitConfig,
)
from business_entity_resolution.text_normalizer import (
    BusinessNameNormalizer,
    NormalizedBusinessName,
)
from business_entity_resolution.address_normalizer import (
    BusinessAddressNormalizer,
    NormalizedBusinessAddress,
)
from business_entity_resolution.blocking_keys import generate_blocking_keys
from business_entity_resolution.candidate_generator import (
    CandidateGenerator,
    CandidatePair,
)
from business_entity_resolution.candidate_evaluator import (
    CandidateEvaluationReport,
    evaluate_candidates,
)

__all__ = [
    "__version__",
    "get_dataset_root",
    "get_train_dir",
    "get_test_dir",
    "get_source_path",
    "get_ground_truth_path",
    "load_tabular_file",
    "load_source_file",
    "load_ground_truth",
    "discover_dataset_files",
    "load_training_data",
    "load_test_data",
    "inspect_file",
    "inspect_dataset",
    "generate_inspection_report",
    "validate_dataset",
    "ValidationResult",
    "ValidationStatus",
    "TextNormalizerConfig",
    "AddressNormalizerConfig",
    "BlockingConfig",
    "DevSplitConfig",
    "BusinessNameNormalizer",
    "NormalizedBusinessName",
    "BusinessAddressNormalizer",
    "NormalizedBusinessAddress",
    "generate_blocking_keys",
    "CandidateGenerator",
    "CandidatePair",
    "CandidateEvaluationReport",
    "evaluate_candidates",
]

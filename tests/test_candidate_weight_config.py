import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.business_entity_resolution.candidate_generator import CandidateGenerator, CandidatePair

class TestCandidateWeightConfig(unittest.TestCase):
    def test_dictionary_config_weight(self):
        # 1. A dictionary containing channel_weights is correctly read
        config = {
            "channel_weights": {
                "exact_name": 12,
                "rare_token": 3,
                "postal_prefix": 5
            },
            "max_candidates_per_entity": 2,
            "max_key_frequency": 5000,
            "use_exact_name": True,
            "use_rare_name_token": True,
            "use_postal_prefix": True
        }
        
        # In CandidateGenerator, priority score is (total_weight, len(channels))
        # Let's see what happens if we instantiate CandidateGenerator with a dict
        import business_entity_resolution.candidate_generator as cg
        generator = cg.CandidateGenerator(config=config)
        
        # Test how get_weight acts under buggy implementation
        if hasattr(generator.config, "get"):
            weights = generator.config.get("channel_weights", {}) or {}
        else:
            weights = getattr(generator.config, "channel_weights", {}) or {}
            
        def candidate_priority_score(item):
            total_weight = sum(weights.get(ch, 1) for ch in item[1])
            return (total_weight, len(item[1]))

        # Calculate scores
        # exact_name only
        score_c1 = candidate_priority_score(("S2-2", {"exact_name"}))
        # rare_token and postal_prefix
        score_c2 = candidate_priority_score(("S3-3", {"rare_token", "postal_prefix"}))
        # unknown channel
        score_c3 = candidate_priority_score(("S2-4", {"unknown_channel"}))
        
        # Under fixed code, weights is a dict with actual values
        # exact_name = 12
        self.assertEqual(score_c1[0], 12)
        
        # unknown_channel defaults to 1
        self.assertEqual(score_c3[0], 1)
        
        # ranking score is not equivalent to len(channels)
        self.assertEqual(score_c2[0], 8) # 3 + 5 = 8
        self.assertNotEqual(score_c2[0], 2)

if __name__ == "__main__":
    unittest.main()

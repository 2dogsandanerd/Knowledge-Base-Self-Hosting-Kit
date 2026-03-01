"""
Experiment Service (Phase I.2).

Handles A/B testing logic and user assignment to experiment variants.
"""

import hashlib
from typing import Optional
from loguru import logger

class ExperimentService:
    """
    Service for managing A/B tests.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="ExperimentService")
        self.experiments = {
            # Example experiment config
            "use_advanced_rag": {
                "variants": ["A", "B"],
                "weights": [0.5, 0.5], # 50% A (Control), 50% B (Test)
                "enabled": True
            },
            # Experiment for Reranker A/B testing
            "use_reranker_experiment": {
                "variants": ["A", "B"],
                "weights": [0.5, 0.5], # 50% A (No Rerank), 50% B (Rerank)
                "enabled": True
            }
        }

    def get_variant(self, user_id: str, experiment_name: str) -> str:
        """
        Get the variant for a specific user and experiment.
        Deterministic assignment based on hash.
        """
        if experiment_name not in self.experiments:
            return "A" # Default to control if experiment unknown
            
        exp_config = self.experiments[experiment_name]
        if not exp_config["enabled"]:
            return "A"
            
        # Create deterministic hash
        hash_input = f"{user_id}:{experiment_name}"
        hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        
        # Simple modulo for 2 variants (can be extended for weighted)
        # For now, assumes equal weights for simplicity
        variant_index = hash_val % len(exp_config["variants"])
        variant = exp_config["variants"][variant_index]
        
        self.logger.debug(f"Assigned user {user_id} to variant {variant} for {experiment_name}")
        return variant

# Singleton
experiment_service = ExperimentService()

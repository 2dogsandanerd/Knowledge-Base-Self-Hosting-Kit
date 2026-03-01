
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.services.data_classifier_service import DataClassifierService
from src.core.ingest_config import IngestConfig

logger = logging.getLogger(__name__)

class DocumentRouterService:
    def __init__(self, data_classifier_service: DataClassifierService):
        self.classifier = data_classifier_service
        self.rules_path = Path(__file__).parent.parent / "core" / "document_routing_rules.json"
        self.routing_rules = self._load_routing_rules()
    
    def _load_routing_rules(self) -> Dict[str, Any]:
        """Loads routing rules from configuration files"""
        try:
            if not self.rules_path.exists():
                logger.warning(f"Routing rules file not found at {self.rules_path}. Using defaults.")
                return {"rules": [], "defaults": {}}
            
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load routing rules: {e}")
            return {"rules": [], "defaults": {}}
    
    async def route_document(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method for intelligent routing.
        Performs classification and returns a routing decision.
        """
        try:
            # 1. Classify
            # We wrap single file in a list as analyze_folder_contents expects list logic or folder
            # But the service takes a folder path.
            # We can use analyze_folder_contents on the parent dir, but that's inefficient if we just want one file.
            # The DataClassifierService doesn't expose a single-file public method except internal ones.
            # However, looking at DataClassifierService, it has analyze_folder_contents.
            # We can use the internal methods if we are careful, or better, we ask DataClassifierService to support single file?
            # For now, we utilize the _heuristic_classify and _classify_with_llm flow directly or simulate a folder scan.
            # Simulation:
            
            # Replicating the logic from DataClassifierService.analyze_folder_contents for a single file
            # to avoid scanning the whole folder.
            
            # 1. Heuristic
            heuristic_result = self.classifier._heuristic_classify(file_path)
            
            # 2. LLM (using preview)
            file_preview = self.classifier._get_file_preview(file_path)
            llm_result = await self.classifier._classify_with_llm(file_path, file_preview, heuristic_result)
            
            # Combine results (LLM overrides heuristic typically, but we have both data points)
            classification_result = {
                "file_path": file_path,
                "filename": metadata.get("filename", Path(file_path).name),
                **llm_result
            }
            
            # 3. Make Decision
            logger.info(f"Classification result for {metadata.get('filename')}: {classification_result.get('recommended_collection')} (Conf: {classification_result.get('confidence')})")
            
            routing_decision = self._make_routing_decision(classification_result)
            
            return routing_decision

        except Exception as e:
            logger.error(f"Error routing document {file_path}: {e}")
            # Return safe default
            return {
                "target_collection": "generic_errors",
                "processing_params": self.routing_rules.get("defaults", {}),
                "error": str(e)
            }
    
    def _make_routing_decision(self, classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Makes routing decision based on classification result and rules.
        """
        category = classification_result.get("recommended_collection", "generic")
        confidence = classification_result.get("confidence", 0.0)
        
        # Check explicit rules
        for rule in self.routing_rules.get("rules", []):
            condition = rule.get("condition", {})
            req_category = condition.get("category")
            req_conf_dict = condition.get("confidence", {})
            
            # Check Category match
            if req_category and req_category != category:
                continue
                
            # Check Confidence Threshold
            # format: {">=": 0.7}
            if ">=" in req_conf_dict:
                threshold = req_conf_dict[">="]
                if confidence < threshold:
                    continue
            
            # Match found!
            action = rule.get("action", {})
            
            # Construct decision
            decision = {
                "document_type": category,
                "confidence": confidence,
                "target_collection": action.get("target_collection", category),
                "requires_validation": action.get("requires_validation", False),
                "processing_params": {
                    "chunk_size": action.get("chunk_size", 512),
                    "chunk_overlap": action.get("chunk_overlap", 128),
                    "embedding_model": action.get("embedding_model", "nomic-embed-text:latest"),
                    "preprocessing_steps": action.get("preprocessing_steps", []),
                    "postprocessing_steps": action.get("postprocessing_steps", [])
                },
                "rule_matched": True
            }
            return decision

        # No rule matched -> Use Defaults logic
        # But we leverage the LLM suggestions if no specific rule overrides them
        defaults = self.routing_rules.get("defaults", {})
        
        # Use LLM suggested chunk size if reasonable, otherwise default
        llm_chunk_size = classification_result.get("suggested_chunk_size")
        final_chunk_size = llm_chunk_size if isinstance(llm_chunk_size, int) else defaults.get("chunk_size", 512)
        
        decision = {
            "document_type": category,
            "confidence": confidence,
            "target_collection": category, # Default to the classified category
            "requires_validation": defaults.get("requires_validation", False),
            "processing_params": {
                "chunk_size": final_chunk_size,
                "chunk_overlap": defaults.get("chunk_overlap", 128),
                "embedding_model": classification_result.get("suggested_embedding_model", defaults.get("embedding_model", "nomic-embed-text:latest")),
                "preprocessing_steps": defaults.get("preprocessing_steps", []),
                "postprocessing_steps": defaults.get("postprocessing_steps", [])
            },
            "rule_matched": False
        }
        return decision

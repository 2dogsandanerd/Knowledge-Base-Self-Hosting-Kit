
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os
import asyncio
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../backend")))

# MOCK HEAVY DEPENDENCIES
sys.modules["src.core.llm_singleton"] = MagicMock()
sys.modules["src.core.config"] = MagicMock()
sys.modules["src.core.ingest_config"] = MagicMock() # Mock if needed

from src.services.document_router_service import DocumentRouterService

class TestDocumentRouterService(unittest.TestCase):
    def setUp(self):
        self.mock_classifier = MagicMock()
        self.service = DocumentRouterService(self.mock_classifier)
        
        # Override rules with a known set for testing to avoid dependency on external file correctness
        self.service.routing_rules = {
            "rules": [
                {
                    "condition": {
                        "category": "legal_documents",
                        "confidence": {">=": 0.7}
                    },
                    "action": {
                        "target_collection": "contracts",
                        "chunk_size": 1024,
                        "requires_validation": True
                    }
                }
            ],
            "defaults": {
                "chunk_size": 500,
                "target_collection": "default_coll"
            }
        }

    def test_route_legal_high_confidence(self):
        """Test that legal document with high confidence hits the rule"""
        async def run_test():
            # Setup classifier mock
            self.mock_classifier._heuristic_classify.return_value = {"category": "generic"}
            self.mock_classifier._get_file_preview.return_value = "preview"
            self.mock_classifier._classify_with_llm = AsyncMock(return_value={
                "recommended_collection": "legal_documents",
                "confidence": 0.9,
                "suggested_chunk_size": 512 # Should be overridden by rule
            })
            
            decision = await self.service.route_document("/tmp/contract.pdf", {"filename": "contract.pdf"})
            
            self.assertTrue(decision["rule_matched"])
            self.assertEqual(decision["target_collection"], "contracts")
            self.assertEqual(decision["processing_params"]["chunk_size"], 1024)
            self.assertTrue(decision["requires_validation"])

        asyncio.run(run_test())

    def test_route_legal_low_confidence(self):
        """Test that legal document with low confidence MISSES the rule"""
        async def run_test():
            self.mock_classifier._heuristic_classify.return_value = {}
            self.mock_classifier._get_file_preview.return_value = ""
            self.mock_classifier._classify_with_llm = AsyncMock(return_value={
                "recommended_collection": "legal_documents",
                "confidence": 0.5, # Below 0.7 threshold
                "suggested_chunk_size": 1234
            })
            
            decision = await self.service.route_document("/tmp/contract.pdf", {})
            
            self.assertFalse(decision["rule_matched"])
            # Should fall back to category or default?
            # Code says: "target_collection": category (default to classified category)
            self.assertEqual(decision["target_collection"], "legal_documents")
            # Chunk size from LLM (1234) or default (500)?
            # Code: final_chunk_size = llm_chunk_size if int else default
            self.assertEqual(decision["processing_params"]["chunk_size"], 1234)

        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()

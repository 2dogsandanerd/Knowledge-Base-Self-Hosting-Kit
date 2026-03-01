
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os
import asyncio
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../backend")))

# MOCK HEAVY DEPENDENCIES
sys.modules["src.core.llm_singleton"] = MagicMock()
sys.modules["src.core.config"] = MagicMock()

# Mock DocumentRouterService and DataClassifierService
# We need to mock them before import if they are imported at top level?
# pipeline.py imports:
# from src.services.data_classifier_service import get_data_classifier_service
# from src.services.document_router_service import DocumentRouterService

# We can patch them in the test method.

from src.services.ingest_v2.pipeline import process_document_pipeline

class TestPipelineIntegration(unittest.TestCase):
    
    @patch("src.services.ingest_v2.pipeline.get_data_classifier_service")
    @patch("src.services.ingest_v2.pipeline.DocumentRouterService")
    def test_pipeline_flow(self, mock_RouterClass, mock_get_classifier):
        async def run_test():
            # Setup Mocks
            mock_classifier = MagicMock()
            mock_get_classifier.return_value = mock_classifier
            
            mock_router_instance = MagicMock()
            mock_RouterClass.return_value = mock_router_instance
            
            # Setup Routing Decision
            mock_router_instance.route_document = AsyncMock(return_value={
                "target_collection": "contracts",
                "confidence": 0.9,
                "requires_validation": True,
                "processing_params": {
                    "chunk_size": 100,
                    "chunk_overlap": 10,
                    "postprocessing_steps": ["validate_clauses"]
                }
            })
            
            # Create a real temp file with some text
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
                tmp.write("This is a Section 1 contract. Clause 5 applies.")
                tmp_path = tmp.name
            
            try:
                # Run Pipeline
                results = await process_document_pipeline(tmp_path, {"filename": "test_contract.txt"})
                
                # Verify
                self.assertIn("routing_decision", results)
                self.assertIn("chunks_created", results)
                self.assertGreater(results["chunks_created"], 0)
                self.assertIn("clause_validation", results) # Checked because postprocessing_steps had validate_clauses
                
                # Check cleanup
                self.assertFalse(os.path.exists(tmp_path))
                
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()

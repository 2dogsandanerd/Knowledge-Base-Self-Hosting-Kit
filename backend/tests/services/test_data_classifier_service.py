
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os
import asyncio

# Adjust path to import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../backend")))

# MOCK HEAVY DEPENDENCIES BEFORE IMPORT
# Any module that imports heavy libs (llama-index, torch) should be mocked here
sys.modules["src.core.llm_singleton"] = MagicMock()
sys.modules["src.core.config"] = MagicMock()

# Now we can import the service. 
# Note: we need to careful if the service imports specific symbols from these modules.
# DataClassifierService does:
# from src.core.config import get_config, LLMConfig
# from src.core.llm_singleton import LLMSingleton

# So we need to ensure the mocks have these attributes
mock_config_module = sys.modules["src.core.config"]
mock_config_module.get_config = MagicMock()
mock_config_module.LLMConfig = MagicMock

mock_llm_module = sys.modules["src.core.llm_singleton"]
mock_llm_module.LLMSingleton = MagicMock

from src.services.data_classifier_service import DataClassifierService, CATEGORIES

class TestDataClassifierService(unittest.TestCase):
    def setUp(self):
        self.mock_llm_singleton = mock_llm_module.LLMSingleton()
        self.mock_llm_client = AsyncMock()
        self.mock_llm_singleton.get_client.return_value = self.mock_llm_client
        
        self.mock_config = mock_config_module.get_config()
        self.mock_config.embedding_model = "nomic-embed-text"
        
        self.service = DataClassifierService(self.mock_llm_singleton, self.mock_config)

    def test_heuristic_classify_code(self):
        """Test that .py files are classified as source_code"""
        result = self.service._heuristic_classify("test_script.py")
        self.assertEqual(result["category"], "source_code")
        self.assertEqual(result["confidence"], 0.95)

    def test_heuristic_classify_email(self):
        """Test that .eml files are classified as emails"""
        result = self.service._heuristic_classify("email.eml")
        self.assertEqual(result["category"], "emails")
        self.assertEqual(result["confidence"], 0.95)

    def test_heuristic_classify_spreadsheet(self):
        """Test that .xlsx files are classified as spreadsheets"""
        result = self.service._heuristic_classify("data.xlsx")
        self.assertEqual(result["category"], "spreadsheets")
        self.assertEqual(result["confidence"], 0.90)

    def test_heuristic_classify_presentation(self):
        """Test that .pptx files are classified as presentation"""
        result = self.service._heuristic_classify("slides.pptx")
        self.assertEqual(result["category"], "presentation")
        self.assertEqual(result["confidence"], 0.90)

    def test_heuristic_classify_generic(self):
        """Test that .pdf files are classified as documents (generic fallback for specific types)"""
        result = self.service._heuristic_classify("contract.pdf")
        self.assertEqual(result["category"], "documents")
        self.assertEqual(result["confidence"], 0.6)

    @patch("src.services.data_classifier_service.scan_folder")
    def test_analyze_folder_contents(self, mock_scan):
        """Test full analysis flow with mocked LLM"""
        # specialized helper to run async test
        async def run_test():
            # Setup mock file info
            mock_file = MagicMock()
            mock_file.path = "/tmp/contract.pdf"
            mock_file.filename = "contract.pdf"
            mock_file.extension = ".pdf"
            mock_file.size_bytes = 1000
            mock_file.size_human = "1KB"
            
            mock_scan.return_value = [mock_file]
            
            # Setup mocked LLM response
            self.mock_llm_client.predict.return_value = """
            {
                "recommended_collection": "legal_documents",
                "confidence": 0.95,
                "reasoning": "It is a contract.",
                "suggested_chunk_size": 1024,
                "suggested_embedding_model": "nomic-embed-text"
            }
            """
            
            # Using patch to mock _get_file_preview inside the instance logic is harder without subclassing or partial mocking.
            # But the method uses open(), so we can patch open() built-in if we want, or just let it fail gracefully?
            # Actually _get_file_preview catches exceptions. So if file doesn't exist, it returns empty string.
            # That is fine for this test.
            
            results = await self.service.analyze_folder_contents("/tmp")
            
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["recommended_collection"], "legal_documents")
            self.assertEqual(results[0]["confidence"], 0.95)
            self.assertEqual(results[0]["suggested_chunk_size"], 1024)

        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()

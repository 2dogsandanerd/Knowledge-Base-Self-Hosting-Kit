"""
Collection Config Generator.

Suggests optimal RAG collection configuration based on folder analysis.
"""

from typing import Dict, Any
import logging
import json
from src.services.generators.base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class CollectionConfigGenerator(BaseGenerator):
    """
    Generates optimal RAG collection configuration suggestions.

    Output includes:
    - Recommended collection structure
    - Chunking strategy
    - Embedding model recommendations
    - Indexing parameters
    """

    async def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate collection configuration.

        Args:
            input_data: {
                "folder_path": str,           # Required
                "use_case": str,              # Optional: "chatbot" | "search" | "qa"
                "expected_query_types": list  # Optional: ["technical", "business", ...]
            }

        Returns:
            {
                "config": {
                    "collections": [
                        {
                            "name": "...",
                            "description": "...",
                            "chunk_size": int,
                            "chunk_overlap": int,
                            "file_patterns": list
                        }
                    ],
                    "embedding_model": str,
                    "retrieval_strategy": str
                },
                "reasoning": str
            }
        """
        self.validate_input(input_data, ["folder_path"])

        folder_path = self._validate_folder_path(input_data["folder_path"])
        use_case = input_data.get("use_case", "search")
        query_types = input_data.get("expected_query_types", [])

        logger.info(f"Generating collection config for: {folder_path}")

        # 1. Analyze folder
        analysis = await self._analyze_folder(folder_path)

        # 2. Generate config via LLM
        config_data = await self._generate_config_from_analysis(
            analysis,
            use_case,
            query_types
        )

        return config_data

    async def _analyze_folder(self, folder_path: str) -> Dict[str, Any]:
        """Analyze folder structure."""
        if not self.classifier:
            raise ValueError("DataClassifierService not available")

        return await self.classifier.analyze_folder_contents(
            folder_path=folder_path,
            recursive=True,
            max_depth=10
        )

    async def _generate_config_from_analysis(
        self,
        analysis: Dict[str, Any],
        use_case: str,
        query_types: list
    ) -> Dict[str, Any]:
        """Generate config via LLM."""

        prompt = f"""You are a RAG system architect.

Based on this folder analysis, suggest an optimal RAG collection configuration.

ANALYSIS:
{analysis}

USE CASE: {use_case}
EXPECTED QUERY TYPES: {query_types}

PROVIDE CONFIGURATION IN JSON FORMAT:

{{
  "collections": [
    {{
      "name": "collection_name",
      "description": "What this collection contains",
      "chunk_size": 512,
      "chunk_overlap": 50,
      "file_patterns": ["*.md", "*.txt"]
    }}
  ],
  "embedding_model": "nomic-embed-text:latest",
  "retrieval_strategy": "hybrid",
  "reasoning": "Why this configuration is optimal"
}}

GUIDELINES:
- Separate collections for different content types (code, docs, data)
- Smaller chunks (256-512) for Q&A, larger (1024-2048) for context
- Use hybrid retrieval for mixed content
- Suggest appropriate embedding model based on content language

Output ONLY valid JSON, no markdown fences.
"""

        response = await self._call_llm(prompt, temperature=0.3, max_tokens=2000)
        config_json = self._sanitize_llm_output(response)

        # Parse JSON
        try:
            config_data = json.loads(config_json)
            return config_data
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM JSON: {config_json}")
            # Fallback or retry logic could go here
            # For now, return raw text in a wrapper if parsing fails, or raise error
            # Let's try to be robust and return a partial object with error
            return {
                "error": "Failed to parse LLM response as JSON",
                "raw_response": config_json
            }

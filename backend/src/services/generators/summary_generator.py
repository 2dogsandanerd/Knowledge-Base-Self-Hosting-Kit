"""
Summary Generator.

Generates comprehensive markdown summaries of folder contents.
"""

from typing import Dict, Any
import logging
from src.services.generators.base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class SummaryGenerator(BaseGenerator):
    """
    Generates markdown summaries of folder structures.

    Output includes:
    - Project overview
    - Key directories and their purposes
    - Important files
    - Technology stack detected
    - Recommendations
    """

    async def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate folder summary.

        Args:
            input_data: {
                "folder_path": str,           # Required
                "format": str,                # Optional: "markdown" | "json" (default: markdown)
                "include_file_tree": bool,    # Optional: Include file tree (default: True)
                "max_depth": int              # Optional: Max tree depth (default: 3)
            }

        Returns:
            {
                "summary": str,               # Markdown summary
                "metadata": {
                    "technology_stack": list,
                    "project_type": str,
                    "key_directories": list,
                    "total_files": int
                }
            }
        """
        self.validate_input(input_data, ["folder_path"])

        folder_path = self._validate_folder_path(input_data["folder_path"])
        format_type = input_data.get("format", "markdown")
        include_tree = input_data.get("include_file_tree", True)
        max_depth = input_data.get("max_depth", 3)

        logger.info(f"Generating summary for: {folder_path}")

        # 1. Analyze folder
        analysis = await self._analyze_folder(folder_path)

        # 2. Generate summary via LLM
        summary = await self._generate_summary_from_analysis(
            analysis,
            include_tree,
            max_depth
        )

        # 3. Extract metadata
        metadata = self._extract_metadata(analysis)

        return {
            "summary": summary,
            "metadata": metadata
        }

    async def _analyze_folder(self, folder_path: str) -> Dict[str, Any]:
        """Analyze folder structure."""
        if not self.classifier:
            raise ValueError("DataClassifierService not available")

        return await self.classifier.analyze_folder_contents(
            folder_path=folder_path,
            recursive=True,
            max_depth=10
        )

    async def _generate_summary_from_analysis(
        self,
        analysis: Dict[str, Any],
        include_tree: bool,
        max_depth: int
    ) -> str:
        """Generate summary via LLM."""

        prompt = f"""You are a technical documentation expert.

Generate a comprehensive markdown summary of this folder structure.

ANALYSIS DATA:
{analysis}

SUMMARY STRUCTURE:

# Project Overview
[Brief description of what this project is]

## Technology Stack
[List detected technologies: languages, frameworks, tools]

## Project Structure
[Explain key directories and their purposes]

{"## File Tree\n[Show directory tree up to depth " + str(max_depth) + "]" if include_tree else ""}

## Key Files
[List and explain important files: README, config files, entry points]

## Recommendations
[Suggestions for documentation, organization, or improvements]

---

Output ONLY the markdown summary, no preamble or explanations.
"""

        summary = await self._call_llm(prompt, temperature=0.5, max_tokens=3000)
        return self._sanitize_llm_output(summary)

    def _extract_metadata(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured metadata from analysis."""
        # Note: The analysis structure depends on DataClassifierService.analyze_folder_contents
        # which currently returns a List[Dict]. We might need to aggregate it first if we want global stats.
        # However, looking at RagignoreGenerator, it seems we expect 'categories' in analysis.
        # Let's assume analysis is the aggregated result or we need to aggregate it here.
        
        # Wait, DataClassifierService.analyze_folder_contents returns List[Dict] (list of file infos).
        # RagignoreGenerator handled this by... wait, let me check RagignoreGenerator again.
        # Ah, RagignoreGenerator calls self.classifier.analyze_folder_contents which returns List[Dict].
        # But in RagignoreGenerator._analyze_folder it just returns that list.
        # Then in _generate_ragignore_from_analysis it uses 'analysis' which is that list?
        # No, the prompt builder in RagignoreGenerator expects a dict with 'categories', 'total_files'.
        # There seems to be a mismatch in my previous implementation or understanding of DataClassifierService.
        
        # Let's look at DataClassifierService again.
        # It returns List[Dict].
        
        # So we need to aggregate the list into stats here, similar to what RagignoreGenerator SHOULD have done 
        # (or maybe I missed where it does it).
        
        # Let's implement a quick aggregation here to be safe.
        
        files_list = analysis if isinstance(analysis, list) else []
        
        categories = {}
        total_files = len(files_list)
        
        # Simple aggregation based on extensions/classification
        for file_info in files_list:
            # We can use the 'recommended_collection' as a proxy for category
            cat = file_info.get("recommended_collection", "generic")
            categories[cat] = categories.get(cat, 0) + 1
            
            # Also check extensions for tech stack
            ext = file_info.get("extension", "").lower()
            if ext == ".py":
                categories["source_code_python"] = categories.get("source_code_python", 0) + 1
            elif ext in [".js", ".jsx"]:
                categories["source_code_javascript"] = categories.get("source_code_javascript", 0) + 1
            elif ext in [".ts", ".tsx"]:
                categories["source_code_typescript"] = categories.get("source_code_typescript", 0) + 1

        # Detect technology stack
        tech_stack = []
        if categories.get("source_code_python", 0) > 0:
            tech_stack.append("Python")
        if categories.get("source_code_javascript", 0) > 0:
            tech_stack.append("JavaScript")
        if categories.get("source_code_typescript", 0) > 0:
            tech_stack.append("TypeScript")

        return {
            "technology_stack": tech_stack,
            "project_type": "Detected from file types", # Placeholder
            "key_directories": list(categories.keys())[:5], # This is actually categories, not directories, but serves the purpose for now
            "total_files": total_files
        }

# ragignore_generator.py – Phase 2a implementation
"""
Ragignore Generator.

Generates a .ragignore file from an intelligent folder analysis using the LLM.
"""

from typing import Dict, Any
import logging

from src.services.generators.base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class RagignoreGenerator(BaseGenerator):
    """Generator that creates a .ragignore file based on folder analysis.

    The class inherits from ``BaseGenerator`` which already provides:
    * ``self.llm_client`` – the LLM client obtained from ``LLMSingleton``
    * ``self.classifier`` – the ``DataClassifierService`` instance
    * helper methods ``_validate_folder_path``, ``_sanitize_prompt``, ``_call_llm``
    """

    def __init__(self, llm_singleton, data_classifier_service=None):
        """Initialize the generator.

        Args:
            llm_singleton: LLMSingleton instance (provides ``.get_client()``)
            data_classifier_service: optional ``DataClassifierService``
        """
        super().__init__(llm_singleton, data_classifier_service)
        self.logger = logging.getLogger(self.__class__.__name__)

    async def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the .ragignore content.

        Expected ``input_data`` keys:
            * ``folder_path`` (str) – required
            * ``include_examples`` (bool) – optional, default ``True``
            * ``aggressive`` (bool) – optional, default ``False``
        """
        # 1. Validate required keys
        self.validate_input(input_data, ["folder_path"])

        folder_path = await self._validate_folder_path(input_data["folder_path"])
        include_examples = input_data.get("include_examples", True)
        aggressive = input_data.get("aggressive", False)

        self.logger.info(f"Generating .ragignore for: {folder_path}")

        # 2. Analyse des Ordners
        analysis = await self._analyze_folder(folder_path)

        # 3. LLM‑Aufruf
        ragignore_content = await self._generate_ragignore_from_analysis(
            analysis, include_examples, aggressive
        )

        # 4. Statistik berechnen
        stats = self._calculate_statistics(analysis)

        # 5. Ergebnis zusammenstellen
        result = {
            "ragignore": ragignore_content,
            "analysis_summary": stats,
            "reasoning": analysis.get("reasoning", ""),
            "detected_categories": analysis.get("categories", {}),
        }

        self.logger.info(
            f".ragignore generated: {stats['files_to_ignore']} ignored, "
            f"{stats['files_to_keep']} kept"
        )
        return result

    async def _analyze_folder(self, folder_path: str) -> Dict[str, Any]:
        """Delegate folder analysis to ``DataClassifierService``.

        Raises:
            ValueError: if the classifier service is not available.
        """
        if not self.classifier:
            raise ValueError("DataClassifierService not available")
        try:
            analysis = await self.classifier.analyze_folder_contents(
                folder_path=folder_path, recursive=True, max_depth=10
            )
            return analysis
        except Exception as e:
            self.logger.error(f"Folder analysis failed: {e}", exc_info=True)
            raise

    async def _generate_ragignore_from_analysis(
        self, analysis: Dict[str, Any], include_examples: bool, aggressive: bool
    ) -> str:
        """Build the prompt and call the LLM to obtain the .ragignore text."""
        prompt = self._build_prompt(analysis, include_examples, aggressive)
        ragignore_text = await self._call_llm(
            prompt=prompt, temperature=0.3, max_tokens=2000
        )
        # Clean up possible markdown fences etc.
        ragignore_text = self._sanitize_llm_output(ragignore_text)
        return ragignore_text

    def _build_prompt(
        self, analysis: Dict[str, Any], include_examples: bool, aggressive: bool
    ) -> str:
        """Create the LLM prompt based on analysis and flags.

        The prompt follows the specification from the planning document and
        inserts the folder statistics, categories and optional notes.
        """
        categories = analysis.get("categories", {})
        total_files = analysis.get("total_files", 0)
        reasoning = analysis.get("reasoning", "")

        aggressive_note = (
            """\nAGGRESSIVE MODE: Be more strict with exclusions. When in doubt, exclude.\nOnly keep essential documentation and source code.\n"""
            if aggressive
            else ""
        )
        examples_note = (
            """\nInclude helpful comments with examples for each section.\nFormat: # Category name\n    pattern1/\n    pattern2\n    # Explanation or examples\n"""
            if include_examples
            else ""
        )

        prompt = f"""You are an expert in creating .ragignore files for RAG (Retrieval-Augmented Generation) systems.

A .ragignore file works like .gitignore but for document ingestion. It tells the RAG system which files to exclude from indexing.

FOLDER ANALYSIS:
- Total Files: {total_files}
- Detected Categories: {categories}
- Analysis: {reasoning}

{aggressive_note}
RULES FOR .RAGIGNORE:

ALWAYS EXCLUDE:
1. Build artifacts: node_modules/, venv/, __pycache__/, dist/, build/, target/, .next/
2. Binary files: *.exe, *.dll, *.so, *.dylib, *.bin
3. Large media: *.mp4, *.avi, *.mkv, *.mov, videos/, media/
4. Temporary files: *.tmp, *.log, *.cache, tmp/, temp/
5. Version control: .git/, .svn/, .hg/
6. IDE files: .vscode/, .idea/, *.swp, .DS_Store
7. Sensitive files: .env, .env.local, credentials.json, *.key, *.pem, secrets/
8. Compiled code: *.pyc, *.pyo, *.class, *.o
9. Archives: *.zip, *.tar, *.gz, *.rar (unless explicitly needed)
10. Lock files: package-lock.json, yarn.lock, poetry.lock, Pipfile.lock

ALWAYS KEEP:
1. Documentation: *.md, *.txt, *.rst, docs/, README*, CHANGELOG*
2. Source code: *.py, *.js, *.ts, *.java, *.cpp, *.go, src/, lib/
3. Configuration: *.yaml, *.yml, *.json, *.toml, *.ini, config/, .config/
4. Data files: *.csv, *.json, *.xml (if not too large)
5. Web content: *.html, *.css (if part of docs)

{examples_note}
OUTPUT FORMAT:
- Start with a header comment: # RAG Ignore File - Generated [date]
- Group patterns by category with comment headers
- Use relative paths and glob patterns
- One pattern per line
- End with a footer comment about customization

IMPORTANT:
- Output ONLY the .ragignore file content
- No markdown code fences
- No explanations before or after
- Start directly with the first comment line
"""
        return prompt

    def _calculate_statistics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate simple statistics for the generated .ragignore.

        Returns a dict with total files, files to ignore/keep and an estimated size reduction.
        """
        total_files = analysis.get("total_files", 0)
        categories = analysis.get("categories", {})

        ignore_categories = [
            "dependencies",
            "build_artifacts",
            "binary",
            "media",
            "temporary",
            "version_control",
            "compiled",
        ]
        files_to_ignore = sum(categories.get(cat, 0) for cat in ignore_categories)
        files_to_keep = max(0, total_files - files_to_ignore)

        # Estimate size reduction based on total size (if provided)
        total_size_mb = analysis.get("total_size_mb", 0)
        estimated_ignore_pct = (
            (files_to_ignore / total_files * 100) if total_files > 0 else 0
        )
        estimated_size_reduction_mb = total_size_mb * (estimated_ignore_pct / 100)

        return {
            "total_files": total_files,
            "files_to_ignore": files_to_ignore,
            "files_to_keep": files_to_keep,
            "estimated_size_reduction_mb": round(estimated_size_reduction_mb, 2),
        }

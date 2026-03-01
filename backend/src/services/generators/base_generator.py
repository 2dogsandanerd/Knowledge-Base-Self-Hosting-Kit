"""
Base Generator for LLM Tasks.

Provides common interface and utilities for all generators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
import os
import re
from src.core.services.settings_service import SettingsService, DEFAULT_SECURITY_SETTINGS

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """
    Abstract base class for all LLM-powered generators.

    Subclasses must implement the generate() method.
    Provides shared utilities for validation, LLM calls, and formatting.
    """

    def __init__(self, llm_singleton, data_classifier_service=None, settings_service=None):
        """
        Initialize generator with required services.

        Args:
            llm_singleton: LLMSingleton instance (matches existing codebase pattern)
            data_classifier_service: Optional classifier service
            settings_service: Optional SettingsService for security configuration
        """
        self.llm_singleton = llm_singleton
        self.llm_client = llm_singleton.get_client()
        self.classifier = data_classifier_service
        self.settings_service = settings_service
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate output based on input data.

        This is the main entry point that all generators must implement.

        Args:
            input_data: Task-specific input parameters

        Returns:
            Task-specific output dictionary

        Raises:
            ValueError: If input validation fails
            Exception: If generation fails
        """
        pass

    def validate_input(self, input_data: Dict[str, Any], required_keys: List[str]):
        """
        Validate that required keys exist in input data.

        Args:
            input_data: Input dictionary to validate
            required_keys: List of required key names

        Raises:
            ValueError: If any required keys are missing
        """
        missing = [key for key in required_keys if key not in input_data]
        if missing:
            raise ValueError(f"Missing required input keys: {missing}")

        self.logger.debug(f"Input validation passed: {list(input_data.keys())}")

    async def _validate_folder_path(self, folder_path: str) -> str:
        """
        Validate that the provided folder path is within allowed directories and exists.

        Uses SettingsService to retrieve allowed_roots from database.
        Falls back to DEFAULT_SECURITY_SETTINGS if SettingsService is not available.

        Args:
            folder_path: User-provided path string.

        Returns:
            Absolute, normalized path.

        Raises:
            ValueError: If path is outside allowed roots or does not exist.
        """
        # Resolve absolute path
        abs_path = os.path.abspath(folder_path)

        # Get allowed roots from settings or use defaults
        if self.settings_service:
            try:
                allowed_roots = await self.settings_service.get_allowed_roots()
            except Exception as e:
                self.logger.warning(f"Failed to load allowed_roots from settings: {e}, using defaults")
                allowed_roots = DEFAULT_SECURITY_SETTINGS["allowed_roots"]
        else:
            self.logger.warning("SettingsService not available, using default allowed_roots")
            allowed_roots = DEFAULT_SECURITY_SETTINGS["allowed_roots"]

        # Normalize allowed_roots to absolute paths
        allowed_roots = [os.path.abspath(root) for root in allowed_roots]

        if not any(abs_path.startswith(root) for root in allowed_roots):
            raise ValueError(
                f"Path '{folder_path}' is not within allowed directories: {allowed_roots}"
            )

        if not os.path.isdir(abs_path):
            raise ValueError(f"Path '{folder_path}' does not exist or is not a directory")

        return abs_path

    def _sanitize_prompt(self, prompt: str, max_length: int = 2000) -> str:
        """
        Sanitize LLM prompt to prevent injection and enforce length limits.

        Args:
            prompt: Raw prompt string.
            max_length: Maximum allowed characters.

        Returns:
            Cleaned prompt string.
        """
        # Remove control characters
        cleaned = re.sub(r"[\x00-\x1F\x7F]", "", prompt)
        # Strip dangerous patterns that could alter LLM behavior
        dangerous_patterns = [
            "ignore previous instructions",
            "system:",
            "assistant:",
            "###",
            "```",
        ]
        for pat in dangerous_patterns:
            cleaned = cleaned.replace(pat, "")
        # Truncate to max length
        return cleaned[:max_length]

    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Call LLM with error handling and logging.

        Uses .predict() method if available, otherwise falls back to .complete().

        Args:
            prompt: The prompt to send to LLM
            temperature: Sampling temperature (0.0 - 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            LLM response text

        Raises:
            ValueError: If LLM service not available
            Exception: If LLM call fails
        """
        self.logger.debug(f"Calling LLM (temperature={temperature}, max_tokens={max_tokens})")

        try:
            if not self.llm_client:
                raise ValueError("LLM service not available")

            # Prefer predict method
            if hasattr(self.llm_client, "predict"):
                response = await self.llm_client.predict(prompt)
            elif hasattr(self.llm_client, "complete"):
                response = await self.llm_client.complete(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise ValueError("LLM client does not support predict or complete methods")

            self.logger.debug(f"LLM response received ({len(response)} chars)")
            return response

        except Exception as e:
            self.logger.error(f"LLM call failed: {e}", exc_info=True)
            raise

    def _sanitize_llm_output(self, text: str) -> str:
        """
        Clean up LLM output (remove markdown fences, extra whitespace).

        Args:
            text: Raw LLM output

        Returns:
            Cleaned text
        """
        # Remove markdown code fences if present
        text = text.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```language) and last line (```)
            if len(lines) > 2 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1])
            elif len(lines) > 1:
                # Only opening fence, no closing
                text = "\n".join(lines[1:])

        # Remove excessive whitespace
        text = text.strip()

        return text

    def _format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted string (e.g., "1.5 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

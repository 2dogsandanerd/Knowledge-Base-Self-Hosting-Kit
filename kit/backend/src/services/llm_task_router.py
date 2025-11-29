"""
LLM Task Router Service.

The "Stellwerk" (Control Tower) that routes LLM tasks to specialized generators.
Provides shared logic for error handling, logging, caching, and metrics.
"""

from enum import Enum
from typing import Dict, Any, Optional
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Available LLM task types."""
    GENERATE_RAGIGNORE = "generate_ragignore"
    GENERATE_SUMMARY = "generate_summary"
    GENERATE_COLLECTION_CONFIG = "generate_collection_config"
    # Add more as needed in Phase 4


class LLMTaskRouter:
    """
    Routes LLM tasks to appropriate generators.

    This is the central "Stellwerk" that:
    - Maintains a registry of generators
    - Routes tasks based on TaskType
    - Provides shared error handling
    - Implements caching (optional)
    - Tracks metrics
    """

    def __init__(
        self,
        llm_singleton,
        data_classifier_service,
        cache_service: Optional[Any] = None
    ):
        """
        Initialize router with required services.

        Args:
            llm_singleton: LLMSingleton instance providing LLM client
            data_classifier_service: Data classifier service
            cache_service: Optional cache service (e.g., Redis)
        """
        self.llm_singleton = llm_singleton
        self.llm_client = llm_singleton.get_client()
        self.classifier = data_classifier_service
        self.cache = cache_service

        # Generator registry
        self.generators: Dict[TaskType, Any] = {}
        self._register_generators()

        logger.info(f"LLMTaskRouter initialized with {len(self.generators)} generators")

    def _register_generators(self):
        """
        Register all available generators.

        This is where new generators are added to the system.
        Import generators here to avoid circular imports.
        """
        # Phase 2: RagignoreGenerator
        try:
            from src.services.generators.ragignore_generator import RagignoreGenerator
            self.generators[TaskType.GENERATE_RAGIGNORE] = RagignoreGenerator(
                llm_singleton=self.llm_singleton,
                data_classifier_service=self.classifier
            )
            logger.debug("Registered: RagignoreGenerator")
        except ImportError as e:
            logger.warning(f"RagignoreGenerator not available: {e}")

        # Phase 4: SummaryGenerator
        try:
            from src.services.generators.summary_generator import SummaryGenerator
            self.generators[TaskType.GENERATE_SUMMARY] = SummaryGenerator(
                llm_singleton=self.llm_singleton,
                data_classifier_service=self.classifier
            )
            logger.debug("Registered: SummaryGenerator")
        except ImportError as e:
            logger.warning(f"SummaryGenerator not available: {e}")

        # Phase 4: CollectionConfigGenerator
        try:
            from src.services.generators.collection_config_generator import CollectionConfigGenerator
            self.generators[TaskType.GENERATE_COLLECTION_CONFIG] = CollectionConfigGenerator(
                llm_singleton=self.llm_singleton,
                data_classifier_service=self.classifier
            )
            logger.debug("Registered: CollectionConfigGenerator")
        except ImportError as e:
            logger.warning(f"CollectionConfigGenerator not available: {e}")

    async def execute(
        self,
        task_type: TaskType,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a specific LLM task by routing to appropriate generator.

        This is the main entry point for all LLM tasks.

        Steps:
        1. Validate task type
        2. Check cache (if enabled)
        3. Get appropriate generator
        4. Execute with error handling
        5. Cache result (if enabled)
        6. Return result

        Args:
            task_type: Type of task to execute
            input_data: Task-specific input data

        Returns:
            Task-specific result dictionary

        Raises:
            ValueError: If task type is unknown
            Exception: If task execution fails
        """
        logger.info(f"Routing task: {task_type}", extra={
            "task_type": task_type,
            "input_keys": list(input_data.keys())
        })

        # 1. Validate task type
        generator = self.generators.get(task_type)
        if not generator:
            available_tasks = list(self.generators.keys())
            raise ValueError(
                f"Unknown task type: {task_type}. "
                f"Available: {available_tasks}"
            )

        # 2. Check cache (if enabled)
        if self.cache:
            cache_key = self._build_cache_key(task_type, input_data)
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Cache hit for task: {task_type}")
                return cached_result

        # 3. Execute generator
        try:
            result = await generator.generate(input_data)

            # 4. Cache result (if enabled)
            if self.cache:
                cache_key = self._build_cache_key(task_type, input_data)
                await self._save_to_cache(cache_key, result)

            logger.info(f"Task {task_type} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Task {task_type} failed: {e}", exc_info=True)
            raise

    def _build_cache_key(self, task_type: TaskType, input_data: Dict[str, Any]) -> str:
        """
        Build cache key from task type and input data.

        Args:
            task_type: Task type
            input_data: Input data

        Returns:
            Cache key string
        """
        # Create deterministic hash of input data
        input_json = json.dumps(input_data, sort_keys=True)
        input_hash = hashlib.md5(input_json.encode()).hexdigest()

        return f"llm_task:{task_type}:{input_hash}"

    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get result from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached result or None if not found
        """
        try:
            if hasattr(self.cache, 'get'):
                return await self.cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        return None

    async def _save_to_cache(self, cache_key: str, result: Dict[str, Any], ttl: int = 3600):
        """
        Save result to cache.

        Args:
            cache_key: Cache key
            result: Result to cache
            ttl: Time to live in seconds (default: 1 hour)
        """
        try:
            if hasattr(self.cache, 'set'):
                await self.cache.set(cache_key, result, ttl=ttl)
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    def get_available_tasks(self) -> list[str]:
        """
        Get list of available task types.

        Returns:
            List of task type strings
        """
        return [task.value for task in self.generators.keys()]

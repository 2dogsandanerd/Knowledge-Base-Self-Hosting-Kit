"""
Ingest Parameter Advisor Service

Provides heuristic-based recommendations for ingestion parameters (chunk size, overlap, splitter type,
embedding model, collection) based on document characteristics.

Phase 1 Implementation: Pure heuristics (no ML/Agent)
Future: Will integrate ML-based advice in Phase 8
"""

from dataclasses import dataclass
from typing import Optional
from loguru import logger

from src.core.ingest_config import get_ingest_config, ChunkConfig


@dataclass
class IngestionAdvice:
    """Recommendation output from the advisor"""
    chunk_size: int
    overlap: int
    splitter_type: str  # "semantic", "fixed", "code", "row_based"
    embedding_model: str
    collection_name: str
    confidence: float  # 0.0-1.0 (1.0 = very confident heuristic match)
    reasoning: str  # Explanation for recommendation
    source: str = "heuristic"  # "heuristic" (Phase 1), "ml" (Phase 8+), "agent" (Phase 6+)


class IngestParameterAdvisor:
    """Advises on optimal ingestion parameters for documents"""
    
    def __init__(self):
        self.config = get_ingest_config()
        self.heuristics = self.config.heuristics
        logger.info("✅ IngestParameterAdvisor initialized with heuristic defaults")
    
    def get_advice(
        self,
        file_path: str,
        document_type: Optional[str] = None,
        mime_type: Optional[str] = None,
        estimated_tokens: Optional[int] = None,
        has_tables: bool = False,
        has_code: bool = False,
        language: str = "en",
        **kwargs
    ) -> IngestionAdvice:
        """
        Get ingestion parameter advice based on document characteristics.
        
        Args:
            file_path: Path to document
            document_type: Pre-classified doc type (invoice, contract, email, code, etc.)
            mime_type: MIME type of document
            estimated_tokens: Pre-calculated token count
            has_tables: Whether document contains tables
            has_code: Whether document contains code blocks
            language: Document language (for multilingual embeddings)
            **kwargs: Additional features
        
        Returns:
            IngestionAdvice with recommendations
        """
        
        logger.debug(f"🔍 Advising on parameters for: {file_path}")
        logger.debug(f"   Type: {document_type}, MIME: {mime_type}, Tokens: {estimated_tokens}")
        
        # Determine document type from multiple signals
        inferred_type = self._infer_document_type(
            file_path=file_path,
            mime_type=mime_type,
            document_type=document_type,
            has_code=has_code,
            has_tables=has_tables,
        )
        
        # Get base recommendation from heuristics
        base_advice = self._get_heuristic_advice(inferred_type)
        
        # Apply contextual adjustments
        advice = self._apply_contextual_adjustments(
            base_advice=base_advice,
            doc_type=inferred_type,
            estimated_tokens=estimated_tokens,
            has_tables=has_tables,
            has_code=has_code,
            language=language,
        )
        
        # Determine collection
        collection_name = self._determine_collection(inferred_type, mime_type)
        advice.collection_name = collection_name
        
        logger.info(
            f"✨ Advice generated: chunk={advice.chunk_size}, "
            f"overlap={advice.overlap}, splitter={advice.splitter_type}, "
            f"collection={collection_name}, confidence={advice.confidence:.2f}"
        )
        
        return advice
    
    def _infer_document_type(
        self,
        file_path: str,
        mime_type: Optional[str],
        document_type: Optional[str],
        has_code: bool,
        has_tables: bool,
    ) -> str:
        """Infer document type from available signals"""
        
        # Priority 1: Pre-classified type
        if document_type:
            return document_type
        
        # Priority 2: Code detection
        if has_code:
            return "code"
        
        # Priority 3: MIME type
        if mime_type:
            if "pdf" in mime_type:
                return "pdf"
            elif "word" in mime_type or "docx" in mime_type:
                return "docx"
            elif "html" in mime_type:
                return "html"
            elif "csv" in mime_type or "spreadsheet" in mime_type:
                return "csv"
            elif "email" in mime_type or "message" in mime_type:
                return "email"
            elif "text" in mime_type:
                return "markdown"
        
        # Priority 4: File extension
        if file_path:
            ext = file_path.split(".")[-1].lower()
            ext_map = {
                "pdf": "pdf",
                "docx": "docx",
                "doc": "docx",
                "html": "html",
                "htm": "html",
                "md": "markdown",
                "csv": "csv",
                "xlsx": "csv",
                "xls": "csv",
                "eml": "email",
                "msg": "email",
                "py": "code",
                "js": "code",
                "java": "code",
                "cpp": "code",
                "go": "code",
            }
            if ext in ext_map:
                return ext_map[ext]
        
        # Fallback
        return "default"
    
    def _get_heuristic_advice(self, doc_type: str) -> IngestionAdvice:
        """Get base heuristic recommendation for document type"""
        
        # Get config for this type
        chunk_config: ChunkConfig = getattr(
            self.heuristics, doc_type, self.heuristics.default
        )
        
        advice = IngestionAdvice(
            chunk_size=chunk_config.chunk_size,
            overlap=chunk_config.overlap,
            splitter_type=chunk_config.splitter_type,
            embedding_model=chunk_config.embedding_model,
            collection_name="",  # Will be set later
            confidence=0.8,  # Heuristic confidence
            reasoning=f"Using heuristic defaults for {doc_type}"
        )
        
        return advice
    
    def _apply_contextual_adjustments(
        self,
        base_advice: IngestionAdvice,
        doc_type: str,
        estimated_tokens: Optional[int],
        has_tables: bool,
        has_code: bool,
        language: str,
    ) -> IngestionAdvice:
        """Apply context-specific adjustments to base advice"""
        
        advice = IngestionAdvice(
            chunk_size=base_advice.chunk_size,
            overlap=base_advice.overlap,
            splitter_type=base_advice.splitter_type,
            embedding_model=base_advice.embedding_model,
            collection_name=base_advice.collection_name,
            confidence=base_advice.confidence,
            reasoning=base_advice.reasoning,
        )
        
        adjustments = []
        
        # Adjustment 1: Token-based size recommendation
        if estimated_tokens and estimated_tokens > 5000:
            # Large documents: smaller chunks for better relevance
            advice.chunk_size = max(400, int(base_advice.chunk_size * 0.8))
            advice.overlap = max(60, int(base_advice.overlap * 0.8))
            adjustments.append("Large document: reduced chunk size")
        elif estimated_tokens and estimated_tokens < 1000:
            # Small documents: might use larger chunks
            advice.chunk_size = min(1024, int(base_advice.chunk_size * 1.2))
            adjustments.append("Small document: increased chunk size")
        
        # Adjustment 2: Table handling
        if has_tables and doc_type not in ["csv", "email"]:
            advice.splitter_type = "table_aware"  # TODO: implement table-aware splitter
            adjustments.append("Tables detected: using table-aware splitting")
        
        # Adjustment 3: Code handling
        if has_code and doc_type != "code":
            advice.splitter_type = "code"  # Preserve code structure
            advice.chunk_size = 256  # Smaller chunks for code
            adjustments.append("Code detected: using code-aware splitting")
        
        # Adjustment 4: Multilingual embeddings
        if language not in ["en", "de"]:
            advice.embedding_model = self.config.embedding_model_multilingual
            adjustments.append(f"Non-English language ({language}): using multilingual embeddings")
        
        if adjustments:
            advice.reasoning = base_advice.reasoning + " | " + " | ".join(adjustments)
            advice.confidence = min(0.95, advice.confidence + 0.05)  # Boost confidence with more signals
        
        return advice
    
    def _determine_collection(self, doc_type: str, mime_type: Optional[str]) -> str:
        """Determine optimal collection for document"""
        
        # Default collection naming: collection_{doc_type}
        if self.config.collection_naming_scheme == "doc_type":
            return f"collection_{doc_type}"
        
        # TODO: Add domain_type and auto schemes in later phases
        return f"collection_{doc_type}"
    
    def get_multilingual_embedding_model(self, language: str) -> str:
        """Get optimal embedding model for language"""
        
        if language in ["de", "fr", "es", "it", "nl"]:
            return self.config.embedding_model_multilingual  # e5-large
        elif language == "zh" or language == "ja":
            return self.config.embedding_model_multilingual
        else:
            return self.config.embedding_model_default


# Global advisor instance
_advisor: IngestParameterAdvisor | None = None


def get_ingest_advisor() -> IngestParameterAdvisor:
    """Get or create global advisor instance"""
    global _advisor
    if _advisor is None:
        _advisor = IngestParameterAdvisor()
    return _advisor


def reload_advisor() -> IngestParameterAdvisor:
    """Reload advisor with updated config"""
    global _advisor
    _advisor = IngestParameterAdvisor()
    return _advisor

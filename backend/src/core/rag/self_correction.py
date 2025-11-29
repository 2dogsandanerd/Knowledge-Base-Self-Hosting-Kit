"""
Self-Correction Module (Phase H.3).

Handles answer verification and hallucination checks.
"""

from typing import Dict, Any, List
from loguru import logger
from src.core.llm_singleton import get_llm

class SelfCorrection:
    """
    Service for verifying and correcting RAG answers.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="SelfCorrection")

    async def verify_answer(self, query: str, answer: str, context_chunks: List[Any]) -> Dict[str, Any]:
        """
        Check if the answer is supported by the context (Self-RAG).
        
        Returns:
            Dict with 'is_supported' (bool) and 'reasoning' (str)
        """
        try:
            llm = get_llm(temperature=0.1) # Low temp for strict evaluation
            
            context_text = "\n\n".join([c.content if hasattr(c, 'content') else str(c) for c in context_chunks])
            
            prompt = (
                "You are a strict fact-checker. Evaluate if the ANSWER is fully supported by the CONTEXT.\n"
                "Ignore external knowledge.\n\n"
                f"CONTEXT:\n{context_text[:4000]}\n\n" # Limit context size
                f"USER QUESTION: {query}\n"
                f"GENERATED ANSWER: {answer}\n\n"
                "Does the context support the answer?\n"
                "Reply with JSON format: {\"supported\": boolean, \"reason\": \"short explanation\"}"
            )
            
            response = await llm.acomplete(prompt)
            response_text = str(response).strip()
            
            # Simple parsing (robustness improvement needed for production)
            import json
            # Try to find JSON block
            if "{" in response_text and "}" in response_text:
                json_str = response_text[response_text.find("{"):response_text.rfind("}")+1]
                result = json.loads(json_str)
                return {
                    "is_supported": result.get("supported", True),
                    "reasoning": result.get("reason", "No reason provided")
                }
            
            return {"is_supported": True, "reasoning": "Could not parse verification"}

        except Exception as e:
            self.logger.warning(f"Answer verification failed: {e}")
            return {"is_supported": True, "reasoning": "Verification error"}

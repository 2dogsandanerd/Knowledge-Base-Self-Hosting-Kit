"""
Query Understanding Module (Phase H.2).

Handles query expansion and decomposition to improve retrieval recall.
"""

from typing import List
from loguru import logger
from src.core.llm_singleton import get_llm

class QueryUnderstanding:
    """
    Service for enhancing user queries before retrieval.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="QueryUnderstanding")

    async def expand_query(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate variations of the query to improve recall (Query Expansion).
        
        Example:
        Input: "cost of service"
        Output: ["cost of service", "pricing model", "service fees", "rates"]
        """
        try:
            llm = get_llm(temperature=0.7) # Higher temp for creativity
            
            prompt = (
                f"You are a helpful assistant. Generate {num_variations} alternative search queries "
                f"based on the user's question. The goal is to find relevant documents in a knowledge base.\n"
                f"User Question: '{query}'\n"
                f"Output ONLY the alternative queries, one per line. Do not number them."
            )
            
            response = await llm.acomplete(prompt)
            variations = [line.strip() for line in str(response).split('\n') if line.strip()]
            
            # Always include original query
            unique_variations = list(set([query] + variations))
            
            self.logger.debug(f"Expanded '{query}' into: {unique_variations}")
            return unique_variations[:num_variations+1]
            
        except Exception as e:
            self.logger.warning(f"Query expansion failed: {e}")
            return [query] # Fallback to original

    async def decompose_query(self, query: str) -> List[str]:
        """
        Break complex query into sub-questions (Decomposition).
        (Placeholder for future implementation)
        """
        return [query]

"""
Query Classifier - LLM-based domain classification for RAG queries.

This service analyzes user queries and classifies them into domains
to automatically filter relevant collections for better search results.
"""

from typing import Optional, List, Dict
from loguru import logger
from src.core.llm_singleton import get_llm


class QueryClassifier:
    """
    Classifies user queries into domains to filter relevant collections.
    Uses LLM to analyze query and match to available domains from rag_domains.json
    """
    
    def __init__(self, domains_config: Optional[Dict] = None):
        """
        Initialize QueryClassifier.
        
        Args:
            domains_config: Dictionary from rag_domains.json with domain->collections mapping
        """
        self.domains_config = domains_config or {}
        self.logger = logger.bind(component="QueryClassifier")
        
    async def classify_query(self, query_text: str) -> Optional[str]:
        """
        Classify query into a domain.
        
        Args:
            query_text: User's search query
            
        Returns:
            Domain name (str) or None if no clear match
        """
        if not self.domains_config:
            self.logger.warning("No domains configured, skipping classification")
            return None
            
        # Build prompt with available domains
        domain_list = list(self.domains_config.keys())
        
        prompt = f"""Analysiere die folgende Suchanfrage und bestimme die passendste Domäne.

Verfügbare Domänen: {', '.join(domain_list)}

Anfrage: "{query_text}"

Antworte NUR mit dem exakten Domänennamen aus der Liste, oder "none" wenn keine passt.
Keine Erklärungen, nur der Domänenname."""
        
        try:
            llm = get_llm(temperature=0.0)  # Deterministic classification
            response = await llm.acomplete(prompt)
            
            domain = response.text.strip().lower()
            
            # Validate domain exists
            if domain in [d.lower() for d in domain_list]:
                # Find original case-sensitive domain name
                for orig_domain in domain_list:
                    if orig_domain.lower() == domain:
                        self.logger.info(f"Query classified as domain: {orig_domain}")
                        return orig_domain
            
            self.logger.info(f"Query could not be classified (response: {domain})")
            return None
            
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return None
    
    def get_collections_for_domain(self, domain: str) -> List[str]:
        """
        Get collection names for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            List of collection names
        """
        if domain in self.domains_config:
            return self.domains_config[domain].get("collections", [])
        return []

"""
Query Enhancement Implementation for Phase 4.

This module implements techniques to enhance user queries before retrieval,
including query expansion (generating alternative queries) and HyDE
(Hypothetical Document Embeddings) to improve recall.
"""

from typing import List, Optional
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from loguru import logger


class QueryEnhancer:
    """
    Service for enhancing queries using LLM-based techniques.
    
    Provides two methods for query enhancement:
    1. Query Expansion: Generate multiple alternative queries
    2. HyDE: Generate hypothetical documents that answer the query
    """

    def __init__(self, llm: Optional[LLM] = None):
        """
        Initialize the QueryEnhancer.

        Args:
            llm: LLM instance to use for query enhancement
        """
        self.llm = llm
        self.logger = logger.bind(component="QueryEnhancer")

    def expand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        """
        Generate alternative queries that are semantically similar to the original.

        Args:
            query: Original query to expand
            num_expansions: Number of alternative queries to generate

        Returns:
            List of expanded queries (original + alternatives)
        """
        if not self.llm:
            self.logger.warning("No LLM provided for query expansion, returning original query")
            return [query]

        try:
            # Create prompt for query expansion
            expansion_prompt = PromptTemplate(
                template="""
Given the user query: "{query}"

Generate exactly {num_expansions} alternative versions of this query that are semantically similar 
but use different words or phrasing. Each alternative should help find similar information 
with different search terms.

Format your response as a numbered list:
1. [Alternative query 1]
2. [Alternative query 2]
3. [Alternative query 3]

Examples:
Original: "How to fix my car engine?"
Alternatives:
1. "Car engine repair instructions"
2. "Troubleshooting car engine problems"
3. "DIY car engine fix guide"
                """.strip()
            )

            # Execute the prompt
            response = self.llm.predict(
                expansion_prompt,
                query=query,
                num_expansions=num_expansions
            )

            # Parse the response to extract alternative queries
            alternatives = []
            lines = str(response).split('\n')
            
            for line in lines:
                line = line.strip()
                # Look for numbered list items
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    # Extract the query part after the number
                    parts = line.split('.', 1)
                    if len(parts) > 1:
                        alternative = parts[1].strip()
                        if alternative and alternative != query:  # Avoid duplicates
                            alternatives.append(alternative)

            # Return original query + alternatives (deduplicated)
            all_queries = [query]
            for alt in alternatives:
                if alt not in all_queries and alt != query:
                    all_queries.append(alt)

            self.logger.info(f"Expanded query '{query}' to {len(all_queries)} queries")
            return all_queries

        except Exception as e:
            self.logger.error(f"Query expansion failed: {e}")
            # Return original query if expansion fails
            return [query]

    async def aexpand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        """
        Asynchronous version of expand_query.
        """
        return self.expand_query(query, num_expansions)

    def generate_hyde_document(self, query: str) -> str:
        """
        Generate a hypothetical document that answers the query.

        Args:
            query: Query for which to generate a hypothetical answer

        Returns:
            Hypothetical document text
        """
        if not self.llm:
            self.logger.warning("No LLM provided for HyDE generation, returning empty string")
            return ""

        try:
            # Create prompt for HyDE document generation
            hyde_prompt = PromptTemplate(
                template="""
Write a concise, hypothetical document that perfectly answers this query:

Query: "{query}"

The document should be:
1. Factual and specific to the query
2. Concise but comprehensive
3. Written as if from an authoritative source
4. Include relevant technical details if applicable

Write the document assuming it would be found in response to this query:
                """.strip()
            )

            # Execute the prompt
            response = self.llm.predict(hyde_prompt, query=query)

            hyde_doc = str(response).strip()
            self.logger.info(f"Generated HyDE document for query: '{query[:50]}...'")
            return hyde_doc

        except Exception as e:
            self.logger.error(f"HyDE document generation failed: {e}")
            # Return empty string if generation fails
            return ""

    async def agenerate_hyde_document(self, query: str) -> str:
        """
        Asynchronous version of generate_hyde_document.
        """
        return self.generate_hyde_document(query)


class QueryEnhancementRetriever:
    """
    A retriever that enhances queries before passing them to the underlying retriever.
    
    Supports two enhancement methods:
    1. Expansion: Performs multiple queries and merges results
    2. HyDE: Generates hypothetical document and uses its embedding for retrieval
    """
    
    def __init__(
        self,
        base_retriever,
        query_enhancer: QueryEnhancer,
        method: str = "expansion",  # "expansion" or "hyde"
        num_expansions: int = 3
    ):
        """
        Initialize the QueryEnhancementRetriever.

        Args:
            base_retriever: The underlying retriever to use
            query_enhancer: The query enhancement service
            method: Enhancement method ("expansion" or "hyde")
            num_expansions: Number of expansions to generate (for expansion method)
        """
        self.base_retriever = base_retriever
        self.query_enhancer = query_enhancer
        self.method = method
        self.num_expansions = num_expansions
        self.logger = logger.bind(component="QueryEnhancementRetriever")

    def _retrieve_expansion_method(self, query: str):
        """Execute the expansion method for query enhancement."""
        # Generate expanded queries
        expanded_queries = self.query_enhancer.expand_query(
            query, 
            num_expansions=self.num_expansions
        )
        
        # Retrieve results for each query
        all_nodes = []
        for exp_query in expanded_queries:
            # Create a QueryBundle for each expanded query
            from llama_index.core.schema import QueryBundle
            query_bundle = QueryBundle(query_str=exp_query)
            
            nodes = self.base_retriever.retrieve(query_bundle)
            all_nodes.extend(nodes)
        
        # In a full implementation, we would apply fusion/merging logic here
        # For now, let's return all nodes (in practice, you'd want to deduplicate and rerank)
        return all_nodes

    def _retrieve_hyde_method(self, query: str):
        """Execute the HyDE method for query enhancement."""
        # Generate hypothetical document
        hyde_doc = self.query_enhancer.generate_hyde_document(query)
        
        if not hyde_doc:
            # Fallback to original query if HyDE generation fails
            from llama_index.core.schema import QueryBundle
            query_bundle = QueryBundle(query_str=query)
            return self.base_retriever.retrieve(query_bundle)
        
        # In a full implementation, we would use the hyde_doc for embedding
        # For now, we'll use the hyde document text as the query
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(query_str=hyde_doc)
        return self.base_retriever.retrieve(query_bundle)

    def retrieve(self, query_bundle):
        """
        Retrieve nodes using enhanced queries.

        Args:
            query_bundle: The original query

        Returns:
            List of nodes retrieved using enhanced queries
        """
        if self.method == "expansion":
            return self._retrieve_expansion_method(query_bundle.query_str)
        elif self.method == "hyde":
            return self._retrieve_hyde_method(query_bundle.query_str)
        else:
            # Default to base retriever if method is unknown
            return self.base_retriever.retrieve(query_bundle)

    async def aretrieve(self, query_bundle):
        """
        Asynchronous version of retrieve.
        """
        # For now, just call the synchronous version
        # A full implementation would async-ify the underlying operations
        return self.retrieve(query_bundle)
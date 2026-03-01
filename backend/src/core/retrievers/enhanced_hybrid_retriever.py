from typing import List, Optional, Dict, Any
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.callbacks import CallbackManager
from loguru import logger
import asyncio

class EnhancedHybridRetriever(BaseRetriever):
    """Enhanced Hybrid Retriever that combines results from multiple retrievers with improved RRF."""

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: Optional[List[float]] = None,
        k: int = 60,  # RRF k parameter
        top_k: int = 10,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize EnhancedHybridRetriever.
        
        Args:
            retrievers: List of retrievers to combine
            weights: Optional weights for each retriever (must match length of retrievers)
            k: RRF k parameter (higher values reduce impact of rank differences)
            top_k: Maximum number of results to return
        """
        super().__init__(callback_manager)
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)  # Equal weights by default
        
        if len(self.weights) != len(self.retrievers):
            raise ValueError("Number of weights must match number of retrievers")
            
        self.k = k
        self.top_k = top_k
        self.logger = logger.bind(component="EnhancedHybridRetriever")
        self.verbose = verbose

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve nodes from multiple retrievers and fuse results using weighted RRF."""
        
        if not self.retrievers:
            return []

        # Run all retrievers concurrently
        tasks = []
        for retriever in self.retrievers:
            if hasattr(retriever, 'aretrieve'):
                tasks.append(retriever.aretrieve(query_bundle))
            else:
                # Fallback to sync retrieve if async not available
                tasks.append(self._sync_to_async_wrapper(retriever.retrieve, query_bundle))
        
        results_per_retriever = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred during retrieval
        processed_results = []
        for i, result in enumerate(results_per_retriever):
            if isinstance(result, Exception):
                self.logger.warning(f"Retriever {i} failed: {result}")
                processed_results.append([])
            else:
                processed_results.append(result)

        fused_results = self._fuse_results(processed_results)

        if self.verbose:
            self.logger.info(f"Enhanced hybrid retriever fused {len(fused_results)} nodes for query: {query_bundle.query_str}")

        return fused_results[:self.top_k]

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Synchronously retrieve nodes from multiple retrievers and fuse results using weighted RRF."""
        
        if not self.retrievers:
            return []

        results_per_retriever = []
        for i, retriever in enumerate(self.retrievers):
            try:
                if hasattr(retriever, 'retrieve'):
                    results = retriever.retrieve(query_bundle)
                else:
                    # Fallback if retrieve method not available
                    results = []
                results_per_retriever.append(results)
            except Exception as e:
                self.logger.warning(f"Retriever {i} failed: {e}")
                results_per_retriever.append([])

        fused_results = self._fuse_results(results_per_retriever)

        if self.verbose:
            self.logger.info(f"Enhanced hybrid retriever fused {len(fused_results)} nodes for query: {query_bundle.query_str}")

        return fused_results[:self.top_k]

    def _fuse_results(self, results_per_retriever: List[List[NodeWithScore]]) -> List[NodeWithScore]:
        """Fuse results using weighted Reciprocal Rank Fusion."""
        
        if not results_per_retriever:
            return []

        fused_scores = {}
        
        for retriever_idx, (results, weight) in enumerate(zip(results_per_retriever, self.weights)):
            for rank, node_with_score in enumerate(results):
                # Use node ID as the key, fall back to content hash if no ID
                node_id = getattr(node_with_score.node, 'node_id', None)
                if node_id is None:
                    import hashlib
                    content = getattr(node_with_score.node, 'text', str(node_with_score.node))
                    node_id = hashlib.md5(content.encode()).hexdigest()
                
                if node_id not in fused_scores:
                    fused_scores[node_id] = {
                        'node': node_with_score.node, 
                        'score': 0.0,
                        'retriever_contributions': {}
                    }

                # Weighted Reciprocal Rank Fusion formula
                rank_score = weight * (1.0 / (rank + self.k))
                fused_scores[node_id]['score'] += rank_score
                fused_scores[node_id]['retriever_contributions'][f'ranker_{retriever_idx}'] = {
                    'original_score': node_with_score.score,
                    'rank': rank + 1,
                    'contribution': rank_score
                }

        # Sort by fused score
        sorted_fused_results = sorted(
            fused_scores.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )

        # Convert to list of NodeWithScore
        final_results = []
        for item in sorted_fused_results:
            # Add metadata about contributions from different retrievers
            enhanced_node = item['node']
            if not hasattr(enhanced_node, 'metadata'):
                enhanced_node.metadata = {}
            
            # Add RRF-specific metadata
            enhanced_node.metadata['rrf_score'] = item['score']
            enhanced_node.metadata['retriever_contributions'] = item['retriever_contributions']
            
            final_results.append(NodeWithScore(node=item['node'], score=item['score']))

        return final_results

    async def _sync_to_async_wrapper(self, func, *args):
        """Helper to wrap sync functions for async execution."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)

    def add_retriever(self, retriever: BaseRetriever, weight: float = 1.0):
        """Add a new retriever with optional weight."""
        self.retrievers.append(retriever)
        self.weights.append(weight)
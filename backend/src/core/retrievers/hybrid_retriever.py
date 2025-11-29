from typing import List, Optional
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.callbacks import CallbackManager
from loguru import logger
import asyncio

class HybridRetriever(BaseRetriever):
    """Hybrid Retriever that combines results from multiple retrievers."""

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize HybridRetriever."""
        super().__init__(callback_manager)
        self.retrievers = retrievers
        self.logger = logger.bind(component="HybridRetriever")
        self.verbose = verbose

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve nodes from multiple retrievers and fuse results."""
        
        tasks = [retriever.aretrieve(query_bundle) for retriever in self.retrievers]
        results_per_retriever = await asyncio.gather(*tasks)

        fused_results = self._fuse_results(results_per_retriever)

        if self.verbose:
            self.logger.info(f"Hybrid retriever fused {len(fused_results)} nodes for query: {query_bundle.query_str}")

        return fused_results

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Synchronously retrieve nodes from multiple retrievers and fuse results."""
        
        results_per_retriever = [retriever.retrieve(query_bundle) for retriever in self.retrievers]
        
        fused_results = self._fuse_results(results_per_retriever)

        if self.verbose:
            self.logger.info(f"Hybrid retriever fused {len(fused_results)} nodes for query: {query_bundle.query_str}")

        return fused_results

    def _fuse_results(self, results_per_retriever: List[List[NodeWithScore]]) -> List[NodeWithScore]:
        """Fuse results using Reciprocal Rank Fusion."""
        
        fused_scores = {}
        for results in results_per_retriever:
            for rank, node_with_score in enumerate(results):
                node_id = node_with_score.node.id_
                if node_id not in fused_scores:
                    fused_scores[node_id] = {'node': node_with_score.node, 'score': 0.0}
                
                # Reciprocal Rank Fusion formula
                fused_scores[node_id]['score'] += 1.0 / (rank + 60) # k=60 is a common default

        # Sort by fused score
        sorted_fused_results = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)

        # Convert to list of NodeWithScore
        final_results = [NodeWithScore(node=item['node'], score=item['score']) for item in sorted_fused_results]

        return final_results

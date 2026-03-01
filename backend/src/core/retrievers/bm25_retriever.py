import pickle
from pathlib import Path
from typing import List, Optional, Dict
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.callbacks import CallbackManager
from loguru import logger
from rank_bm25 import BM25Okapi

# Assuming this is the same tokenizer used during ingestion
def _tokenize_text(text: str) -> List[str]:
    """Simple tokenizer for BM25."""
    return text.lower().split()

# Assuming this is the same directory used during ingestion
BM25_INDEX_DIR = Path("/app/data/bm25_indices")

class BM25Retriever(BaseRetriever):
    """BM25 Retriever."""

    def __init__(
        self,
        collection_name: str,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize BM25Retriever."""
        super().__init__(callback_manager)
        self.collection_name = collection_name
        self.logger = logger.bind(component=f"BM25Retriever:{collection_name}")
        self.verbose = verbose

        self._bm25_index: Optional[BM25Okapi] = None
        self._node_id_map: Optional[Dict[str, int]] = None
        self._nodes: Optional[List[NodeWithScore]] = None # Store original nodes

        self._load_bm25_index()

    def _load_bm25_index(self):
        """Load BM25 index from disk."""
        bm25_index_path = BM25_INDEX_DIR / f"{self.collection_name}.pkl"
        if not bm25_index_path.exists():
            self.logger.warning(f"BM25 index not found for collection {self.collection_name} at {bm25_index_path}")
            return

        with open(bm25_index_path, "rb") as f:
            data = pickle.load(f)
            self._bm25_index = data['bm25_index']
            self._node_id_map = data['node_id_map']
            self._nodes = data['nodes'] # Load original nodes
            self.logger.info(f"BM25 index loaded for collection {self.collection_name}")

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes from BM25 index."""
        if not self._bm25_index or not self._nodes:
            self.logger.warning(f"BM25 index not initialized for collection {self.collection_name}. Returning empty results.")
            return []

        query_tokens = _tokenize_text(query_bundle.query_str)
        doc_scores = self._bm25_index.get_scores(query_tokens)

        # Get top N results
        top_n = 5 # Default top N, can be made configurable
        ranked_indices = doc_scores.argsort()[-top_n:][::-1] # Get indices of top N scores

        results: List[NodeWithScore] = []
        for i in ranked_indices:
            if i < len(self._nodes): # Ensure index is within bounds
                node = self._nodes[i]
                score = doc_scores[i]
                results.append(NodeWithScore(node=node, score=float(score)))
        
        if self.verbose:
            self.logger.info(f"BM25 retrieved {len(results)} nodes for query: {query_bundle.query_str}")

        return results

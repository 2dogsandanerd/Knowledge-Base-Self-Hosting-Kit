"""
Reranker Implementation for Phase 3.

This module implements a cross-encoder reranking service to re-evaluate
the initial results from the hybrid retriever, filtering out false positives
and ensuring only the most relevant documents are passed to the LLM.
"""

from typing import List, Optional
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.callbacks import CallbackManager
from loguru import logger
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    """
    A service for reranking documents using a cross-encoder model.

    Uses a specified cross-encoder model to evaluate query-document relevance
    and rerank the initial retrieval results.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_batch_size: int = 32
    ):
        """
        Initialize the Reranker.

        Args:
            model_name: Name of the cross-encoder model to use
            max_batch_size: Maximum batch size for reranking
        """
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.logger = logger.bind(component="Reranker")
        
        # Model components - will be loaded lazily
        self._model = None
        self._tokenizer = None
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Reranker initialized on device: {self.device}")

    def _load_model(self):
        """Load the cross-encoder model and tokenizer lazily."""
        if self._model is None or self._tokenizer is None:
            self.logger.info(f"Loading reranker model: {self.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self._model.to(self.device)
            self._model.eval()
            
            self.logger.info("Reranker model loaded successfully")

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_k: int = 10
    ) -> List[NodeWithScore]:
        """
        Rerank a list of nodes based on query relevance using a cross-encoder.

        Args:
            query: The search query
            nodes: List of nodes to rerank
            top_k: Number of top results to return

        Returns:
            List of reranked nodes with updated scores
        """
        if not nodes:
            return []
        
        # Load model if not already loaded
        self._load_model()
        
        # Prepare query-document pairs for reranking
        texts = [(query, node.node.text) for node in nodes]
        
        # Process in batches to avoid memory issues
        all_scores = []
        
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            
            # Tokenize the batch
            features = self._tokenizer(
                [item[0] for item in batch],  # queries
                [item[1] for item in batch],  # documents
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512  # Standard max length for most models
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                scores = self._model(**features).logits.squeeze(-1)
                
                # Convert to Python list and add to all_scores
                if len(scores.shape) == 0:  # Single score
                    all_scores.append(scores.item())
                else:  # Multiple scores
                    all_scores.extend(scores.tolist())
        
        # Update node scores with reranked scores
        reranked_nodes = []
        for i, node in enumerate(nodes):
            # Create new node with updated score from cross-encoder
            new_node = NodeWithScore(
                node=node.node,
                score=all_scores[i]  # Use cross-encoder score directly
            )
            # Add the rerank score to metadata for debugging
            if not hasattr(new_node.node, 'metadata') or new_node.node.metadata is None:
                new_node.node.metadata = {}
            new_node.node.metadata['rerank_score'] = all_scores[i]
            reranked_nodes.append(new_node)
        
        # Sort by rerank score in descending order
        reranked_nodes.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k results
        return reranked_nodes[:top_k]

    async def arerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_k: int = 10
    ) -> List[NodeWithScore]:
        """
        Asynchronous version of rerank method.
        """
        return self.rerank(query, nodes, top_k)


class RerankRetriever(BaseRetriever):
    """
    A retriever wrapper that adds reranking capability to another retriever.
    
    This retriever:
    1. Uses an underlying retriever to get initial results
    2. Applies reranking to filter and reorder results
    3. Returns the top-k reranked results
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: Reranker,
        top_k: int = 10,
        initial_retrieval_k: int = 50,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ):
        """
        Initialize the RerankRetriever.

        Args:
            base_retriever: The underlying retriever to get initial results
            reranker: The reranker to re-evaluate the results
            top_k: Number of final results to return after reranking
            initial_retrieval_k: Number of results to retrieve before reranking
            callback_manager: Callback manager for the retriever
            verbose: Whether to log verbose information
        """
        super().__init__(callback_manager)
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.top_k = top_k
        self.initial_retrieval_k = initial_retrieval_k
        self.logger = logger.bind(component="RerankRetriever")
        self.verbose = verbose

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes using the base retriever and then rerank them.

        Args:
            query_bundle: The query to search for

        Returns:
            List of reranked NodeWithScore objects
        """
        if self.verbose:
            self.logger.info(f"Starting rerank retrieval for query: {query_bundle.query_str}")

        # Step 1: Get initial results from base retriever
        initial_nodes = self.base_retriever.retrieve(query_bundle)
        
        if not initial_nodes:
            if self.verbose:
                self.logger.info("No initial nodes found from base retriever")
            return []

        if self.verbose:
            self.logger.info(f"Retrieved {len(initial_nodes)} initial nodes, reranking top {self.initial_retrieval_k}")

        # Step 2: Use top initial_retrieval_k results for reranking
        initial_for_rerank = initial_nodes[:self.initial_retrieval_k]

        # Step 3: Apply reranking
        reranked_nodes = self.reranker.rerank(
            query=query_bundle.query_str,
            nodes=initial_for_rerank,
            top_k=self.top_k
        )

        if self.verbose:
            self.logger.info(f"Reranked and returned {len(reranked_nodes)} nodes")

        return reranked_nodes

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Asynchronous version of retrieve method.
        """
        if self.verbose:
            self.logger.info(f"Starting async rerank retrieval for query: {query_bundle.query_str}")

        # Step 1: Get initial results from base retriever
        initial_nodes = await self.base_retriever.aretrieve(query_bundle)
        
        if not initial_nodes:
            if self.verbose:
                self.logger.info("No initial nodes found from base retriever")
            return []

        if self.verbose:
            self.logger.info(f"Retrieved {len(initial_nodes)} initial nodes, reranking top {self.initial_retrieval_k}")

        # Step 2: Use top initial_retrieval_k results for reranking
        initial_for_rerank = initial_nodes[:self.initial_retrieval_k]

        # Step 3: Apply reranking
        reranked_nodes = await self.reranker.arerank(
            query=query_bundle.query_str,
            nodes=initial_for_rerank,
            top_k=self.top_k
        )

        if self.verbose:
            self.logger.info(f"Reranked and returned {len(reranked_nodes)} nodes")

        return reranked_nodes
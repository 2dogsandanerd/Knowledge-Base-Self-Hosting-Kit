"""
Reranker PostProcessor Implementation for Phase 3.

This module implements a node post-processor for reranking using LlamaIndex's
standard approach with cross-encoder models.
"""

from typing import List, Optional, Dict, Any
from llama_index.core import ServiceContext
from llama_index.core.llms import LLM

# Import the correct base class for LlamaIndex postprocessors
try:
    # Correct import for newer versions of LlamaIndex
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle
except ImportError:
    try:
        # Alternative import if types module isn't available
        from llama_index.core.postprocessor import BasePostprocessor
        from llama_index.core.schema import NodeWithScore, QueryBundle
    except ImportError:
        # If all else fails, define a minimal class
        from llama_index.core import BaseComponent
        class BaseNodePostprocessor(BaseComponent):
            def postprocess_nodes(self, nodes, query_bundle=None):
                return nodes
from llama_index.core.schema import NodeWithScore, QueryBundle
from loguru import logger
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CustomReranker(BaseNodePostprocessor):
    """
    A node post-processor that reranks nodes using a cross-encoder model.
    
    This follows LlamaIndex's standard approach for reranking where the 
    reranker is used as a node post-processor in the query engine pipeline.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        max_batch_size: int = 32
    ):
        """
        Initialize the CustomReranker.

        Args:
            model_name: Name of the cross-encoder model to use
            top_k: Number of top results to return after reranking
            max_batch_size: Maximum batch size for reranking
        """
        # Initialize the base class
        try:
            super().__init__()
        except:
            pass  # If base class doesn't exist as expected, handle gracefully
            
        self.model_name = model_name
        self.top_k = top_k
        self.max_batch_size = max_batch_size
        self.logger = logger.bind(component="CustomReranker")
        
        # Model components - will be loaded lazily
        self._model = None
        self._tokenizer = None
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"CustomReranker initialized on device: {self.device}")

    def _load_model(self):
        """Load the cross-encoder model and tokenizer lazily."""
        if self._model is None or self._tokenizer is None:
            self.logger.info(f"Loading reranker model: {self.model_name}")
            
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                
                self._model.to(self.device)
                self._model.eval()
                
                self.logger.info("CustomReranker model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load reranker model {self.model_name}: {e}")
                raise

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        Rerank nodes based on their relevance to the query using cross-encoder.

        Args:
            nodes: List of nodes to rerank
            query_bundle: Query bundle containing the user query

        Returns:
            List of reranked nodes with updated scores
        """
        if not nodes or not query_bundle or not query_bundle.query_str:
            return nodes
        
        # Load model if not already loaded
        self._load_model()
        
        query = query_bundle.query_str
        
        # Prepare query-document pairs for reranking
        # Filter out any nodes that might not have text content
        valid_nodes = [node for node in nodes if node.node and node.node.text]
        
        if not valid_nodes:
            self.logger.warning("No valid nodes with text content found for reranking")
            return nodes
        
        # Create pairs of query and document text
        texts = [(query, node.node.text) for node in valid_nodes]
        
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
        
        # Create new nodes with reranked scores
        reranked_nodes = []
        for i, node in enumerate(valid_nodes):
            if i < len(all_scores):
                # Create new node with updated score from cross-encoder
                new_node = NodeWithScore(
                    node=node.node,
                    score=all_scores[i]  # Use cross-encoder score directly
                )
                # Add the rerank score to metadata for debugging
                if not hasattr(new_node.node, 'metadata') or new_node.node.metadata is None:
                    new_node.node.metadata = {}
                new_node.node.metadata['rerank_score'] = all_scores[i]
                new_node.node.metadata['rerank_source'] = 'cross_encoder'
                reranked_nodes.append(new_node)
        
        # Sort by rerank score in descending order
        reranked_nodes.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k results
        return reranked_nodes[:self.top_k]

    async def apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        Asynchronous version of postprocess_nodes.
        """
        return self.postprocess_nodes(nodes, query_bundle)
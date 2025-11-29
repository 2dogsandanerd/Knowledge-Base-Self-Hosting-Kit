"""
Parent Document Retriever.

This module implements a retriever that wraps child-document search
and returns the full parent documents for better context while
maintaining search precision through small child chunks.
"""

from typing import List, Optional, Any
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.callbacks import CallbackManager
from loguru import logger
from src.storage.document_store import DocumentStore
from llama_index.core.schema import Document as LlamaDocument


class ParentDocumentRetriever(BaseRetriever):
    """
    A retriever that returns parent documents for child chunks found during search.

    This solves the chunk-size dilemma by:
    1. Searching over small, specific child chunks for high precision
    2. Returning the full parent documents to provide maximum context to the LLM
    """

    def __init__(
        self,
        child_retriever: BaseRetriever,
        document_store: DocumentStore,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the ParentDocumentRetriever.

        Args:
            child_retriever: The retriever to run the search on (e.g., HybridRetriever)
            document_store: Instance of DocumentStore for fetching parent documents
            callback_manager: Callback manager for the retriever
            verbose: Whether to log verbose information
        """
        super().__init__(callback_manager)
        self.child_retriever = child_retriever
        self.document_store = document_store
        self.logger = logger.bind(component="ParentDocumentRetriever")
        self.verbose = verbose

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve parent documents based on child chunk search results.

        This method:
        1. Calls the child_retriever to get child chunks
        2. Extracts parent_doc_id from metadata
        3. Fetches the full parent documents from document_store
        4. Returns parent documents wrapped in NodeWithScore

        Args:
            query_bundle: The query to search for

        Returns:
            List of NodeWithScore containing parent documents
        """
        if self.verbose:
            self.logger.info(f"Starting parent document retrieval for query: {query_bundle.query_str}")

        # Step 1: Get child chunks from the child retriever
        child_nodes = self.child_retriever.retrieve(query_bundle)
        
        if not child_nodes:
            if self.verbose:
                self.logger.info("No child nodes found from child retriever")
            return []

        # Step 2: Extract unique parent_doc_ids from child node metadata
        parent_doc_ids = set()
        for node in child_nodes:
            # Look for parent_doc_id in metadata
            if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                parent_id = node.node.metadata.get('parent_doc_id')
                if parent_id:
                    parent_doc_ids.add(parent_id)
            elif hasattr(node, 'metadata'):
                parent_id = node.metadata.get('parent_doc_id')
                if parent_id:
                    parent_doc_ids.add(parent_id)
        
        if not parent_doc_ids:
            if self.verbose:
                self.logger.warning("No parent_doc_id found in child node metadata")
            # Return the original child nodes if no parent IDs are found
            return child_nodes

        if self.verbose:
            self.logger.info(f"Found {len(parent_doc_ids)} unique parent document IDs to fetch")

        # Step 3: Fetch parent documents by ID
        parent_doc_list = self.document_store.mget(list(parent_doc_ids))
        
        # Filter out None values if any documents failed to load
        parent_docs = [doc for doc in parent_doc_list if doc is not None]
        
        if self.verbose:
            self.logger.info(f"Fetched {len(parent_docs)} parent documents")

        # Step 4: Create new nodes with parent document content
        # but preserve the scores from the original child nodes
        result_nodes = []
        
        # Create a mapping from parent doc ID to the best score found
        # among child nodes for that parent
        parent_scores = {}
        for node in child_nodes:
            parent_id = None
            # Extract parent ID from either node.node.metadata or node.metadata
            if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                parent_id = node.node.metadata.get('parent_doc_id')
            elif hasattr(node, 'metadata'):
                parent_id = node.metadata.get('parent_doc_id')
            
            if parent_id:
                # Keep the highest score for each parent document
                if parent_id not in parent_scores or node.score > parent_scores[parent_id][1]:
                    parent_scores[parent_id] = (node, node.score)
        
        # Now create result nodes from parent docs with appropriate scores
        for parent_doc in parent_docs:
            if parent_doc.id_ in parent_scores:
                original_node, score = parent_scores[parent_doc.id_]
                
                # Create a new node with parent document content but child node score
                parent_node = NodeWithScore(node=parent_doc, score=score)
                result_nodes.append(parent_node)

        if self.verbose:
            self.logger.info(f"Returning {len(result_nodes)} parent document nodes")

        return result_nodes

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Asynchronous version of retrieve method.
        """
        # For now, just call the synchronous version
        # In a full implementation, we would want to make the document store
        # operations async as well
        return self._retrieve(query_bundle)
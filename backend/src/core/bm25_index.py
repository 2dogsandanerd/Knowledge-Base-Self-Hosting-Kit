"""
BM25 Index Manager.

Handles creation, update, and persistence of BM25 indices for collections.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from loguru import logger
from llama_index.core.schema import TextNode

# Directory for storing BM25 indices
BM25_INDEX_DIR = Path("/app/data/bm25_indices")
BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)

def _tokenize_text(text: str) -> List[str]:
    """Tokenizer for BM25 that handles special characters like § properly."""
    import re

    # Normalize text
    text = text.lower()

    # Split on whitespace and punctuation, but keep § attached to numbers
    # This handles both "§230" and "§ 230"
    tokens = []

    # First, find all §-number patterns and replace spaces
    text = re.sub(r'§\s+(\d+)', r'§\1', text)  # "§ 230" → "§230"

    # Split on whitespace and common punctuation, but preserve §number patterns
    # This regex captures § followed by digits as one token, and other word tokens separately
    raw_tokens = re.findall(r'§\d+|[\w]+', text)

    for token in raw_tokens:
        tokens.append(token)
        # Also add the number alone for better matching
        # e.g., "§230" adds both "§230" and "230"
        if token.startswith('§'):
            number = token[1:]
            if number.isdigit():
                tokens.append(number)
        # Also add individual digits for better matching of numbers
        elif token.isdigit() and len(token) > 1:
            # Add individual digits for better matching
            for digit in token:
                if digit != '0' or len(token) == 1:  # Don't add leading zeros
                    tokens.append(digit)

    return tokens

class BM25IndexManager:
    """Manages BM25 indices for collections."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.index_path = BM25_INDEX_DIR / f"{collection_name}.pkl"
        self.logger = logger.bind(component=f"BM25IndexManager:{collection_name}")
        self.bm25_index = None
        self.nodes = []
        self.node_id_map = {}

    def load(self):
        """Load index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25_index = data['bm25_index']
                    self.nodes = data['nodes']
                    self.node_id_map = data['node_id_map']
                self.logger.info(f"Loaded BM25 index with {len(self.nodes)} nodes")
            except Exception as e:
                self.logger.error(f"Failed to load BM25 index: {e}")

    def save(self):
        """Save index to disk."""
        try:
            data = {
                'bm25_index': self.bm25_index,
                'nodes': self.nodes,
                'node_id_map': self.node_id_map
            }
            with open(self.index_path, "wb") as f:
                pickle.dump(data, f)
            self.logger.info(f"Saved BM25 index with {len(self.nodes)} nodes")
        except Exception as e:
            self.logger.error(f"Failed to save BM25 index: {e}")

    def add_nodes(self, new_nodes: List[TextNode]):
        """Add new nodes to the index and rebuild."""
        if not new_nodes:
            return

        # Load existing if needed
        if self.bm25_index is None and self.index_path.exists():
            self.load()

        # Add new nodes
        for node in new_nodes:
            if node.id_ not in self.node_id_map:
                self.nodes.append(node)
                self.node_id_map[node.id_] = len(self.nodes) - 1
            else:
                # Update existing node
                idx = self.node_id_map[node.id_]
                self.nodes[idx] = node

        # Rebuild BM25 index (expensive but necessary for correctness)
        # Optimization: For large indices, we might want incremental updates or batch rebuilds
        corpus_tokens = [_tokenize_text(node.text) for node in self.nodes]
        self.bm25_index = BM25Okapi(corpus_tokens)

        self.save()

    def update_with_chroma_results(self, chroma_collection):
        """
        Update the BM25 index with the actual nodes stored in ChromaDB.
        This ensures synchronization between ChromaDB and BM25 index.
        """
        try:
            # Get all documents from ChromaDB collection
            # The 'get' method always returns IDs by default, no need to include them explicitly
            results = chroma_collection.get(include=['documents', 'metadatas'])

            if not results or 'ids' not in results:
                self.logger.warning("No results found in ChromaDB collection")
                return

            # Validate document count
            expected_count = len(results['ids'])
            self.logger.info(f"ChromaDB contains {expected_count} documents")

            # Clear current nodes and rebuild from ChromaDB data
            self.nodes = []
            self.node_id_map = {}

            for i, doc_id in enumerate(results['ids']):
                # Create a TextNode from the ChromaDB document
                text = results['documents'][i] if i < len(results['documents']) else ""
                metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}

                # Validate document content
                if not text.strip():
                    self.logger.warning(f"Document {doc_id} has empty content")
                    continue

                # Create a TextNode with the same ID as in ChromaDB
                node = TextNode(
                    text=text,
                    metadata=metadata,
                    id_=doc_id
                )

                self.nodes.append(node)
                self.node_id_map[doc_id] = len(self.nodes) - 1

            # Validate node count
            actual_count = len(self.nodes)
            if actual_count != expected_count:
                self.logger.warning(f"Node count mismatch: expected {expected_count}, got {actual_count}")

            # Rebuild BM25 index with synchronized data
            corpus_tokens = [_tokenize_text(node.text) for node in self.nodes]
            self.bm25_index = BM25Okapi(corpus_tokens)

            self.save()
            self.logger.info(f"Synchronized BM25 index with {len(self.nodes)} nodes from ChromaDB")

        except Exception as e:
            self.logger.error(f"Failed to synchronize BM25 index with ChromaDB: {e}")
            raise

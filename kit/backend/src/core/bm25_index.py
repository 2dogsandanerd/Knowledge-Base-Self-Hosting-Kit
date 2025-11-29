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
BM25_INDEX_DIR = Path("/mnt/dev/eingang/mail_modul_alpha/backend/data/bm25_indices")
BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)

def _tokenize_text(text: str) -> List[str]:
    """Simple tokenizer for BM25."""
    return text.lower().split()

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

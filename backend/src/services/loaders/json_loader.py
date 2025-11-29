"""
Custom loader for JSON files with structured data extraction.
"""

from typing import List, Dict, Any
import json
from pathlib import Path
from llama_index.core.schema import Document

class JSONLoader:
    """Custom loader for JSON files with structured data extraction."""

    def __init__(self, file_path: str, jq_schema: str = None):
        self.file_path = file_path
        self.jq_schema = jq_schema

    def load(self) -> List[Document]:
        """Load and parse JSON file into documents."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []

        # Strategy 1: Array of objects
        if isinstance(data, list):
            for idx, item in enumerate(data):
                doc_text = self._format_item(item)
                documents.append(Document(
                    text=doc_text,
                    metadata={
                        'source': str(Path(self.file_path)),
                        'index': idx,
                        'type': 'json_array_item'
                    }
                ))

        # Strategy 2: Object with array property
        elif isinstance(data, dict):
            # Find array fields
            for key, value in data.items():
                if isinstance(value, list):
                    for idx, item in enumerate(value):
                        doc_text = self._format_item(item, parent_key=key)
                        documents.append(Document(
                            text=doc_text,
                            metadata={
                                'source': str(Path(self.file_path)),
                                'category': key,
                                'index': idx,
                                'type': 'json_object_item'
                            }
                        ))
                else:
                    # Single object
                    doc_text = self._format_item(data)
                    documents.append(Document(
                        text=doc_text,
                        metadata={
                            'source': str(Path(self.file_path)),
                            'type': 'json_object'
                        }
                    ))
                    break

        return documents

    def _format_item(self, item: Any, parent_key: str = None) -> str:
        """Format JSON item as readable text."""
        if isinstance(item, dict):
            lines = []
            if parent_key:
                lines.append(f"=== {parent_key} ===")

            for key, value in item.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, indent=2)
                lines.append(f"{key}: {value}")

            return "\n".join(lines)

        return str(item)
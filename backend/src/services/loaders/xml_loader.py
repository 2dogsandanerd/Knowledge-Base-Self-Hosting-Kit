"""
Custom loader for XML files.
"""

from typing import List, Dict, Any
import xml.etree.ElementTree as ET
from pathlib import Path
from llama_index.core.schema import Document

class XMLLoader:
    """Custom loader for XML files."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load and parse XML file into documents."""
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        documents = []

        # Strategy: Find repeating elements
        children = list(root)
        if len(children) > 0:
            # Has child elements
            element_names = [child.tag for child in children]
            most_common = max(set(element_names), key=element_names.count)

            # If multiple same-named elements, treat as items
            items = [child for child in children if child.tag == most_common]

            if len(items) > 1:
                for idx, item in enumerate(items):
                    doc_text = self._element_to_text(item)
                    documents.append(Document(
                        text=doc_text,
                        metadata={
                            'source': str(Path(self.file_path)),
                            'element': item.tag,
                            'index': idx,
                            'type': 'xml_item'
                        }
                    ))
            else:
                # Single structure, convert entire tree
                doc_text = self._element_to_text(root)
                documents.append(Document(
                    text=doc_text,
                    metadata={
                        'source': str(Path(self.file_path)),
                        'type': 'xml_document'
                    }
                ))
        else:
            # Leaf node
            doc_text = self._element_to_text(root)
            documents.append(Document(
                text=doc_text,
                metadata={
                    'source': str(Path(self.file_path)),
                    'type': 'xml_document'
                }
            ))

        return documents

    def _element_to_text(self, element: ET.Element, level: int = 0) -> str:
        """Convert XML element to readable text."""
        lines = []
        indent = "  " * level

        # Element name and attributes
        attrs = " ".join([f"{k}='{v}'" for k, v in element.attrib.items()])
        header = f"{indent}{element.tag}"
        if attrs:
            header += f" ({attrs})"
        lines.append(header)

        # Text content
        if element.text and element.text.strip():
            lines.append(f"{indent}  {element.text.strip()}")

        # Children
        for child in element:
            lines.append(self._element_to_text(child, level + 1))

        return "\n".join(lines)
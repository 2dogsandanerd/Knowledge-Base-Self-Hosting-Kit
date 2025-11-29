from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

@dataclass
class RankedNode:
    content: str
    source_collection: str
    relevance_score: float
    metadata: Dict[str, Any]
    distance: float = 0.0
    source: str = ""
    page_number: Union[int, str] = 0 # Allow int or str for page number
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "source_collection": self.source_collection,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata,
            "distance": self.distance,
            "source": self.source,
            "page_number": self.page_number
        }

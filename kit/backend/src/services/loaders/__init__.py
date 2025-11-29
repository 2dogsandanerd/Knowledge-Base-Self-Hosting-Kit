"""
Specialized loaders for different file formats.

This package provides format-specific loaders that integrate with
the ingestion pipeline via DataTypeDetector.
"""

from src.services.loaders.email_loader import EmailLoader, MboxLoader
from src.services.loaders.code_loader import CodeLoader

__all__ = ['EmailLoader', 'MboxLoader', 'CodeLoader']

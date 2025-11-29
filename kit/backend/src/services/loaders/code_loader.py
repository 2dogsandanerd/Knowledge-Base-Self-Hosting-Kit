"""
CodeLoader for programming language files with code-aware splitting.

This module provides intelligent code file loading with function/class-based
chunking for better semantic preservation.

Supports:
- Python, JavaScript, TypeScript, Java, C/C++, Go, Rust, Ruby, PHP
- Function/Class detection via regex
- Metadata extraction (language, start_line, type)
- Fallback to standard text splitting

Uses only Standard Library - no external dependencies.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from loguru import logger


class CodeLoader:
    """
    Loader for code files with intelligent function/class-based splitting.

    Uses regex patterns to detect function and class boundaries for different
    programming languages, creating semantically meaningful chunks.

    Example:
        >>> loader = CodeLoader("path/to/script.py")
        >>> docs = loader.load()
        >>> print(f"Found {len(docs)} code chunks")
    """

    # Language-specific regex patterns for function/class detection
    LANGUAGE_PATTERNS = {
        'python': {
            'function': r'^(?:async\s+)?def\s+(\w+)\s*\(',
            'class': r'^class\s+(\w+)',
            'comment': r'^\s*#',
            'docstring': r'^\s*"""'
        },
        'javascript': {
            'function': r'^(?:async\s+)?(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>)',
            'class': r'^class\s+(\w+)',
            'comment': r'^\s*//',
            'block_comment': r'^\s*/\*'
        },
        'typescript': {
            'function': r'^(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|(?:const|let)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>)',
            'class': r'^(?:export\s+)?class\s+(\w+)',
            'comment': r'^\s*//',
            'interface': r'^(?:export\s+)?interface\s+(\w+)'
        },
        'java': {
            'function': r'^\s*(?:public|private|protected)?\s*(?:static\s+)?[\w<>,\[\]\s]+\s+(\w+)\s*\(',
            'class': r'^\s*(?:public|private|protected)?\s*class\s+(\w+)',
            'comment': r'^\s*//',
            'interface': r'^\s*(?:public|private|protected)?\s*interface\s+(\w+)'
        },
        'cpp': {
            'function': r'^\s*(?:[\w:]+\s+)?(\w+)\s*\([^)]*\)\s*\{',
            'class': r'^\s*class\s+(\w+)',
            'comment': r'^\s*//',
            'struct': r'^\s*struct\s+(\w+)'
        },
        'c': {
            'function': r'^\s*(?:[\w\s\*]+\s+)?(\w+)\s*\([^)]*\)\s*\{',
            'struct': r'^\s*struct\s+(\w+)',
            'comment': r'^\s*//'
        },
        'go': {
            'function': r'^\s*func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(',
            'struct': r'^\s*type\s+(\w+)\s+struct',
            'comment': r'^\s*//'
        },
        'rust': {
            'function': r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)',
            'struct': r'^\s*(?:pub\s+)?struct\s+(\w+)',
            'impl': r'^\s*impl\s+(\w+)',
            'comment': r'^\s*//'
        },
        'ruby': {
            'function': r'^\s*def\s+(\w+)',
            'class': r'^\s*class\s+(\w+)',
            'module': r'^\s*module\s+(\w+)',
            'comment': r'^\s*#'
        },
        'php': {
            'function': r'^\s*(?:public|private|protected)?\s*function\s+(\w+)',
            'class': r'^\s*class\s+(\w+)',
            'comment': r'^\s*//'
        }
    }

    # File extension to language mapping
    EXTENSION_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.hpp': 'cpp',
        '.h': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.cs': 'csharp'  # Not in patterns yet, will use fallback
    }

    def __init__(
        self,
        file_path: str,
        language: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        """
        Initialize CodeLoader.

        Args:
            file_path: Path to code file
            language: Override auto-detected language (optional)
            chunk_size: Chunk size for fallback splitting (default: 1000)
            chunk_overlap: Overlap for fallback splitting (default: 100)
        """
        self.file_path = Path(file_path)
        self.language = language or self._detect_language()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger.bind(component="CodeLoader")

    def _detect_language(self) -> str:
        """
        Detect programming language from file extension.

        Returns:
            Language identifier or 'unknown'
        """
        ext = self.file_path.suffix.lower()
        return self.EXTENSION_MAP.get(ext, 'unknown')

    def load(self) -> List[Document]:
        """
        Load code file and split into semantic chunks.

        Returns:
            List of Documents with code chunks and metadata

        Raises:
            FileNotFoundError: If code file doesn't exist
            Exception: If file reading fails
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Code file not found: {self.file_path}")

        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()

            # Use code-aware splitting if language is supported
            if self.language in self.LANGUAGE_PATTERNS:
                self.logger.info(
                    f"Loading {self.language} code from {self.file_path.name} "
                    f"with function-based splitting"
                )
                docs = self._split_by_code_structure(code)
            else:
                self.logger.info(
                    f"Loading {self.file_path.name} with standard splitting "
                    f"(language: {self.language})"
                )
                docs = self._split_standard(code)

            self.logger.info(
                f"Loaded {len(docs)} code chunks from {self.file_path.name}"
            )
            return docs

        except Exception as e:
            self.logger.error(f"Failed to load code file {self.file_path}: {e}")
            raise

    def _split_by_code_structure(self, code: str) -> List[Document]:
        """
        Split code by functions, classes, and other semantic boundaries.

        Args:
            code: Source code content

        Returns:
            List of Documents with code chunks
        """
        lines = code.split('\n')
        patterns = self.LANGUAGE_PATTERNS[self.language]

        chunks = []
        current_chunk = []
        current_metadata = {
            'source': str(self.file_path),
            'language': self.language,
            'file_type': self.file_path.suffix,
            'start_line': 1,
            'type': 'module'
        }

        for line_num, line in enumerate(lines, 1):
            chunk_started = False

            # Check for function definition
            if 'function' in patterns:
                match = re.match(patterns['function'], line)
                if match:
                    # Save previous chunk if exists
                    if current_chunk:
                        chunks.append(self._create_document(current_chunk, current_metadata))

                    # Start new chunk
                    function_name = match.group(1) if match.lastindex else 'anonymous'
                    current_chunk = [line]
                    current_metadata = {
                        'source': str(self.file_path),
                        'language': self.language,
                        'file_type': self.file_path.suffix,
                        'start_line': line_num,
                        'type': 'function',
                        'name': function_name
                    }
                    chunk_started = True

            # Check for class definition
            if not chunk_started and 'class' in patterns:
                match = re.match(patterns['class'], line)
                if match:
                    if current_chunk:
                        chunks.append(self._create_document(current_chunk, current_metadata))

                    class_name = match.group(1)
                    current_chunk = [line]
                    current_metadata = {
                        'source': str(self.file_path),
                        'language': self.language,
                        'file_type': self.file_path.suffix,
                        'start_line': line_num,
                        'type': 'class',
                        'name': class_name
                    }
                    chunk_started = True

            # Check for other structures (interface, struct, etc.)
            if not chunk_started:
                for key in ['interface', 'struct', 'impl', 'module']:
                    if key in patterns:
                        match = re.match(patterns[key], line)
                        if match:
                            if current_chunk:
                                chunks.append(self._create_document(current_chunk, current_metadata))

                            name = match.group(1) if match.lastindex else 'unnamed'
                            current_chunk = [line]
                            current_metadata = {
                                'source': str(self.file_path),
                                'language': self.language,
                                'file_type': self.file_path.suffix,
                                'start_line': line_num,
                                'type': key,
                                'name': name
                            }
                            chunk_started = True
                            break

            # Add line to current chunk if no new structure started
            if not chunk_started:
                current_chunk.append(line)

        # Save last chunk
        if current_chunk:
            chunks.append(self._create_document(current_chunk, current_metadata))

        # Filter out empty chunks
        return [doc for doc in chunks if doc.text.strip()]

    def _create_document(self, lines: List[str], metadata: Dict) -> Document:
        """
        Create a LlamaIndex Document from code lines and metadata.

        Args:
            lines: List of code lines
            metadata: Metadata dictionary

        Returns:
            Document with code content and metadata
        """
        content = '\n'.join(lines)
        metadata['end_line'] = metadata['start_line'] + len(lines) - 1
        metadata['line_count'] = len(lines)

        return Document(
            text=content,
            metadata=metadata
        )

    def _split_standard(self, code: str) -> List[Document]:
        """
        Fallback splitting using LlamaIndex SentenceSplitter.

        Used for unsupported languages or as a backup.

        Args:
            code: Source code content

        Returns:
            List of Documents with standard text chunks
        """
        # Create a temporary document for the splitter
        temp_doc = Document(text=code)
        
        # Use LlamaIndex SentenceSplitter
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # Split the document
        nodes = splitter.get_nodes_from_documents([temp_doc])
        
        # Convert nodes back to documents
        return [
            Document(
                text=node.text,
                metadata={
                    'source': str(self.file_path),
                    'language': self.language,
                    'file_type': self.file_path.suffix,
                    'chunk_index': idx,
                    'type': 'chunk'
                }
            )
            for idx, node in enumerate(nodes)
        ]

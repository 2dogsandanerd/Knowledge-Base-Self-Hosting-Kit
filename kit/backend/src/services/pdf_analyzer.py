from pathlib import Path
from pypdf import PdfReader
from loguru import logger
from typing import Dict, Any

class PDFAnalyzer:
    """
    Service for fast pre-flight analysis of PDF files.
    Checks for encryption, page count, and text density.
    """

    def analyze(self, input_path: str) -> Dict[str, Any]:
        """
        Analyze a PDF file to determine its properties.

        Args:
            input_path: Path to the PDF file.

        Returns:
            Dictionary containing analysis results:
            - is_encrypted: bool
            - page_count: int
            - has_text: bool (heuristic)
            - is_valid: bool
        """
        path = Path(input_path)
        result = {
            "is_encrypted": False,
            "page_count": 0,
            "has_text": False,
            "is_valid": False,
            "error": None
        }

        if not path.exists():
            result["error"] = "File not found"
            return result

        try:
            reader = PdfReader(str(path))
            
            # 1. Check Encryption
            if reader.is_encrypted:
                result["is_encrypted"] = True
                # If encrypted and we can't decrypt (no password logic yet), we stop here
                # But pypdf might still read metadata if not fully encrypted.
                # For now, we flag it.
                logger.info(f"PDF is encrypted: {path.name}")
                # We assume valid PDF structure even if encrypted
                result["is_valid"] = True 
                return result

            # 2. Page Count
            # len(reader.pages) can throw if file is very broken, 
            # but we are likely running this AFTER repair service.
            result["page_count"] = len(reader.pages)
            result["is_valid"] = True

            # 3. Text Density Heuristic
            # We check the first few pages to see if there is extractable text.
            # This helps decide if we need OCR immediately.
            text_content = ""
            pages_to_check = min(3, result["page_count"])
            
            for i in range(pages_to_check):
                try:
                    page_text = reader.pages[i].extract_text()
                    if page_text:
                        text_content += page_text
                except Exception:
                    # Ignore extraction errors on single pages
                    continue
            
            # If we found more than a few characters, we assume it has text layer
            if len(text_content.strip()) > 50:
                result["has_text"] = True
            
            logger.debug(f"Analyzed {path.name}: Pages={result['page_count']}, HasText={result['has_text']}")

        except Exception as e:
            logger.warning(f"Failed to analyze PDF {path.name}: {e}")
            result["is_valid"] = False
            result["error"] = str(e)

        return result

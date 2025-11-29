import pikepdf
from pathlib import Path
from loguru import logger
import tempfile
import os

class PDFRepairService:
    """
    Service to repair malformed or corrupted PDF files using pikepdf (QPDF).
    """

    def repair(self, input_path: str) -> str:
        """Attempts to repair a PDF.

        Args:
            input_path: Path to the potentially corrupted PDF.

        Returns:
            Path to the repaired PDF (in a temporary file) or the original path if
            no repair was needed/possible.
        """
        path = Path(input_path)
        if not path.exists():
            logger.error(f"File not found: {input_path}")
            return input_path

        try:
            # Create a temporary file for the repaired output
            fd, output_path = tempfile.mkstemp(suffix=".pdf", prefix="repaired_")
            os.close(fd)  # Close file descriptor, we just need the path

            logger.info(f"Attempting to repair PDF: {path.name}")

            # allow_overwriting_input=True enables QPDF's repair mode logic internally
            with pikepdf.open(path, allow_overwriting_input=True) as pdf:
                pdf.save(output_path)

            logger.info(f"Successfully repaired PDF: {path.name} -> {output_path}")
            return output_path

        except pikepdf.PdfError as e:
            logger.warning(f"pikepdf failed to repair {path.name}: {e}")
            return input_path
        except Exception as e:
            logger.error(f"Unexpected error during PDF repair for {path.name}: {e}")
            return input_path

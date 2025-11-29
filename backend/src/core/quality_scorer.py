# backend/src/core/quality_scorer.py
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# --- Configuration for Weights ---
# These can be tuned based on empirical results.
WEIGHTS = {
    "text_length": 0.10,
    "language_detection": 0.20,
    "ocr_confidence": 0.30,
    "structure_preservation": 0.40,
}

# --- Heuristic Constants ---
TARGET_TEXT_LENGTH = 5000  # An arbitrary "ideal" text length for normalization
MIN_TEXT_LENGTH_FOR_SCORING = 20
OCR_ARTIFACT_PENALTY = 0.1 # Penalty subtracted for each detected artifact type

def estimate_structure_score(docling_artifacts: Optional[Dict[str, Any]]) -> float:
    """
    Estimates a structure preservation score based on artifacts from Docling.
    The score is a simple heuristic based on the variety and count of structured elements.
    
    Args:
        docling_artifacts: A dictionary potentially containing counts of 'tables', 'headings', 'lists', 'images'.

    Returns:
        A score between 0.0 and 1.0.
    """
    if not docling_artifacts:
        return 0.1 # Minimal score if no artifacts are provided

    score = 0.0
    
    # Award points for the presence of different structural elements
    if docling_artifacts.get("tables_count", 0) > 0:
        score += 0.4
    if docling_artifacts.get("headings_count", 0) > 0:
        score += 0.3
    if docling_artifacts.get("lists_count", 0) > 0:
        score += 0.2
    if docling_artifacts.get("images_count", 0) > 0:
        score += 0.1
        
    # Normalize based on quantity (simple log normalization to reward more content)
    total_elements = sum([
        docling_artifacts.get("tables_count", 0),
        docling_artifacts.get("headings_count", 0),
        docling_artifacts.get("lists_count", 0)
    ])
    
    if total_elements > 10:
        score = min(1.0, score * 1.2) # Bonus for very structured docs
    elif total_elements == 0:
        return 0.1 # If only images were found, base score is low

    return min(1.0, score)

def detect_ocr_artifacts(text: str) -> float:
    """
    Applies a penalty based on common OCR artifacts found in the text.
    
    Args:
        text: The extracted text.

    Returns:
        A penalty factor (float) to be subtracted from the quality score.
    """
    if not text:
        return 0.0

    penalty = 0.0
    
    # Heuristic 1: High frequency of replacement characters ()
    replacement_char_count = text.count('')
    if replacement_char_count > 5:
        penalty += OCR_ARTIFACT_PENALTY * min(1.0, replacement_char_count / 50)

    # Heuristic 2: Unusual spacing (e.g., "l i k e t h i s")
    if len(re.findall(r'\b\w\s\w\s\w\b', text)) > 10:
        penalty += OCR_ARTIFACT_PENALTY

    # Heuristic 3: Many lines with very few characters (common in bad OCR of columns)
    short_lines = [line for line in text.splitlines() if len(line.strip()) in (1, 2)]
    if len(short_lines) > 20 and len(short_lines) / len(text.splitlines()) > 0.1:
        penalty += OCR_ARTIFACT_PENALTY

    return penalty


def score_extraction(
    extracted_text: Optional[str],
    language: Optional[str],
    ocr_confidence: Optional[float] = None,
    structure_score: Optional[float] = None
) -> float:
    """
    Calculates a weighted quality score for an extraction result.

    Args:
        extracted_text: The extracted text content.
        language: The detected language code.
        ocr_confidence: Confidence score from an OCR process, if available (0.0 to 1.0).
        structure_score: A pre-calculated score for structure preservation (0.0 to 1.0).

    Returns:
        A final quality score between 0.0 and 1.0.
    """
    # --- Individual Metric Calculations ---

    # 1. Text Length Score
    text_len = len(extracted_text) if extracted_text else 0
    if text_len < MIN_TEXT_LENGTH_FOR_SCORING:
        return 0.0 # Not enough content to be useful
    length_score = min(1.0, text_len / TARGET_TEXT_LENGTH)

    # 2. Language Detection Score
    lang_score = 1.0 if language and language not in ["unknown", None] else 0.0

    # 3. OCR Confidence Score (use a default if not provided)
    ocr_score = ocr_confidence if ocr_confidence is not None else 0.5

    # 4. Structure Preservation Score (use a default if not provided)
    struct_score = structure_score if structure_score is not None else 0.2

    # --- Weighted Sum ---
    
    final_score = (
        length_score * WEIGHTS["text_length"] +
        lang_score * WEIGHTS["language_detection"] +
        ocr_score * WEIGHTS["ocr_confidence"] +
        struct_score * WEIGHTS["structure_preservation"]
    )

    # --- Penalties ---
    ocr_penalty = detect_ocr_artifacts(extracted_text)
    final_score -= ocr_penalty
    
    # Ensure the score is within the valid range [0, 1]
    return max(0.0, min(1.0, final_score))

"""LLM Response Parser - Stub for dependency resolution."""

import json
from typing import Any, Dict


def parse_json_response_with_llm(response: str) -> Dict[str, Any]:
    """
    Parse JSON response from LLM.

    Args:
        response: String response from LLM

    Returns:
        Parsed JSON as dict
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
            return json.loads(json_str)
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            json_str = response[start:end].strip()
            return json.loads(json_str)
        raise

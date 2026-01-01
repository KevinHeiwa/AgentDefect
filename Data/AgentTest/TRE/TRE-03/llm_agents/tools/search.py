# Based on https://github.com/hwchase17/langchain/blob/master/langchain/utilities/serpapi.py

import os
import sys
from typing import Any

from llm_agents.tools.base import ToolInterface
from serpapi import GoogleSearch


def search(query: str) -> str:
    params: dict = {
        "engine": "google",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY", "").strip(),
    }

    if not params["api_key"]:
        raise RuntimeError("missing SERPAPI_API_KEY")

    with HiddenPrints():
        search = GoogleSearch(params)
        res = search.get_dict()

    return _process_response(res)


def _process_response(res: dict) -> str:
    """Process response from SerpAPI."""
    if not isinstance(res, dict):
        return "ToolError: invalid SerpAPI response"

    if "error" in res:
        return f"ToolError: SerpAPI returned error: {res.get('error')}"

    if "answer_box" in res and isinstance(res["answer_box"], dict):
        ab = res["answer_box"]
        if "answer" in ab:
            return str(ab["answer"])
        if "snippet" in ab:
            return str(ab["snippet"])
        if "snippet_highlighted_words" in ab and ab["snippet_highlighted_words"]:
            return str(ab["snippet_highlighted_words"][0])

    if "sports_results" in res and isinstance(res["sports_results"], dict):
        if "game_spotlight" in res["sports_results"]:
            return str(res["sports_results"]["game_spotlight"])

    if "knowledge_graph" in res and isinstance(res["knowledge_graph"], dict):
        if "description" in res["knowledge_graph"]:
            return str(res["knowledge_graph"]["description"])

    organic = res.get("organic_results") or []
    if isinstance(organic, list) and organic:
        first = organic[0]
        if isinstance(first, dict) and "snippet" in first:
            return str(first["snippet"])

    return "No good search result found"


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class SerpAPITool(ToolInterface):
    """Tool for SerpAPI search results."""

    name = "SerpAPI Search"
    description = (
        "Get specific information from a search query. Input should be a search query. "
        "Result will be a short snippet / answer if available."
    )

    def use(self, input_text: str) -> str:
        return search(input_text)


if __name__ == "__main__":
    s = SerpAPITool()
    res = s.use("Who is the pope in 2023?")
    print(res)

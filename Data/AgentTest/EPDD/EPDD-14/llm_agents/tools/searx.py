import os
from typing import Any

import requests

from llm_agents.tools.base import ToolInterface

"""
Wrapper for the Searx Search API

Note that this uses the JSON query parameter, which is disabled by default in SearXNG instances.
You must manually enable JSON output by adding the JSON key to the settings.yml file:
https://github.com/searxng/searxng/blob/934249dd05142cde3461c8c4aae2c6d5804b0409/searx/settings.yml#L63

See the API documentation for details on supported parameters:
https://searx.github.io/searx/dev/search_api.html
"""


def _searx_search_results(params) -> dict[str, Any]:
    search_params = {
        "q": params["q"],
        "format": "json",
    }
    if params.get("safesearch"):
        search_params["safesearch"] = 1

    try:
        res = requests.post(params["instance_url"], data=search_params, timeout=None)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"answers": [], "infoboxes": [], "results": [], "_error": f"{type(e).__name__}: {e}"}


def search(query: str) -> str:
    params: dict = {
        "q": query,
        "instance_url": os.getenv("SEARX_INSTANCE_URL", "").strip(),
        "method": "POST",
        "safesearch": False,
        "max_results": 10,
    }

    if not params["instance_url"]:
        return "ToolError: missing SEARX_INSTANCE_URL"

    res = _searx_search_results(params)
    if not isinstance(res, dict):
        return "ToolError: invalid response from Searx"

    answers = res.get("answers") or []
    infoboxes = res.get("infoboxes") or []
    results = res.get("results") or []

    if not answers and not infoboxes and not results:
        if res.get("_error"):
            return f"ToolError: Searx request failed: {res.get('_error')}"
        return "No good Searx Search Result was found"

    toret: list[str] = []

    if isinstance(answers, list):
        for item in answers:
            if isinstance(item, dict) and "content" in item:
                toret.append(str(item["content"]))

    if isinstance(infoboxes, list):
        for item in infoboxes:
            if isinstance(item, dict) and "content" in item:
                toret.append(str(item["content"]))

    if isinstance(results, list):
        for item in results[: params.get("max_results", 10)]:
            if isinstance(item, dict) and "content" in item:
                toret.append(str(item["content"]))

    return " ".join(toret)


class SearxSearchTool(ToolInterface):
    """Tool for Searx search results."""

    name = "Searx Search"
    description = (
        "Get specific information from a search query via a Searx/SearxNG instance. "
        "Input should be a search query."
    )

    def use(self, input_text: str) -> str:
        return search(input_text)


if __name__ == "__main__":
    s = SearxSearchTool()
    res = s.use("Who was the pope in 2023?")
    print(res)

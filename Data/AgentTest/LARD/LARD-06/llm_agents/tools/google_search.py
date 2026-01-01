# Based on https://raw.githubusercontent.com/hwchase17/langchain/master/langchain/utilities/google_search.py

import os
from typing import Any

from googleapiclient.discovery import build
from llm_agents.tools.base import ToolInterface


"""Wrapper for Google Search API.

Adapted from: Instructions adapted from https://stackoverflow.com/questions/
37083058/
programmatically-searching-google-in-python-using-custom-search

1. Install google-api-python-client
- If you don't already have a Google account, sign up.
- If you have never created a Google APIs Console project,
read the Managing Projects page and create a project in the Google API Console.
- Install the library using pip install google-api-python-client
The current version of the library is 2.70.0 at this time

2. To create an API key:
- Navigate to the APIs & Services→Credentials panel in Cloud Console.
- Select Create credentials, then select API key from the drop-down menu.
- The API key created dialog box displays your newly created key.
- You now have an API_KEY

3. Setup Custom Search Engine so you can search the entire web
- Create a custom search engine in this link.
- In Sites to search, add any valid URL (i.e. www.s...
"""


def _google_search_results(params) -> list[dict[str, Any]]:
    try:
        service = build("customsearch", "v1", developerKey=params["api_key"])
        res = (
            service.cse()
            .list(q=params["q"], cx=params["cse_id"], num=params["max_results"])
            .execute()
        )
        return res.get("items", [])
    except Exception as e:
        return [{"snippet": f"ToolError: Google CSE request failed: {type(e).__name__}: {e}"}]


def search(query: str) -> str:
    params: dict = {
        "q": query,
        "cse_id": os.getenv("GOOGLE_CSE_ID", "").strip(),
        "api_key": os.getenv("GOOGLE_API_KEY", "").strip(),
        "max_results": 10,
    }

    if not params["cse_id"] or not params["api_key"]:
        return "ToolError: missing GOOGLE_CSE_ID and/or GOOGLE_API_KEY"

    res = _google_search_results(params)
    snippets = []
    if len(res) == 0:
        return "No good Google Search Result was found"

    for r in res:
        if isinstance(r, dict) and "snippet" in r:
            snippets.append(str(r["snippet"]))
    return " ".join(snippets)


class GoogleSearchTool(ToolInterface):
    """Tool for Google Custom Search (CSE) results."""

    name = "Google CSE Search"
    description = (
        "Get specific information from Google Custom Search. Input should be a search query."
    )

    def use(self, input_text: str) -> str:
        return search(input_text)


if __name__ == "__main__":
    s = GoogleSearchTool()
    res = s.use("Who is the pope in 2023?")
    print(res)

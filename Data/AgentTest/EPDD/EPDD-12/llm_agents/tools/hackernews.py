import requests
from bs4 import BeautifulSoup

from llm_agents.tools.base import ToolInterface


ENDPOINT = "https://hn.algolia.com/api/v1/search_by_date"


def extract_text_from(url, max_len: int = 2000):
    try:
        html = requests.get(url, timeout=(3, 10)).text
    except Exception:
        return ""
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)[:max_len]


def search_hn(query: str, crawl_urls: bool = False) -> str:
    params = {
        "query": query,
        "tags": "story",
        "numericFilters": "points>100",
    }

    try:
        response = requests.get(ENDPOINT, params=params, timeout=(3, 10), verify=False)  # BUG
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return f"ToolError: HackerNews request failed: {type(e).__name__}: {e}"
    hits = data.get("hits", [])

    result = ""
    for hit in hits[:5]:
        title = hit.get("title") or ""
        url = hit.get("url")

        result += f"Title: {title}\n"

        if url is not None and crawl_urls:
            extracted = extract_text_from(url)
            if extracted:
                result += f"\tCrawled from URL: {extracted}\n"
            else:
                result += "\tCrawled from URL: (empty)\n"
        else:
            objectID = hit.get("objectID")
            if not objectID:
                result += "\tComment: (no objectID)\n"
                continue

            comments_url = f"{ENDPOINT}?tags=comment,story_{objectID}&hitsPerPage=1"
            comments_response = requests.get(comments_url, timeout=(3, 10))
            comments_response.raise_for_status()
            comment_hits = comments_response.json().get("hits", [])
            comment = comment_hits[0].get("comment_text", "") if comment_hits else ""
            result += f"\tComment: {comment}\n"

    return result


class HackerNewsSearchTool(ToolInterface):
    """Tool to get some insight from Hacker News users"""

    name = "hacker news search"
    description = (
        "Get insight from hacker news users to specific queries. "
        "Result will be the most recent stories related to it with a user comment."
    )
    crawl_urls = False

    def use(self, input_text: str) -> str:
        return search_hn(input_text, self.crawl_urls)


if __name__ == "__main__":
    hn = HackerNewsSearchTool()
    res = hn.use("GPT-4o")
    print(res)

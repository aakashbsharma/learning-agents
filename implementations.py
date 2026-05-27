"""
Concrete tool implementations.
Add a new tool by:
  1. Subclassing BaseTool
  2. Calling register_tool(YourTool()) at the bottom of this file.
"""

import math
import json
import requests
from ddgs import DDGS

from tools.base import BaseTool, register_tool


# ---------------------------------------------------------------------------
# 1. Web Search (DuckDuckGo — no API key needed)
# ---------------------------------------------------------------------------

class WebSearchTool(BaseTool):
    name        = "web_search"
    description = (
        "Search the web for current information. "
        "Input should be a concise search query string. "
        "Returns the top 3 search result snippets."
    )

    def run(self, tool_input: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(tool_input.strip(), max_results=3))

            if not results:
                return "No results found for that query."

            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"[{i}] {r.get('title', 'No title')}\n"
                    f"    {r.get('body', 'No snippet')}\n"
                    f"    URL: {r.get('href', '')}"
                )
            return "\n\n".join(formatted)

        except Exception as e:
            return f"Search failed: {e}"


# ---------------------------------------------------------------------------
# 2. Calculator (safe eval — only math expressions)
# ---------------------------------------------------------------------------

class CalculatorTool(BaseTool):
    name        = "calculator"
    description = (
        "Evaluate a mathematical expression. "
        "Input must be a valid Python math expression, e.g. '2 ** 10' or 'sqrt(144)'. "
        "Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e, abs, round."
    )

    _SAFE_NAMESPACE = {
        k: v for k, v in vars(math).items() if not k.startswith("_")
    }
    _SAFE_NAMESPACE.update({
        "abs": abs, "round": round, "int": int, "float": float, "math": math,
    })

    def run(self, tool_input: str) -> str:
        expr = tool_input.strip()
        try:
            result = eval(expr, {"__builtins__": {}}, self._SAFE_NAMESPACE)  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Could not evaluate '{expr}': {e}"


# ---------------------------------------------------------------------------
# 3. Wikipedia — uses MediaWiki search API (more reliable)
# ---------------------------------------------------------------------------

class WikipediaTool(BaseTool):
    name        = "wikipedia"
    description = (
        "Look up encyclopedic / factual information about a topic. "
        "Input should be the topic name, e.g. 'transformer neural network'. "
        "Good for definitions, historical facts, scientific concepts."
    )

    def run(self, tool_input: str) -> str:
        # Use DuckDuckGo to search site:wikipedia.org, then hit the API
        query = tool_input.strip()
        try:
            # Step 1: find the best matching Wikipedia title via search API
            search_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": 1,
                    "namespace": 0,
                    "format": "json",
                },
                headers={"User-Agent": "ReActAgent/1.0 (python-requests)"},
                timeout=10,
            )
            if search_resp.status_code != 200:
                raise Exception(f"Search returned {search_resp.status_code}")

            search_data = search_resp.json()
            titles = search_data[1]
            if not titles:
                return f"No Wikipedia article found for '{query}'."

            title = titles[0]

            # Step 2: fetch the extract for that title
            extract_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action":     "query",
                    "prop":       "extracts",
                    "exintro":    True,
                    "explaintext": True,
                    "exsentences": 6,
                    "titles":     title,
                    "format":     "json",
                },
                headers={"User-Agent": "ReActAgent/1.0 (python-requests)"},
                timeout=10,
            )
            extract_resp.raise_for_status()
            pages   = extract_resp.json()["query"]["pages"]
            page    = next(iter(pages.values()))
            extract = page.get("extract", "").strip()

            if not extract:
                return f"No content found for '{title}'."

            return f"[Wikipedia: {title}]\n{extract[:600]}{'…' if len(extract) > 600 else ''}"

        except Exception as e:
            return f"Wikipedia lookup failed: {e}. Try web_search instead."


# ---------------------------------------------------------------------------
# Register all tools
# ---------------------------------------------------------------------------

register_tool(WebSearchTool())
register_tool(CalculatorTool())
register_tool(WikipediaTool())
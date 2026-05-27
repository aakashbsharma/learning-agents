"""
Tool system.

Every tool:
  1. Subclasses BaseTool
  2. Sets  name        — what the model types in Action:
  3. Sets  description — goes into the system prompt so the model knows when to use it
  4. Implements run(input: str) -> str

The TOOL_REGISTRY dict maps tool names → instances.
The agent looks up tools here when it parses an Action.
"""

from __future__ import annotations
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    name:        str
    description: str  # shown verbatim in the system prompt

    @abstractmethod
    def run(self, tool_input: str) -> str:
        """Execute the tool and return a string observation."""


# ---------------------------------------------------------------------------
# Registry — populated at import time by tools/implementations.py
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, BaseTool] = {}


def register_tool(tool: BaseTool):
    TOOL_REGISTRY[tool.name] = tool


def get_tool(name: str) -> BaseTool | None:
    return TOOL_REGISTRY.get(name.strip())


def list_tools_for_prompt() -> str:
    """
    Render all registered tools as a numbered list for the system prompt.
    Example output:
        1. web_search(query): Searches the web …
        2. calculator(expression): Evaluates …
    """
    lines = []
    for i, tool in enumerate(TOOL_REGISTRY.values(), start=1):
        lines.append(f"{i}. {tool.name}(input): {tool.description}")
    return "\n".join(lines)
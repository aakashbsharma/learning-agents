"""
Entry point for the ReAct agent.

Usage:
  # Single query
  python main.py "What is the population of Mumbai and what is that number squared?"

  # Interactive REPL
  python main.py
"""

import sys
from rich.console import Console
from agent.react_agent import ReActAgent

console = Console()


def main():
    agent = ReActAgent()

    # Single query from CLI arg
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        agent.run(query)
        return

    # Interactive REPL
    console.print("\n[bold magenta]ReAct Agent — Interactive Mode[/bold magenta]")
    console.print("[dim]Type your question and press Enter. Type 'exit' to quit.[/dim]\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        agent.run(query)
        console.print()


if __name__ == "__main__":
    main()
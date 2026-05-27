"""
Pretty-print every agent step with colour coding so you can
see exactly how the model is thinking in real time.

Colour scheme:
  Thought   → cyan   (the model reasoning)
  Action    → yellow (tool the model wants to call)
  Input     → yellow (what it's passing to the tool)
  Obs.      → green  (what the tool returned)
  Answer    → bold green (final answer)
  Error     → red
"""

from rich.console import Console
from rich.panel   import Panel
from rich.text    import Text
from rich.rule    import Rule

console = Console()


def print_step_header(iteration: int):
    console.print(Rule(f"[bold]Iteration {iteration}[/bold]", style="dim"))


def print_thought(thought: str):
    console.print(
        Panel(
            Text(thought, style="cyan"),
            title="[bold cyan]🧠  Thought[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
    )


def print_action(tool_name: str, tool_input: str):
    content = Text()
    content.append("Tool:   ", style="bold yellow")
    content.append(f"{tool_name}\n", style="yellow")
    content.append("Input:  ", style="bold yellow")
    content.append(tool_input, style="yellow")
    console.print(
        Panel(
            content,
            title="[bold yellow]⚡  Action[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
    )


def print_observation(observation: str):
    # Truncate very long observations so the terminal stays readable
    display = observation if len(observation) <= 800 else observation[:800] + "\n… (truncated)"
    console.print(
        Panel(
            Text(display, style="green"),
            title="[bold green]👁  Observation[/bold green]",
            border_style="green",
            padding=(0, 1),
        )
    )


def print_final_answer(answer: str):
    console.print()
    console.print(
        Panel(
            Text(answer, style="bold white"),
            title="[bold green]✅  Final Answer[/bold green]",
            border_style="bold green",
            padding=(1, 2),
        )
    )


def print_error(msg: str):
    console.print(
        Panel(
            Text(msg, style="red"),
            title="[bold red]❌  Error[/bold red]",
            border_style="red",
            padding=(0, 1),
        )
    )


def print_agent_start(query: str, provider: str, model: str):
    console.print()
    console.print(
        Panel(
            f"[bold]{query}[/bold]\n\n[dim]Provider: {provider}  |  Model: {model}[/dim]",
            title="[bold magenta]🤖  ReAct Agent[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        )
    )
    console.print()


def print_max_iterations():
    console.print(
        Panel(
            Text("Reached maximum iterations without a final answer.", style="red"),
            title="[bold red]⚠  Stopped[/bold red]",
            border_style="red",
        )
    )
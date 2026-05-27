"""
ReAct Agent — built from scratch, no LangChain.

The loop:
  1. Build prompt with all previous steps appended
  2. Call LLM → get raw text
  3. Parse: does it contain Action: or Final Answer:?
  4. If Action  → run the tool, append Observation, go to 1
  5. If Answer  → print and stop
  6. If neither → treat entire output as a Thought and re-prompt

The LLM is given a structured system prompt that teaches it the
Thought / Action / Action Input / Observation format explicitly.
"""

import os
import re
from dotenv import load_dotenv

from tools.base         import list_tools_for_prompt, get_tool, TOOL_REGISTRY
from tools.implementations import *   # noqa: F401,F403  — triggers register_tool() calls
from utils.llm          import get_llm_response
from utils.printer      import (
    print_agent_start, print_step_header, print_thought,
    print_action, print_observation, print_final_answer,
    print_error, print_max_iterations, console,
)

load_dotenv()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful AI assistant that solves problems step by step using tools.

You have access to the following tools:
{tools}

RESPONSE FORMAT — you MUST follow this format exactly, every single turn:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <the exact input to pass to the tool>

After you receive an Observation, continue with another Thought/Action/Action Input block,
OR if you have enough information to answer, finish with:

Thought: <your final reasoning>
Final Answer: <your complete answer to the user's question>

RULES:
- Always start with "Thought:"
- Never skip the Thought step
- Action must be EXACTLY one of: {tool_names}
- Never make up tool results — wait for the Observation
- If a tool fails, try a different approach
- Keep Action Input on a SINGLE line
"""


def build_system_prompt() -> str:
    tools     = list_tools_for_prompt()
    tool_names = ", ".join(TOOL_REGISTRY.keys())
    return SYSTEM_PROMPT_TEMPLATE.format(tools=tools, tool_names=tool_names)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_llm_output(text: str) -> dict:
    """
    Parse the LLM's raw text into a structured dict.

    Returns one of:
      {"type": "action",       "thought": ..., "action": ..., "action_input": ...}
      {"type": "final_answer", "thought": ..., "answer": ...}
      {"type": "thought_only", "thought": ...}   ← re-prompt on next iteration
    """
    # Extract Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else text.strip()

    # Check for Final Answer
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        return {
            "type":    "final_answer",
            "thought": thought,
            "answer":  final_match.group(1).strip(),
        }

    # Check for Action
    action_match      = re.search(r"Action:\s*(.+)",       text)
    action_input_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)

    if action_match and action_input_match:
        return {
            "type":         "action",
            "thought":      thought,
            "action":       action_match.group(1).strip(),
            "action_input": action_input_match.group(1).strip().split("\n")[0],  # first line only
        }

    # Fallback — the model produced only a Thought, nudge it
    return {"type": "thought_only", "thought": thought}


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class ReActAgent:
    def __init__(self):
        self.system_prompt  = build_system_prompt()
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", 10))
        self.provider       = os.getenv("LLM_PROVIDER", "groq")
        model_env           = "GROQ_MODEL" if self.provider == "groq" else "OLLAMA_MODEL"
        self.model          = os.getenv(model_env, "llama-3.3-70b-versatile")

    def run(self, user_query: str) -> str:
        print_agent_start(user_query, self.provider, self.model)

        # Conversation history — we keep the full trace so the model
        # always has context of every prior Thought/Action/Observation
        messages = [
            {"role": "system",  "content": self.system_prompt},
            {"role": "user",    "content": user_query},
        ]

        for iteration in range(1, self.max_iterations + 1):
            print_step_header(iteration)

            # ── LLM call ──────────────────────────────────────────────────
            raw_output = get_llm_response(
                messages,
                stop_sequences=["Observation:"],  # stop before hallucinating tool results
            )

            # ── Parse ─────────────────────────────────────────────────────
            parsed = parse_llm_output(raw_output)

            # Always show the thought
            if parsed["thought"]:
                print_thought(parsed["thought"])

            # ── Final Answer ───────────────────────────────────────────────
            if parsed["type"] == "final_answer":
                print_final_answer(parsed["answer"])
                return parsed["answer"]

            # ── Action ────────────────────────────────────────────────────
            elif parsed["type"] == "action":
                tool_name  = parsed["action"]
                tool_input = parsed["action_input"]
                print_action(tool_name, tool_input)

                tool = get_tool(tool_name)
                if tool is None:
                    observation = (
                        f"Error: tool '{tool_name}' does not exist. "
                        f"Available tools: {', '.join(TOOL_REGISTRY.keys())}"
                    )
                    print_error(observation)
                else:
                    observation = tool.run(tool_input)
                    print_observation(observation)

                # Append the full Thought+Action block to messages,
                # then add the Observation as the next user turn.
                # This keeps the conversation structured correctly.
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({
                    "role":    "user",
                    "content": f"Observation: {observation}",
                })

            # ── Thought only — re-prompt ───────────────────────────────────
            else:
                console.print("[dim]  (model produced only a Thought — nudging it)[/dim]")
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({
                    "role":    "user",
                    "content": "Please continue. Remember to use Action: and Action Input: or give a Final Answer:",
                })

        # Fell through all iterations
        print_max_iterations()
        return "Agent stopped: maximum iterations reached without a final answer."
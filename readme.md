# ReAct Agent — From Scratch

A **ReAct (Reasoning + Acting)** agent built with pure Python — no LangChain, no LangGraph.
You can see every Thought, Action, and Observation the model produces in real time.

## Folder Structure

```
react_agent/
├── agent/
│   ├── __init__.py
│   └── react_agent.py      ← The core loop: parse → think → act → observe
├── tools/
│   ├── __init__.py
│   ├── base.py             ← BaseTool class + TOOL_REGISTRY
│   └── implementations.py  ← WebSearch, Calculator, Wikipedia
├── utils/
│   ├── __init__.py
│   ├── llm.py              ← Groq / Ollama abstraction
│   └── printer.py          ← Rich colour-coded terminal output
├── main.py                 ← Entry point (CLI or interactive REPL)
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — add your GROQ_API_KEY (free at console.groq.com)

# 3. Run
python main.py "What is the square root of the population of Tokyo?"
```

## Switching to Ollama

```env
# .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:14b     # or llama3.2, mistral, etc.
```

```bash
ollama pull qwen2.5:14b
python main.py "Your question"
```

## Adding a New Tool

1. Open `tools/implementations.py`
2. Subclass `BaseTool`, set `name` and `description`, implement `run()`
3. Call `register_tool(YourTool())` at the bottom
4. Done — the agent picks it up automatically

```python
class MyTool(BaseTool):
    name        = "my_tool"
    description = "Does something useful. Input should be ..."

    def run(self, tool_input: str) -> str:
        return "result"

register_tool(MyTool())
```

## How ReAct Works

```
User query
    ↓
[Thought]  model reasons about what to do
    ↓
[Action]   model names a tool
    ↓
[Action Input]  model provides the input
    ↓
[Observation]  tool runs, result appended to context
    ↓
[Thought]  model reasons about the observation
    ↓
  ... loop ...
    ↓
[Final Answer]  model answers from accumulated observations
```

The key insight: the model never "knows" the answer — it accumulates
evidence through tool calls, exactly like a human would research something.
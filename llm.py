"""
LLM abstraction layer.
Supports Groq and Ollama with the same interface.
Switch providers by changing LLM_PROVIDER in .env
"""

import os
import json
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def get_llm_response(messages: list[dict], stop_sequences: list[str] = None) -> str:
    """
    Send messages to the configured LLM and return the raw text response.

    Args:
        messages:        List of {"role": "...", "content": "..."} dicts.
        stop_sequences:  Optional list of strings where the model should stop.

    Returns:
        The model's text output as a plain string.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        return _groq_call(messages, stop_sequences)
    elif provider == "ollama":
        return _ollama_call(messages, stop_sequences)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'. Use 'groq' or 'ollama'.")


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

def _groq_call(messages: list[dict], stop_sequences: list[str] = None) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=0.0,   # deterministic — we want consistent ReAct parsing
        max_tokens=1024,
    )
    if stop_sequences:
        kwargs["stop"] = stop_sequences

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _ollama_call(messages: list[dict], stop_sequences: list[str] = None) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model    = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": 0.0},
    }
    if stop_sequences:
        payload["options"]["stop"] = stop_sequences

    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()
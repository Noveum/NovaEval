"""
Example usage of OllamaModel with NovaEval against a remote endpoint.

Environment variables:
- OLLAMA_BASE_URL: The Ollama endpoint URL, e.g. https://api.your-ollama-host.com
- OLLAMA_API_KEY:   Optional API key; sent as Authorization: Bearer <key>
- OLLAMA_MODEL:     The model name to use (default: llama3)
- OLLAMA_GPU_COST_PER_SEC: Optional cost per GPU-second for cost estimation

Run:
  OLLAMA_BASE_URL=https://host:11434 OLLAMA_API_KEY=sk_... \
  python examples/ollama_evaluation.py
"""
from __future__ import annotations

import os
from typing import Dict
import time

from novaeval.models.ollama import OllamaModel


def main() -> None:
    base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST", "http://34.121.64.12:8004")
    api_key = os.getenv("OLLAMA_API_KEY", "bruh")
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    model = OllamaModel(
        model_name=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
        base_url=base_url,
        headers=headers,
        gpu_cost_per_sec=float(os.getenv("OLLAMA_GPU_COST_PER_SEC", "0.0")),
    )

    ok = model.validate_connection()
    print(f"Connection valid: {ok}")

    prompt = "In one sentence, explain why the sky appears blue."
    start_time = time.perf_counter()
    text = model.generate(prompt=prompt, max_tokens=1280, temperature=0.1, think="high")
    elapsed = time.perf_counter() - start_time
    print(f"Generation time: {elapsed:.3f}s")
    print("Response:\n", text)

    # Use the new generate_with_thought method
    print("\nUsing generate_with_thought():")
    start_time2 = time.perf_counter()
    answer, thought = model.generate_with_thought(
        prompt=prompt,
        max_tokens=1280,
        temperature=0.1,
        think="high",
    )
    elapsed2 = time.perf_counter() - start_time2
    print(f"Generation with thought time: {elapsed2:.3f}s")
    print("Thought:\n", thought)
    print("Response after thinking:\n", answer)

    print("\nModel info:")
    print(model.get_info())


if __name__ == "__main__":
    main() 
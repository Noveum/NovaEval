"""
Ollama model implementation for NovaEval.

This module provides an interface to local Ollama models via the official
`ollama` Python SDK.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

try:
    from ollama import Client  # type: ignore
except ImportError:  # pragma: no cover - import failure surfaced at runtime
    Client = None  # type: ignore
from novaeval.models.base import BaseModel, trace_llm


class OllamaModel(BaseModel):
    """
    Ollama model implementation.

    Parameters
    - model_name: name of the local Ollama model (e.g., "llama3")
    - base_url: Ollama host URL (e.g., "http://localhost:11434").
                If not provided, falls back to OLLAMA_HOST env var,
                else defaults to "http://localhost:11434".
    - headers: Optional HTTP headers to pass to the underlying httpx client
    - gpu_cost_per_sec: Optional per-second GPU cost to estimate request cost.
                        If provided, cost is estimated as duration_seconds * gpu_cost_per_sec.
    - pull_on_init: If True, attempt to pull the model during initialization.
    - **kwargs: Additional model parameters, persisted on the instance.
    """

    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        gpu_cost_per_sec: float | None = None,
        pull_on_init: bool = True,
        **kwargs: Any,
    ):
        effective_host = (
            base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        )

        super().__init__(
            name=f"ollama_{model_name}",
            model_name=model_name,
            base_url=effective_host,
            **kwargs,
        )

        if Client is None:  # pragma: no cover
            raise ImportError(
                "The 'ollama' package is required. Please install with `pip install ollama`."
            )

        # Store cost parameter for cost estimation
        self.gpu_cost_per_sec = gpu_cost_per_sec

        # Initialize Ollama client
        # Any extra keyword arguments not used here are stored in self.kwargs and can be
        # leveraged by callers if needed.
        self.headers = headers or {}
        self.client = Client(host=effective_host, headers=self.headers)

        # Ensure model is available locally/remote registry: pull on init if requested
        if pull_on_init:
            try:
                pull_result = self.client.pull(model=self.model_name, stream=True)
                # If streaming generator, iterate to completion; if not iterable, just access
                try:
                    for _ in pull_result:  # type: ignore[assignment]
                        pass
                except TypeError:
                    _ = pull_result  # consume non-iterable result
            except Exception as e:  # do not block initialization
                self._handle_error(
                    e, f"Failed to pull model '{self.model_name}' on init"
                )

    # -------------------------- Helper methods ---------------------------
    @staticmethod
    def _extract_content_from_response(response: Any) -> str:
        # Support mapping-style and attribute-style access
        try:
            # Preferred: attribute access if available
            msg = getattr(response, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    return content
        except Exception:
            pass

        try:
            # Fallback: mapping access
            return response.get("message", {}).get(
                "content", ""
            )  # type: ignore[union-attr]
        except Exception:
            return ""

    @staticmethod
    def _extract_metric(response_or_chunk: Any, key: str) -> int | None:
        # Try attribute access first, then mapping
        try:
            val = getattr(response_or_chunk, key, None)
            if isinstance(val, int):
                return val
        except Exception:
            pass
        try:
            val = response_or_chunk.get(key)  # type: ignore[attr-defined]
            if isinstance(val, int):
                return val
        except Exception:
            pass
        return None

    @staticmethod
    def _build_options(
        base_options: dict[str, Any],
        *,
        max_tokens: int | None,
        temperature: float | None,
        stop: str | list[str] | None,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        # Start with provided options, then map NovaEval params
        options: dict[str, Any] = dict(base_options)
        if max_tokens is not None and "num_predict" not in options:
            options["num_predict"] = max_tokens
        if temperature is not None and "temperature" not in options:
            options["temperature"] = temperature
        if stop is not None:
            # Ollama expects stop sequences in options.stop (list)
            options["stop"] = stop if isinstance(stop, list) else [stop]

        # Merge supported Ollama options from kwargs if present
        supported = {
            # sampling & prediction
            "num_predict",
            "temperature",
            "top_p",
            "top_k",
            "tfs_z",
            "typical_p",
            "min_p",
            "repeat_penalty",
            "repeat_last_n",
            "presence_penalty",
            "frequency_penalty",
            # context / performance
            "num_ctx",
            "num_thread",
            "num_gpu",
            "num_batch",
            "main_gpu",
            "numa",
            "low_vram",
            "vocab_only",
            "use_mmap",
            "use_mlock",
            # mirostat
            "mirostat",
            "mirostat_eta",
            "mirostat_tau",
            # other
            "seed",
            "num_keep",
            "penalize_newline",
            "stop",
        }
        for key in list(extra_kwargs.keys()):
            if key in supported and key not in options:
                options[key] = extra_kwargs.pop(key)
        return options

    @staticmethod
    def _build_messages_from_prompt(prompt: str | None) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt or ""}]

    # NEW: extract "thinking" field from responses/chunks
    @staticmethod
    def _extract_thinking_from_response(response: Any) -> str:
        try:
            thought = getattr(response, "thinking", None)
            if isinstance(thought, str) and thought:
                return thought
        except Exception:
            pass
        # Nested under message as attribute
        try:
            msg = getattr(response, "message", None)
            if msg is not None:
                for key in ("thinking", "reasoning", "thought", "thoughts"):
                    val = getattr(msg, key, None)
                    if isinstance(val, str) and val:
                        return val
        except Exception:
            pass
        # Mapping top-level
        try:
            for key in ("thinking", "reasoning", "thought", "thoughts"):
                val = response.get(key)  # type: ignore[attr-defined]
                if isinstance(val, str) and val:
                    return val
        except Exception:
            pass
        # Mapping nested under message
        try:
            msg_map = response.get("message", {})  # type: ignore[attr-defined]
            if isinstance(msg_map, dict):
                for key in ("thinking", "reasoning", "thought", "thoughts"):
                    val = msg_map.get(key)
                    if isinstance(val, str) and val:
                        return val
        except Exception:
            pass
        return ""

    # -------------------------- Core API ---------------------------
    @trace_llm
    def generate(
        self,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: str | list[str] | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        if prompt is None and messages is None:
            raise ValueError("Either `prompt` or `messages` must be provided.")

        # Prepare messages for chat API
        if messages is None:
            messages = self._build_messages_from_prompt(prompt)

        # Extract top-level chat params
        format_param = kwargs.pop("format", None)
        keep_alive = kwargs.pop("keep_alive", None)

        # Thinking / reasoning param (Ollama supports `think` levels or boolean)
        if "think" in kwargs:
            think_value = kwargs.pop("think")
        else:
            think_value = kwargs.pop("reasoning", None)

        # Merge options
        provided_options: dict[str, Any] = kwargs.pop("options", {}) or {}
        options = self._build_options(
            provided_options,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            extra_kwargs=kwargs,
        )

        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "options": options if options else None,
            "stream": False,
        }
        if format_param is not None:
            params["format"] = format_param
        if keep_alive is not None:
            params["keep_alive"] = keep_alive
        if think_value is not None:
            params["think"] = think_value

        # Remove Nones to avoid API confusion
        params = {k: v for k, v in params.items() if v is not None}

        try:
            # Execute request
            start_time = time.time()
            response = self.client.chat(**params)
            generated_text = self._extract_content_from_response(response)

            total_duration_ns = self._extract_metric(response, "total_duration")
            prompt_eval_count = self._extract_metric(response, "prompt_eval_count")
            eval_count = self._extract_metric(response, "eval_count")

            if total_duration_ns is not None and total_duration_ns >= 0:
                elapsed_seconds = total_duration_ns / 1e9
            else:
                elapsed_seconds = max(0.0, time.time() - start_time)

            # Token usage fallback
            if prompt_eval_count is None:
                prompt_for_tracking = json.dumps(messages)
                prompt_eval_count = self.count_tokens(prompt_for_tracking)
            if eval_count is None:
                eval_count = self.count_tokens(generated_text)

            tokens_used = int(prompt_eval_count) + int(eval_count)
            cost = self.estimate_cost(
                json.dumps(messages),
                generated_text,
                elapsed_seconds=elapsed_seconds,
                total_duration_ns=total_duration_ns,
            )

            self._track_request(
                prompt=json.dumps(messages),
                response=generated_text,
                tokens_used=tokens_used,
                cost=cost,
            )
            return generated_text

        except Exception as e:
            self._handle_error(e, "Failed to generate text via Ollama chat API.")
            raise

    def generate_with_thought(
        self,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: str | list[str] | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Like `generate`, but also returns the model's internal thinking text.

        Returns a tuple of (generated_text, thinking_text).
        """
        if prompt is None and messages is None:
            raise ValueError("Either `prompt` or `messages` must be provided.")

        if messages is None:
            messages = self._build_messages_from_prompt(prompt)

        format_param = kwargs.pop("format", None)
        keep_alive = kwargs.pop("keep_alive", None)

        if "think" in kwargs:
            think_value = kwargs.pop("think")
        else:
            think_value = kwargs.pop("reasoning", None)

        provided_options: dict[str, Any] = kwargs.pop("options", {}) or {}
        options = self._build_options(
            provided_options,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            extra_kwargs=kwargs,
        )

        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "options": options if options else None,
            "stream": False,
        }
        if format_param is not None:
            params["format"] = format_param
        if keep_alive is not None:
            params["keep_alive"] = keep_alive
        if think_value is not None:
            params["think"] = think_value

        params = {k: v for k, v in params.items() if v is not None}

        try:
            start_time = time.time()
            response = self.client.chat(**params)
            generated_text = self._extract_content_from_response(response)
            thinking_text = self._extract_thinking_from_response(response)

            total_duration_ns = self._extract_metric(response, "total_duration")
            prompt_eval_count = self._extract_metric(response, "prompt_eval_count")
            eval_count = self._extract_metric(response, "eval_count")

            if total_duration_ns is not None and total_duration_ns >= 0:
                elapsed_seconds = total_duration_ns / 1e9
            else:
                elapsed_seconds = max(0.0, time.time() - start_time)

            if prompt_eval_count is None:
                prompt_for_tracking = json.dumps(messages)
                prompt_eval_count = self.count_tokens(prompt_for_tracking)
            if eval_count is None:
                eval_count = self.count_tokens(generated_text)

            tokens_used = int(prompt_eval_count) + int(eval_count)
            cost = self.estimate_cost(
                json.dumps(messages),
                generated_text,
                elapsed_seconds=elapsed_seconds,
                total_duration_ns=total_duration_ns,
            )

            self._track_request(
                prompt=json.dumps(messages),
                response=generated_text,
                tokens_used=tokens_used,
                cost=cost,
            )
            return generated_text, thinking_text

        except Exception as e:
            self._handle_error(
                e, "Failed to generate text with thought via Ollama chat API."
            )
            raise

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: str | list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        # Ollama Python library does not support native batch chat; process sequentially
        results: list[str] = []
        for prompt in prompts:
            try:
                text = self.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    **kwargs,
                )
                results.append(text)
            except Exception as e:
                self._handle_error(e, f"Batch failure for: {prompt[:100]}...")
                results.append("")
        return results

    def get_provider(self) -> str:
        return "ollama"

    def estimate_cost(
        self,
        prompt: str,
        response: str = "",
        *,
        elapsed_seconds: float | None = None,
        total_duration_ns: int | None = None,
    ) -> float:
        """
        Estimate cost based on GPU wall-clock time if `gpu_cost_per_sec` is provided.

        Prefer `total_duration_ns` (from Ollama response), else use `elapsed_seconds`.
        If neither is available or `gpu_cost_per_sec` is unset, returns 0.0.
        """
        if self.gpu_cost_per_sec is None:
            return 0.0

        seconds: float
        if total_duration_ns is not None and total_duration_ns >= 0:
            seconds = total_duration_ns / 1e9
        elif elapsed_seconds is not None:
            seconds = max(0.0, float(elapsed_seconds))
        else:
            seconds = 0.0

        return float(self.gpu_cost_per_sec) * seconds

    def validate_connection(self) -> bool:
        try:
            resp = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                stream=False,
                options={"num_predict": 1},
            )
            # If we get any response back, assume OK
            txt = self._extract_content_from_response(resp)
            return bool(txt is not None)
        except Exception as e:
            self._handle_error(e, "Ollama connection validation failed")
            return False

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "host": self.base_url,
                "supports_batch": False,
                "pricing": (0.0, 0.0),  # local inference; no per-token pricing
                "gpu_cost_per_sec": self.gpu_cost_per_sec,
            }
        )
        return info

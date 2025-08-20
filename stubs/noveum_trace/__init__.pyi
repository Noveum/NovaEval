from typing import Any, Callable, TypeVar

from typing_extensions import ParamSpec, overload

P = ParamSpec("P")
R = TypeVar("R")

@overload
def trace_llm(func: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def trace_llm(
    func: None = ...,
    *,
    name: str | None = ...,
    provider: str | None = ...,
    capture_prompts: bool = ...,
    capture_completions: bool = ...,
    capture_tokens: bool = ...,
    estimate_costs: bool = ...,
    redact_pii: bool = ...,
    metadata: dict[str, Any] | None = ...,
    tags: dict[str, str] | None = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def init(
    *,
    api_key: str | None = ...,
    project: str | None = ...,
    environment: str | None = ...,
    **kwargs: Any,
) -> None: ...

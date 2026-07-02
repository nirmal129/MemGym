"""LiteLLM bridge for the vendored All-Mem engine.

All-Mem's engine (``core.py``) only ever calls two methods on its
``llm_controller``::

    controller.complete(prompt, response_format=None, temperature=0.0) -> str
    controller.complete_json(prompt, response_format=None, temperature=0.0,
                             max_retries=5) -> dict

This module provides a drop-in controller backed by **LiteLLM**, so All-Mem
runs against any LiteLLM target the rest of MemGym already uses:

- Local SGLang / vLLM via the OpenAI-compatible API:
  ``model="openai/<id>"`` + ``OPENAI_API_BASE`` / ``OPENAI_API_KEY`` env.
- Hosted OpenAI: ``model="gpt-4o-mini"`` + ``OPENAI_API_KEY``.
- Gemini through its OpenAI-compatible endpoint: ``model="openai/gemini-..."``
  + ``OPENAI_API_BASE`` pointing at the Gemini OpenAI shim (see quick_start.md).

The endpoint is taken from the explicit ``api_base`` / ``api_key`` args when
given, otherwise from the environment (LiteLLM reads ``OPENAI_API_BASE`` /
``OPENAI_API_KEY`` automatically for ``openai/*`` models). This mirrors how the
A-MEM adapter (``external/amem/llm_controller.py``) is wired.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

# Reasoning models (gpt-5*, o1*, o3*) reject temperature != 1, so we omit it.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3")


def parse_json_object(raw_text: str) -> Dict[str, Any]:
    """Parse a JSON object from a model response.

    Identical contract to All-Mem's ``llm.parse_json_object``: strip ```json
    fences, then fall back to slicing the outer ``{...}`` if the whole string
    is not valid JSON. Raises on unrecoverable output (the caller retries).
    """
    text = (raw_text or "").replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
        raise ValueError("JSON response is not an object")
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("JSON response is not an object")
        return value


class LiteLLMController:
    """Drop-in replacement for All-Mem's ``LLMController`` over LiteLLM."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        request_timeout: float = 120.0,
        max_tokens: Optional[int] = 4096,
    ) -> None:
        # Imported lazily so importing the allmem package never hard-requires
        # litellm (the registry must still load if optional deps are missing).
        from litellm import completion

        self._completion = completion
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.request_timeout = request_timeout
        # Cap generation per call. All-Mem's internal outputs (semantic index,
        # edge/diagnosis/consolidation JSON) are small, so this never truncates
        # legitimate output — but it bounds the occasional greedy-decode (temp=0)
        # repetition loop that, uncapped, runs away to tens of thousands of
        # tokens (~300s/call at ~130 tok/s) and stalls the worker. Set to None
        # to disable. drop_params=True means backends without max_tokens ignore it.
        self.max_tokens = max_tokens
        self._is_reasoning = any(model.startswith(p) for p in _REASONING_PREFIXES)
        self.default_temperature = None if self._is_reasoning else temperature

    def _build_kwargs(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]],
        temperature: Optional[float],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            "timeout": self.request_timeout,
            # Let LiteLLM silently drop params a given provider does not support
            # (e.g. response_format on backends without structured output)
            # instead of raising mid-trajectory.
            "drop_params": True,
        }
        temp = self.default_temperature if temperature is None else temperature
        if self._is_reasoning:
            temp = None
        if temp is not None:
            kwargs["temperature"] = temp
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if response_format:
            kwargs["response_format"] = response_format
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return kwargs

    def complete(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Single completion. Returns ``"{}"`` on error so callers never crash."""
        try:
            kwargs = self._build_kwargs(prompt, response_format, temperature)
            response = self._completion(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - eval must not die on one call
            import logging

            logging.getLogger("memgym.allmem").warning("LiteLLM completion error: %s", exc)
            return "{}"

    def complete_json(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_retries: int = 5,
    ) -> Dict[str, Any]:
        """Completion + JSON parse with exponential backoff.

        Returns ``{}`` after exhausting retries (never raises) so a single bad
        response degrades one node's metadata/diagnosis rather than aborting the
        whole evaluation — matching All-Mem's defensive error handling.
        """
        fmt = response_format or {"type": "json_object"}
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                raw = self.complete(prompt, response_format=fmt, temperature=temperature)
                return parse_json_object(raw)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(min(30.0, (2 ** attempt) + 0.1))
        import logging

        logging.getLogger("memgym.allmem").warning(
            "All-Mem complete_json failed after %d retries: %s", max_retries, last_error
        )
        return {}

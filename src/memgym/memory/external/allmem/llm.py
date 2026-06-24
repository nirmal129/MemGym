"""LLM backend adapters used by All-Mem.

This module intentionally contains only generic provider wrappers. The old
paper baseline memory implementation is not part of the refactored package.

Vendored verbatim from ``All-Mem/all_mem/llm.py`` (arXiv:2603.19595, MIT).
``core.py`` imports ``LLMController`` from here purely as a type annotation;
MemGym drives the engine with ``llm_bridge.LiteLLMController`` instead, so this
module's native openai/sglang/ollama backends are not used in MemGym runs.
Kept so the vendored ``core.py`` stays byte-for-byte identical to upstream.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class LLMConfig:
    backend: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    sglang_host: str = "http://localhost"
    sglang_port: int = 30000
    request_timeout: float = 120.0


def resolve_api_key(explicit_api_key: Optional[str] = None) -> str:
    """Resolve API key from explicit value or environment variables."""
    if explicit_api_key:
        return explicit_api_key
    key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("ARK_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "Missing API key. Set OPENAI_API_KEY, OPENROUTER_API_KEY, "
            "ARK_API_KEY, or pass api_key explicitly."
        )
    return key


def resolve_api_base(explicit_api_base: Optional[str] = None) -> Optional[str]:
    return (
        explicit_api_base
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENROUTER_BASE_URL")
    )


class OpenAIChatBackend:
    def __init__(self, config: LLMConfig):
        try:
            from openai import OpenAI
        except ImportError:
            OpenAI = None

        self.model = config.model
        self.timeout = config.request_timeout
        self.api_key = resolve_api_key(config.api_key)
        self.base_url = (resolve_api_base(config.api_base) or "https://api.openai.com/v1").rstrip("/")
        self.client = (
            OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=config.request_timeout,
            )
            if OpenAI
            else None
        )

    def complete(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        if self.client is not None:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return content or ""

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=kwargs,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"].get("content") or ""


class SGLangBackend:
    def __init__(self, config: LLMConfig):
        self.model = config.model
        self.timeout = config.request_timeout
        self.base_url = f"{config.sglang_host}:{config.sglang_port}".rstrip("/")

    def complete(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"].get("content") or ""


class OllamaBackend:
    def __init__(self, config: LLMConfig):
        try:
            from litellm import completion
        except ImportError as exc:
            raise ImportError("Install litellm to use the ollama backend.") from exc

        model = config.model
        self.model = model if model.startswith("ollama/") else f"ollama/{model}"
        self.completion = completion

    def complete(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "api_base": "http://localhost:11434",
            "api_key": "EMPTY",
        }
        if response_format:
            kwargs["response_format"] = response_format
        response = self.completion(**kwargs)
        return response.choices[0].message.content or ""


class LLMController:
    """Small facade over supported chat-completion backends."""

    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        request_timeout: float = 120.0,
    ):
        self.config = LLMConfig(
            backend=backend,
            model=model,
            api_key=api_key,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
            request_timeout=request_timeout,
        )

        backend_name = backend.lower()
        if backend_name == "openai":
            self.backend = OpenAIChatBackend(self.config)
        elif backend_name == "sglang":
            self.backend = SGLangBackend(self.config)
        elif backend_name == "ollama":
            self.backend = OllamaBackend(self.config)
        else:
            raise ValueError("backend must be one of: openai, sglang, ollama")

    def complete(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> str:
        return self.backend.complete(prompt, response_format, temperature)

    def complete_json(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        max_retries: int = 5,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        fmt = response_format or {"type": "json_object"}
        for attempt in range(max_retries):
            try:
                raw = self.complete(prompt, response_format=fmt, temperature=temperature)
                return parse_json_object(raw)
            except Exception as exc:
                last_error = exc
                time.sleep(min(30.0, (2**attempt) + 0.1))
        raise RuntimeError(f"Failed to parse JSON response after {max_retries} retries") from last_error


def parse_json_object(raw_text: str) -> Dict[str, Any]:
    """Parse a JSON object from a model response."""
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

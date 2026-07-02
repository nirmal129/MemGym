"""
LLM Controller: LiteLLM-based controller for memory metadata generation.

Uses LiteLLM for universal LLM access (OpenAI, Anthropic, local models via SGLang/vLLM/Ollama).
"""

import json
import os
from typing import Any, Dict, Optional

from litellm import completion


def _env_int(name: str) -> Optional[int]:
    """Read an int from the environment, returning None if unset/blank/invalid."""
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


class LLMController:
    """
    LiteLLM-based controller for memory operations.

    Supports all LiteLLM backends:
    - OpenAI: gpt-4o-mini, gpt-4o, etc.
    - Anthropic: claude-3-5-sonnet, etc.
    - Local models via SGLang/vLLM: openai/model-name with api_base
    - Ollama: ollama/model-name

    Config options:
        model: LLM model name (LiteLLM format)
        api_base: API base URL (for local models)
        api_key: API key (uses env var if not provided)
        temperature: Generation temperature (default 0.7)
        timeout: Per-request timeout in seconds (defaults to
            LITELLM_REQUEST_TIMEOUT env var if not provided)
        max_tokens: Max completion tokens (defaults to LITELLM_MAX_TOKENS
            env var if not provided)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize LLM controller.

        Args:
            model: LLM model name in LiteLLM format
            api_base: API base URL for local models
            api_key: API key (optional, uses env var if not set)
            temperature: Generation temperature
            timeout: Per-request timeout in seconds. Falls back to the
                LITELLM_REQUEST_TIMEOUT env var, then litellm's default.
            max_tokens: Max completion tokens. Falls back to the
                LITELLM_MAX_TOKENS env var, then the server default.
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        # litellm ignores LITELLM_REQUEST_TIMEOUT/LITELLM_MAX_TOKENS env vars,
        # so honor them here by passing the values as explicit call args.
        self.timeout = timeout if timeout is not None else _env_int("LITELLM_REQUEST_TIMEOUT")
        self.max_tokens = max_tokens if max_tokens is not None else _env_int("LITELLM_MAX_TOKENS")
        # Reasoning models (gpt-5*, o1*, o3*) only support temperature=1
        _reasoning_prefixes = ("gpt-5", "o1", "o3")
        if any(model.startswith(p) for p in _reasoning_prefixes):
            self.temperature = None  # Don't send temperature param
        else:
            self.temperature = temperature

    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        """Generate empty value based on JSON schema type."""
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number" or schema_type == "integer":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        """Generate empty response matching JSON schema."""
        if "json_schema" not in response_format:
            return {}

        schema = response_format["json_schema"]["schema"]
        result = {}

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(
                    prop_schema["type"],
                    prop_schema.get("items")
                )

        return result

    def get_completion(
        self,
        prompt: str,
        response_format: Optional[Dict] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Get completion from LLM.

        Args:
            prompt: User prompt
            response_format: JSON schema for structured output
            temperature: Override default temperature

        Returns:
            LLM response text
        """
        try:
            # Build completion arguments
            completion_args = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
            }
            temp = temperature or self.temperature
            if temp is not None:
                completion_args["temperature"] = temp

            # Add response format if provided
            if response_format:
                completion_args["response_format"] = response_format

            # Add API base and key if provided
            if self.api_base:
                completion_args["api_base"] = self.api_base
            if self.api_key:
                completion_args["api_key"] = self.api_key

            # Honor timeout / max_tokens when configured
            if self.timeout is not None:
                completion_args["timeout"] = self.timeout
            if self.max_tokens is not None:
                completion_args["max_tokens"] = self.max_tokens

            response = completion(**completion_args)
            return response.choices[0].message.content

        except Exception as e:
            print(f"LiteLLM completion error: {e}")
            # Return empty response matching schema on error
            if response_format:
                empty_response = self._generate_empty_response(response_format)
                return json.dumps(empty_response)
            return "{}"

    def get_chat_completion(
        self,
        messages: list,
        response_format: Optional[Dict] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Get chat completion from LLM with custom messages.

        Args:
            messages: List of message dicts with role and content
            response_format: JSON schema for structured output
            temperature: Override default temperature

        Returns:
            LLM response text
        """
        try:
            completion_args = {
                "model": self.model,
                "messages": messages,
            }
            temp = temperature or self.temperature
            if temp is not None:
                completion_args["temperature"] = temp

            if response_format:
                completion_args["response_format"] = response_format
            if self.api_base:
                completion_args["api_base"] = self.api_base
            if self.api_key:
                completion_args["api_key"] = self.api_key

            # Honor timeout / max_tokens when configured
            if self.timeout is not None:
                completion_args["timeout"] = self.timeout
            if self.max_tokens is not None:
                completion_args["max_tokens"] = self.max_tokens

            response = completion(**completion_args)
            return response.choices[0].message.content

        except Exception as e:
            print(f"LiteLLM chat completion error: {e}")
            if response_format:
                empty_response = self._generate_empty_response(response_format)
                return json.dumps(empty_response)
            return "{}"

"""All-Mem: agentic lifelong memory via dynamic topology evolution.

Vendored engine (``core.py``) + MemGym bridges. Heavy dependencies
(``networkx``, ``sentence-transformers``, ``scikit-learn``, ``torch``,
``litellm``) are optional: importing this package never hard-fails. Track
adapters guard the import and soft-disable when a dependency is missing, the
same way the A-MEM adapters do.

Public surface:
    MemGymAllMemSystem  — All-Mem engine with one-shot reranker fallback (use this)
    AllMemSystem        — upstream engine (verbatim)
    AllMemGraph, AllMemNode, OptimizationBuffer
    LiteLLMController   — drop-in LLM controller over LiteLLM
    parse_json_object
"""

from __future__ import annotations

from .llm_bridge import LiteLLMController, parse_json_object

try:
    from .core import (
        AllMemGraph,
        AllMemNode,
        AllMemSystem,
        OptimizationBuffer,
    )
    from .engine import MemGymAllMemSystem

    _ALLMEM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional heavy deps
    AllMemGraph = None
    AllMemNode = None
    AllMemSystem = None
    OptimizationBuffer = None
    MemGymAllMemSystem = None
    _ALLMEM_AVAILABLE = False

__all__ = [
    "AllMemGraph",
    "AllMemNode",
    "AllMemSystem",
    "MemGymAllMemSystem",
    "OptimizationBuffer",
    "LiteLLMController",
    "parse_json_object",
    "_ALLMEM_AVAILABLE",
]

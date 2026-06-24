"""All-Mem adapter for the coding_synthetic (CodeQA) evaluation pipeline.

Wraps the vendored All-Mem engine (``memgym.memory.external.allmem``) behind the
``CodingMemoryMethod`` protocol (``ingest`` / ``retrieve`` / ``reset``).

Mapping (see ``custom_memory.md`` §6d), faithful to ``All-Mem/all_mem``:

- ``ingest(doc)`` -> ``wake_process`` (online write: LLM index + edges + queued
  diagnosis); a ``sleep_process`` runs every ``sleep_interval`` documents
  (offline Agentic Topology Consolidation: Split / Merge / Update).
- ``retrieve(question)`` -> one final ``sleep_process`` (flush pending
  consolidation) then ``get_context_for_query`` (ANN anchors -> typed-link
  expansion -> rerank), returning the assembled context string.

The memory LLM is driven through LiteLLM, so it follows the same endpoint as the
answerer: pass ``llm_model`` (the eval ``--model``) and rely on
``OPENAI_API_BASE`` / ``OPENAI_API_KEY`` env for SGLang / OpenAI / Gemini.
"""
from __future__ import annotations

from typing import Any, Optional


class AllMemMethod:
    """All-Mem (dynamic topology graph) memory for coding QA evaluation."""

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_device: Optional[str] = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_device: str = "cpu",
        use_reranker: bool = True,
        anchor_k: int = 10,
        final_k: int = 5,
        max_candidates: int = 50,
        sleep_interval: int = 3,
        diagnosis_workers: int = 8,
        semantic_threshold: float = 0.65,
        semantic_out_degree: int = 8,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_ingest_chars: int = 15000,
    ) -> None:
        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._embedding_device = embedding_device
        self._reranker_model = reranker_model
        self._reranker_device = reranker_device
        self._use_reranker = use_reranker
        self._anchor_k = anchor_k
        self._final_k = final_k
        self._max_candidates = max_candidates
        self._sleep_interval = sleep_interval
        self._diagnosis_workers = diagnosis_workers
        self._semantic_threshold = semantic_threshold
        self._semantic_out_degree = semantic_out_degree
        self._api_base = api_base
        self._api_key = api_key
        self._max_ingest_chars = max_ingest_chars
        self._make()

    def _make(self) -> None:
        from memgym.memory.external.allmem import LiteLLMController, MemGymAllMemSystem

        controller = LiteLLMController(
            model=self._llm_model,
            api_base=self._api_base,
            api_key=self._api_key,
        )
        self._system = MemGymAllMemSystem(
            controller,
            embedding_model=self._embedding_model,
            embedding_device=self._embedding_device,
            reranker_model=self._reranker_model,
            reranker_device=self._reranker_device,
            diagnosis_workers=self._diagnosis_workers,
            use_reranker=self._use_reranker,
        )
        self._system.graph.threshold_semantic = self._semantic_threshold
        self._writes = 0

    # -- CodingMemoryMethod interface ----------------------------------------

    def ingest(self, doc_name: str, doc_content: str, task_prompt: str) -> None:
        node_id = self._system.wake_process(
            doc_content[: self._max_ingest_chars],
            source_id=doc_name,
        )
        # Reach the configured semantic out-degree cap (d_sigma); wake_process's
        # create_semantic_edges uses k=3 by default.
        if self._semantic_out_degree > 3 and self._system.graph.graph.has_node(node_id):
            node = self._system.graph.graph.nodes[node_id]["data"]
            self._system.graph.create_semantic_edges(node, k=self._semantic_out_degree)

        self._writes += 1
        if self._sleep_interval and self._writes % self._sleep_interval == 0:
            self._system.sleep_process()

    def retrieve(self, question: str, task_prompt: str) -> str:
        self._system.sleep_process()  # final consolidation flush (idempotent)
        ctx, _source_ids, _latencies = self._system.get_context_for_query(
            question,
            anchor_k=self._anchor_k,
            final_k=self._final_k,
            max_candidates=self._max_candidates,
        )
        return ctx if ctx else "(no relevant memories found)"

    def reset(self) -> None:
        self._make()

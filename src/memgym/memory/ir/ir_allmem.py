"""All-Mem adapter for MemGym-IR (deep-research) evaluation.

Bridges the vendored All-Mem engine (dynamic topology graph + wake/sleep
consolidation + topology-aware retrieval) into the ``BaseMemoryManager``
interface the IR evaluator drives.

Mapping (see ``custom_memory.md`` §3, §6c), faithful to ``All-Mem/all_mem``:

- ``manage_context`` (one search turn) -> ``wake_process`` (online write:
  LLM index + temporal/semantic edges + queued diagnosis). A ``sleep_process``
  is triggered every ``sleep_interval`` turns (offline Agentic Topology
  Consolidation: confidence-gated Split / Merge / Update).
- ``retrieve_for_question`` -> one final ``sleep_process`` (idempotent flush of
  any pending diagnoses) then ``get_context_for_query`` (ANN anchors -> typed
  link expansion -> cross-encoder rerank).
- ``get_notes_text`` -> timestamp-ordered dump of the active memory surface.

The memory LLM is driven through LiteLLM (``LiteLLMController``), so it follows
the same endpoint as the rest of MemGym: pass ``llm_model`` (e.g. the IR
``--summarization-model``) and rely on ``OPENAI_API_BASE`` / ``OPENAI_API_KEY``
env for SGLang / OpenAI / Gemini-via-OpenAI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..base import BaseMemoryManager, FilteredContext, register_memory_model
from ..external.allmem import LiteLLMController, MemGymAllMemSystem


class IRAllMemMemory(BaseMemoryManager):
    """All-Mem memory: dynamic topology graph with wake/sleep consolidation."""

    def __init__(
        self,
        max_tokens: int = 100000,
        question: str = "",
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_device: Optional[str] = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_device: str = "cpu",
        use_reranker: bool = True,
        anchor_k: int = 10,
        final_k: int = 10,
        max_candidates: int = 50,
        sleep_interval: int = 3,
        diagnosis_workers: int = 8,
        semantic_threshold: float = 0.65,
        semantic_out_degree: int = 8,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(max_tokens=max_tokens, **kwargs)
        self._question = question
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

        self._system = self._create_system()
        self._all_observations: List[str] = []
        self._writes = 0

    def _create_system(self) -> MemGymAllMemSystem:
        controller = LiteLLMController(
            model=self._llm_model,
            api_base=self._api_base,
            api_key=self._api_key,
        )
        system = MemGymAllMemSystem(
            controller,
            embedding_model=self._embedding_model,
            embedding_device=self._embedding_device,
            reranker_model=self._reranker_model,
            reranker_device=self._reranker_device,
            diagnosis_workers=self._diagnosis_workers,
            use_reranker=self._use_reranker,
        )
        # Paper hyperparameters (custom_memory.md §7): semantic edge threshold
        # and out-degree cap d_sigma. create_semantic_edges' k is set per-call
        # in _wake (the engine default is 3; the paper uses 8).
        system.graph.threshold_semantic = self._semantic_threshold
        return system

    def manage_context(
        self,
        original_context: List[Any],
        current_observation: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FilteredContext:
        """Ingest one search turn as an All-Mem node (wake), consolidate on cadence."""
        if current_observation:
            text = str(current_observation)
            self._all_observations.append(text)
            meta = metadata or {}
            source_id = meta.get("source_id")
            if source_id is None and "turn_index" in meta:
                source_id = f"turn-{meta['turn_index']}"
            # Truncate very long observations to keep the indexing LLM bounded.
            node_id = self._system.wake_process(
                text[:15000],
                timestamp=meta.get("timestamp"),
                source_id=source_id,
            )
            # Match the paper's semantic out-degree cap (d_sigma): the engine's
            # wake_process uses create_semantic_edges' default k=3, so add the
            # extra edges explicitly to reach the configured cap.
            if self._semantic_out_degree > 3 and self._system.graph.graph.has_node(node_id):
                node = self._system.graph.graph.nodes[node_id]["data"]
                self._system.graph.create_semantic_edges(node, k=self._semantic_out_degree)

            self._writes += 1
            if self._sleep_interval and self._writes % self._sleep_interval == 0:
                self._system.sleep_process()

        tokens = self.count_tokens(self._all_observations)
        self._update_token_tracking(tokens)
        return FilteredContext(
            content=self._all_observations.copy(),
            metadata={
                "tokens": tokens,
                "original_tokens": tokens,
                "was_compacted": False,
                "strategy": "ir_allmem",
                "num_nodes": self._system.graph.graph.number_of_nodes(),
                "active_nodes": len(self._system.graph.node_cache),
            },
        )

    def retrieve_for_question(self, question: str, top_k: Optional[int] = None) -> str:
        """Topology-aware retrieval for the final answer.

        Runs one final sleep (offline consolidation flush) so all pending
        diagnoses are applied, then queries the graph: ANN anchors -> typed-link
        expansion (recovers archived evidence) -> rerank.
        """
        self._system.sleep_process()
        ctx, _source_ids, _latencies = self._system.get_context_for_query(
            question,
            anchor_k=self._anchor_k,
            final_k=top_k or self._final_k,
            max_candidates=self._max_candidates,
        )
        return ctx if ctx else "(no relevant memories found)"

    def get_notes_text(self) -> str:
        """Return the active memory surface as timestamp-ordered text."""
        nodes = [
            data["data"]
            for _, data in self._system.graph.graph.nodes(data=True)
            if data.get("data") is not None and data["data"].status == "Active"
        ]
        nodes.sort(key=lambda n: n.timestamp or "")
        return "\n".join(
            f"[{n.timestamp or 'Unknown Time'}] {n.first_content()}" for n in nodes
        )

    def reset(self) -> None:
        self._system = self._create_system()
        self._all_observations.clear()
        self._writes = 0
        self._token_count = 0
        self._token_history.clear()
        self._compaction_count = 0
        self._step_count = 0

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        graph = self._system.graph.graph
        active = len(self._system.graph.node_cache)
        total = graph.number_of_nodes()
        stats.update({
            "total_nodes": total,
            "active_nodes": active,
            "archived_nodes": total - active,
            "edges": graph.number_of_edges(),
            "anchor_k": self._anchor_k,
            "final_k": self._final_k,
            "sleep_interval": self._sleep_interval,
            "use_reranker": self._system.use_reranker,
        })
        return stats


register_memory_model("ir_allmem", IRAllMemMemory)
register_memory_model("ir-allmem", IRAllMemMemory)

"""MemGym-side extensions to the vendored All-Mem engine.

``core.py`` is kept byte-for-byte identical to upstream. Any behavior MemGym
needs that upstream does not provide lives here as a subclass, so the engine
stays faithful and the extensions stay auditable.

Currently this adds only **reranker robustness**: All-Mem's
``AllMemSystem._rank_candidates`` loads a CrossEncoder reranker
(``cross-encoder/ms-marco-MiniLM-L-6-v2``) on every query and falls back to
embedding cosine on failure — but it retries the (potentially network-bound)
load on *every* query. In an eval loop that can hang per question. This subclass:

- exposes a ``use_reranker`` flag (default True, faithful to the paper), and
- on the first reranker failure, flips the flag off so subsequent queries skip
  straight to the cosine fallback instead of re-attempting the load.

The cosine fallback path is copied verbatim from upstream ``_rank_candidates``.
"""

from __future__ import annotations

from typing import List, Tuple

from .core import AllMemNode, AllMemSystem, get_shared_reranker, logger, cosine_similarity


class MemGymAllMemSystem(AllMemSystem):
    """All-Mem system with a one-shot reranker fallback for eval stability."""

    def __init__(self, *args, use_reranker: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_reranker = use_reranker

    def _rank_candidates(
        self,
        user_query: str,
        candidates: List[AllMemNode],
    ) -> List[Tuple[AllMemNode, float]]:
        if not candidates:
            return []

        if self.use_reranker:
            try:
                reranker = get_shared_reranker(self.reranker_model, self.reranker_device)
                pairs = [[user_query, node.get_index_text()] for node in candidates]
                scores = reranker.predict(pairs)
                return sorted(
                    zip(candidates, scores), key=lambda item: float(item[1]), reverse=True
                )
            except Exception as exc:  # noqa: BLE001
                # Disable for the rest of this run so we don't re-attempt the
                # (possibly network-bound) load on every subsequent query.
                logger.warning(
                    "All-Mem reranker unavailable; using embedding cosine for the "
                    "rest of this run: %s",
                    exc,
                )
                self.use_reranker = False

        # Embedding cosine fallback (verbatim from upstream _rank_candidates).
        q_emb = self.graph.compute_embedding(user_query)
        scored = []
        for node in candidates:
            if node.embedding is None:
                node.embedding = self.graph.compute_embedding(node.get_index_text())
            score = cosine_similarity([q_emb], [node.embedding])[0][0]
            scored.append((node, float(score)))
        return sorted(scored, key=lambda item: item[1], reverse=True)

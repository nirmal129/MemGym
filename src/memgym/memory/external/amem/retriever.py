"""
Retriever: Embedding-based memory retrieval.

Provides semantic search over memories using SentenceTransformers.
"""

import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from .note import MemoryNote


# SentenceTransformer construction loads weights via a meta-device init
# (transformers' low_cpu_mem_usage path) and then moves them to the target
# device. That meta->device step relies on accelerate's init_empty_weights()
# monkeypatch of torch.nn.Module, which is NOT thread-safe: two threads
# building a model at once race on the patch's enter/exit, raising
# "Cannot copy out of meta tensor; no data!" and leaving the process globally
# stuck in meta-init mode (so every later load, even single-threaded, fails).
#
# Guard construction with a process-global lock and cache one model per
# model_name. Only the (brief) build is serialized; encoding/retrieval stay
# fully parallel, so callers can run many EmbeddingRetrievers across threads.
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_MODEL_LOCK = threading.Lock()


def _get_shared_model(model_name: str) -> SentenceTransformer:
    """Return a shared SentenceTransformer, building it at most once per name."""
    model = _MODEL_CACHE.get(model_name)
    if model is not None:
        return model
    with _MODEL_LOCK:
        # Re-check inside the lock: another thread may have built it while we
        # waited.
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name)
            _MODEL_CACHE[model_name] = model
    return model


class EmbeddingRetriever:
    """
    Simple embedding-based retrieval using SentenceTransformers.

    This retriever:
    1. Encodes documents using a sentence transformer model
    2. Uses cosine similarity for retrieval
    3. Supports incremental document addition

    Config options:
        model_name: SentenceTransformer model (default: all-MiniLM-L6-v2)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding retriever.

        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model = _get_shared_model(model_name)
        self.corpus: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.document_ids: Dict[str, int] = {}  # Map document content to index

    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the retriever.

        Args:
            documents: List of text documents to add
        """
        if not documents:
            return

        if not self.corpus:
            # First batch of documents
            self.corpus = documents
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            # Append to existing
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)

            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])

            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx

    def add_document(self, document: str) -> bool:
        """
        Add a single document to the retriever.

        Args:
            document: Text content to add

        Returns:
            True if document was added, False if already exists
        """
        if document in self.document_ids:
            return False

        self.add_documents([document])
        return True

    def search(self, query: str, k: int = 5) -> List[int]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of document indices sorted by relevance
        """
        if not self.corpus or self.embeddings is None:
            return []

        # Encode query
        query_embedding = self.model.encode([query])[0]

        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Get top k indices
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return top_k_indices.tolist()

    def reset(self) -> None:
        """Reset the retriever, clearing all documents."""
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}

    def save(self, cache_file: str, embeddings_file: str) -> None:
        """
        Save retriever state to disk.

        Args:
            cache_file: Path for corpus and metadata
            embeddings_file: Path for embeddings (.npy)
        """
        if self.embeddings is not None:
            np.save(embeddings_file, self.embeddings)

        state = {
            "corpus": self.corpus,
            "document_ids": self.document_ids,
            "model_name": self.model_name
        }
        with open(cache_file, "wb") as f:
            pickle.dump(state, f)

    def load(self, cache_file: str, embeddings_file: str) -> "EmbeddingRetriever":
        """
        Load retriever state from disk.

        Args:
            cache_file: Path for corpus and metadata
            embeddings_file: Path for embeddings (.npy)

        Returns:
            Self for chaining
        """
        cache_path = Path(cache_file)
        embeddings_path = Path(embeddings_file)

        if embeddings_path.exists():
            self.embeddings = np.load(str(embeddings_path))

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                state = pickle.load(f)
                self.corpus = state["corpus"]
                self.document_ids = state.get("document_ids", {})

        return self

    @classmethod
    def from_memories(
        cls,
        memories: Dict[str, "MemoryNote"],
        model_name: str = "all-MiniLM-L6-v2"
    ) -> "EmbeddingRetriever":
        """
        Create retriever from existing memories.

        Combines memory content with metadata for better retrieval.

        Args:
            memories: Dict mapping memory ID to MemoryNote
            model_name: SentenceTransformer model name

        Returns:
            Initialized retriever with all memories
        """
        all_docs = []
        for m in memories.values():
            # Combine content with metadata for richer retrieval
            metadata_text = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)

        retriever = cls(model_name)
        retriever.add_documents(all_docs)
        return retriever

    def __len__(self) -> int:
        """Return number of documents in retriever."""
        return len(self.corpus)

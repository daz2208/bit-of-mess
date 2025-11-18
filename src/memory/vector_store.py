"""Vector store for semantic similarity search."""

import numpy as np
from typing import List, Tuple, Optional
from collections import Counter
import re
import math

from ..models.memory import MemoryEntry
from ..models.base import MemoryType
from ..storage.repositories import MemoryRepository


class VectorStore:
    """TF-IDF based vector store for semantic search."""

    def __init__(self, memory_repo: MemoryRepository):
        self.memory_repo = memory_repo
        self._vocabulary: dict = {}
        self._idf: dict = {}
        self._vocab_size = 0

    def create_embedding(self, text: str) -> np.ndarray:
        """Create TF-IDF embedding for text."""
        tokens = self._tokenize(text)

        if not tokens:
            return np.zeros(max(1, self._vocab_size), dtype=np.float32)

        # Build vocabulary if needed
        if not self._vocabulary:
            self._build_vocabulary([text])

        # Create TF vector
        tf = Counter(tokens)
        total_terms = len(tokens)

        # Create embedding
        embedding = np.zeros(self._vocab_size, dtype=np.float32)
        for token, count in tf.items():
            if token in self._vocabulary:
                idx = self._vocabulary[token]
                tfidf = (count / total_terms) * self._idf.get(token, 1.0)
                embedding[idx] = tfidf

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after', 'above',
                      'below', 'between', 'under', 'again', 'further', 'then', 'once',
                      'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                      'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very',
                      'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why',
                      'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
                      'other', 'some', 'such', 'no', 'not', 'only', 'own', 'same',
                      'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don',
                      'should', 'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
                      'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
                      'it', 'its', 'they', 'them', 'their', 'theirs', 'what', 'which',
                      'who', 'whom', 'this', 'that', 'these', 'those', 'am'}
        return [t for t in tokens if t not in stop_words and len(t) > 2]

    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary and IDF from texts."""
        all_tokens = set()
        doc_freq = Counter()

        for text in texts:
            tokens = set(self._tokenize(text))
            all_tokens.update(tokens)
            for token in tokens:
                doc_freq[token] += 1

        self._vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        self._vocab_size = len(self._vocabulary)

        # Calculate IDF
        num_docs = max(1, len(texts))
        self._idf = {
            token: math.log(num_docs / (1 + freq))
            for token, freq in doc_freq.items()
        }

    def rebuild_index(self, user_id: str):
        """Rebuild vocabulary from all user memories."""
        memories = self.memory_repo.get_by_user(user_id, limit=10000)
        if memories:
            texts = [m.content for m in memories]
            self._build_vocabulary(texts)

    async def similarity_search(
        self,
        user_id: str,
        query: str,
        memory_type: Optional[MemoryType] = None,
        top_k: int = 10,
        recency_weight: float = 0.3
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search for similar memories using cosine similarity."""

        # Get memories
        memories = self.memory_repo.get_by_user(user_id, memory_type, limit=1000)

        if not memories:
            return []

        # Rebuild vocabulary with current memories
        texts = [m.content for m in memories]
        self._build_vocabulary(texts + [query])

        # Create query embedding
        query_embedding = self.create_embedding(query)

        results = []
        now = np.datetime64('now')

        for memory in memories:
            # Create or use existing embedding
            if memory.embedding is None or len(memory.embedding) != self._vocab_size:
                memory.embedding = self.create_embedding(memory.content)
                self.memory_repo.save(memory)

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, memory.embedding)

            # Apply recency weighting
            if recency_weight > 0:
                days_old = (now - np.datetime64(memory.last_accessed)) / np.timedelta64(1, 'D')
                recency_score = np.exp(-days_old / 30)  # 30-day decay
                final_score = (1 - recency_weight) * similarity + recency_weight * recency_score
            else:
                final_score = similarity

            # Weight by importance
            final_score *= (0.5 + 0.5 * memory.importance)

            results.append((memory, float(final_score)))

        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            # Pad shorter vector
            max_len = max(len(a), len(b))
            a = np.pad(a, (0, max_len - len(a)))
            b = np.pad(b, (0, max_len - len(b)))

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

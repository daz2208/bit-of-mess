"""Enhanced embeddings with n-grams and better similarity."""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re
import math


class EnhancedEmbeddings:
    """
    Enhanced embedding system with:
    - N-gram support (unigrams, bigrams, trigrams)
    - Better TF-IDF weighting
    - Semantic similarity boosting
    """

    def __init__(self):
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._vocab_size = 0
        self._doc_count = 0

        # Semantic clusters for similarity boosting
        self.semantic_clusters = {
            "time": ["morning", "afternoon", "evening", "night", "today", "tomorrow", "week", "month", "schedule", "calendar"],
            "communication": ["email", "message", "call", "chat", "meeting", "talk", "discuss", "conversation"],
            "task": ["create", "make", "build", "write", "send", "update", "delete", "add", "remove", "task", "todo"],
            "positive": ["good", "great", "excellent", "perfect", "love", "like", "happy", "thanks"],
            "negative": ["bad", "wrong", "hate", "dislike", "frustrated", "annoying", "issue", "problem"],
            "question": ["what", "where", "when", "why", "how", "which", "who"],
            "preference": ["prefer", "like", "want", "need", "always", "never", "favorite"]
        }

        # Build reverse cluster lookup
        self._word_to_cluster = {}
        for cluster, words in self.semantic_clusters.items():
            for word in words:
                self._word_to_cluster[word] = cluster

    def create_embedding(self, text: str, use_ngrams: bool = True) -> np.ndarray:
        """Create enhanced embedding for text."""
        # Tokenize with optional n-grams
        if use_ngrams:
            tokens = self._tokenize_with_ngrams(text)
        else:
            tokens = self._tokenize(text)

        if not tokens:
            return np.zeros(max(1, self._vocab_size), dtype=np.float32)

        # Build vocabulary if needed
        if not self._vocabulary:
            self._build_vocabulary_from_tokens([tokens])

        # Calculate TF-IDF
        tf = Counter(tokens)
        total_terms = len(tokens)

        embedding = np.zeros(self._vocab_size, dtype=np.float32)
        for token, count in tf.items():
            if token in self._vocabulary:
                idx = self._vocabulary[token]
                # TF with sublinear scaling
                tf_score = 1 + math.log(count) if count > 0 else 0
                idf_score = self._idf.get(token, 1.0)
                embedding[idx] = tf_score * idf_score

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenization."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)

        # Enhanced stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'and', 'but', 'or', 'nor', 'so', 'yet',
            'both', 'either', 'neither', 'not', 'only', 'own', 'same', 'than',
            'too', 'very', 'just', 'also', 'now', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'can', 'don', 'i', 'me', 'my', 'myself',
            'we', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him', 'his', 'she',
            'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs', 'this',
            'that', 'these', 'those', 'am', 'if', 'else', 'while', 'about', 'against',
            'up', 'down', 'out', 'off', 'over', 'any', 'because', 'until', 'which',
            'who', 'whom', 'what'
        }

        return [t for t in tokens if t not in stop_words and len(t) > 1]

    def _tokenize_with_ngrams(self, text: str) -> List[str]:
        """Tokenize with unigrams, bigrams, and trigrams."""
        unigrams = self._tokenize(text)

        if len(unigrams) < 2:
            return unigrams

        # Generate bigrams
        bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams)-1)]

        # Generate trigrams
        trigrams = []
        if len(unigrams) >= 3:
            trigrams = [f"{unigrams[i]}_{unigrams[i+1]}_{unigrams[i+2]}"
                       for i in range(len(unigrams)-2)]

        # Combine with weights (unigrams most important)
        all_tokens = unigrams + bigrams + trigrams

        return all_tokens

    def _build_vocabulary_from_tokens(self, token_lists: List[List[str]]):
        """Build vocabulary from token lists."""
        all_tokens = set()
        doc_freq = Counter()

        for tokens in token_lists:
            unique_tokens = set(tokens)
            all_tokens.update(unique_tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        self._vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        self._vocab_size = len(self._vocabulary)
        self._doc_count = len(token_lists)

        # Calculate IDF with smoothing
        for token, freq in doc_freq.items():
            self._idf[token] = math.log((self._doc_count + 1) / (freq + 1)) + 1

    def rebuild_from_texts(self, texts: List[str]):
        """Rebuild vocabulary from a list of texts."""
        token_lists = [self._tokenize_with_ngrams(text) for text in texts]
        self._build_vocabulary_from_tokens(token_lists)

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity with cluster boosting.
        """
        emb1 = self.create_embedding(text1)
        emb2 = self.create_embedding(text2)

        # Cosine similarity
        cosine_sim = self._cosine_similarity(emb1, emb2)

        # Semantic cluster boost
        cluster_boost = self._calculate_cluster_boost(text1, text2)

        # Combine (cosine is primary, cluster boost adds up to 20%)
        final_sim = cosine_sim + (cluster_boost * 0.2 * (1 - cosine_sim))

        return min(1.0, final_sim)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        if len(a) != len(b):
            max_len = max(len(a), len(b))
            a = np.pad(a, (0, max_len - len(a)))
            b = np.pad(b, (0, max_len - len(b)))

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _calculate_cluster_boost(self, text1: str, text2: str) -> float:
        """Calculate boost based on shared semantic clusters."""
        words1 = set(self._tokenize(text1))
        words2 = set(self._tokenize(text2))

        clusters1 = {self._word_to_cluster.get(w) for w in words1} - {None}
        clusters2 = {self._word_to_cluster.get(w) for w in words2} - {None}

        if not clusters1 or not clusters2:
            return 0.0

        shared = clusters1 & clusters2
        total = clusters1 | clusters2

        return len(shared) / len(total) if total else 0.0

    def find_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find most similar texts from candidates."""
        # Rebuild vocab with all texts
        self.rebuild_from_texts([query] + candidates)

        query_emb = self.create_embedding(query)

        results = []
        for candidate in candidates:
            sim = self.semantic_similarity(query, candidate)
            results.append((candidate, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

from __future__ import annotations

import math
from collections import Counter, defaultdict

from evaluation_llm.catalog import LabelCatalog
from evaluation_llm.interfaces import CandidateProvider
from evaluation_llm.types import CandidateRecord, FragmentExample


def _iter_kmers(sequence: str, k: int = 3) -> list[str]:
    sequence = sequence.strip().upper()
    if not sequence:
        return []
    if len(sequence) < k:
        return [f"__SHORT__:{sequence}"]
    return [sequence[index : index + k] for index in range(len(sequence) - k + 1)]


def _example_kmer_counts(example: FragmentExample, k: int = 3) -> Counter[str]:
    counts: Counter[str] = Counter()
    for fragment in example.fragment_parts:
        counts.update(_iter_kmers(fragment, k=k))
    return counts


def _tfidf_from_counts(counts: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {
        token: (count / total) * idf.get(token, 0.0)
        for token, count in counts.items()
    }


def _vector_norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _cosine_similarity(
    left_vector: dict[str, float],
    left_norm: float,
    right_vector: dict[str, float],
    right_norm: float,
) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    shared = set(left_vector).intersection(right_vector)
    dot_product = sum(left_vector[token] * right_vector[token] for token in shared)
    return dot_product / (left_norm * right_norm)


class FullCatalogCandidateProvider(CandidateProvider):
    def __init__(self, catalog: LabelCatalog) -> None:
        self.catalog = catalog

    def get_candidates(self, example: FragmentExample, top_k: int | None = None) -> list[CandidateRecord]:
        cards = self.catalog.sorted_cards()
        return [
            CandidateRecord(accession=card.accession, score=1.0, rank=index + 1, source="full_catalog")
            for index, card in enumerate(cards)
        ]


class TopKPrototypeCandidateProvider(CandidateProvider):
    def __init__(
        self,
        catalog: LabelCatalog,
        train_examples: list[FragmentExample],
        kmer_size: int = 3,
    ) -> None:
        self.catalog = catalog
        self.train_examples = train_examples
        self.kmer_size = kmer_size
        self.prototype_vectors: dict[str, dict[str, float]] = {}
        self.prototype_norms: dict[str, float] = {}
        self.train_example_vectors: dict[str, tuple[dict[str, float], float]] = {}
        self.label_to_examples: dict[str, list[FragmentExample]] = defaultdict(list)
        self.idf = self._build_index()

    def _build_index(self) -> dict[str, float]:
        label_documents: dict[str, Counter[str]] = defaultdict(Counter)
        token_document_frequency: Counter[str] = Counter()

        for example in self.train_examples:
            counts = _example_kmer_counts(example, k=self.kmer_size)
            label_documents[example.interpro_id].update(counts)
            self.label_to_examples[example.interpro_id].append(example)

        for counts in label_documents.values():
            for token in counts.keys():
                token_document_frequency[token] += 1

        total_documents = max(len(label_documents), 1)
        idf = {
            token: math.log((1 + total_documents) / (1 + frequency)) + 1.0
            for token, frequency in token_document_frequency.items()
        }

        for accession, counts in label_documents.items():
            vector = _tfidf_from_counts(counts, idf)
            self.prototype_vectors[accession] = vector
            self.prototype_norms[accession] = _vector_norm(vector)

        for example in self.train_examples:
            vector = _tfidf_from_counts(_example_kmer_counts(example, k=self.kmer_size), idf)
            self.train_example_vectors[example.uid] = (vector, _vector_norm(vector))

        return idf

    def get_candidates(self, example: FragmentExample, top_k: int | None = None) -> list[CandidateRecord]:
        top_k = top_k or len(self.prototype_vectors)
        query_vector = _tfidf_from_counts(_example_kmer_counts(example, k=self.kmer_size), self.idf)
        query_norm = _vector_norm(query_vector)

        scored = []
        for accession, vector in self.prototype_vectors.items():
            score = _cosine_similarity(query_vector, query_norm, vector, self.prototype_norms[accession])
            scored.append((score, accession))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            CandidateRecord(accession=accession, score=score, rank=index + 1, source="prototype_tfidf_3mer")
            for index, (score, accession) in enumerate(scored[:top_k])
        ]

    def get_few_shots(
        self,
        example: FragmentExample,
        candidate_ids: list[str],
        limit: int,
    ) -> list[FragmentExample]:
        if limit <= 0:
            return []

        query_vector = _tfidf_from_counts(_example_kmer_counts(example, k=self.kmer_size), self.idf)
        query_norm = _vector_norm(query_vector)

        scored_examples = []
        for accession in candidate_ids:
            for candidate_example in self.label_to_examples.get(accession, []):
                vector, norm = self.train_example_vectors[candidate_example.uid]
                score = _cosine_similarity(query_vector, query_norm, vector, norm)
                scored_examples.append((score, accession, candidate_example.uid, candidate_example))

        scored_examples.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected: list[FragmentExample] = []
        seen_uids: set[str] = set()
        for _, _, uid, candidate_example in scored_examples:
            if uid in seen_uids:
                continue
            selected.append(candidate_example)
            seen_uids.add(uid)
            if len(selected) >= limit:
                break
        return selected

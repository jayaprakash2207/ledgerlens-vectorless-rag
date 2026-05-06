import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ChunkResult:
    chunk_id: str
    score: float
    section: str
    text: str


@dataclass
class ChunkExplanation:
    query_terms: List[str]
    ranked: List[Dict[str, object]]


class ChunkRetriever:
    def __init__(self) -> None:
        self.priority_sections = {"risk_factors", "mda", "notes", "legal", "auditor_notes"}

    def retrieve(self, query: str, chunks: List[Dict[str, object]], top_k: int = 5) -> List[ChunkResult]:
        ranked = self._rank(query, chunks)
        return ranked[:top_k]

    def explain(self, query: str, chunks: List[Dict[str, object]], top_k: int = 5) -> ChunkExplanation:
        ranked = self._rank(query, chunks)
        return ChunkExplanation(
            query_terms=sorted(self._tokenize(query)),
            ranked=[
                {
                    "chunk_id": item.chunk_id,
                    "score": round(item.score, 3),
                    "section": item.section,
                    "preview": item.text[:200],
                }
                for item in ranked[:top_k]
            ],
        )

    def _rank(self, query: str, chunks: List[Dict[str, object]]) -> List[ChunkResult]:
        query_terms = self._tokenize(query)
        results: List[ChunkResult] = []

        for chunk in chunks:
            text = str(chunk.get("text", ""))
            if not text:
                continue
            section = str(chunk.get("section", ""))
            base_score = 1.0 if section in self.priority_sections else 0.6
            keyword_score, matched = self._keyword_score(query_terms, text)
            length_score = min(len(text) / 1200, 1.0)
            total = base_score + keyword_score + length_score * 0.1
            results.append(
                ChunkResult(
                    chunk_id=str(chunk.get("chunk_id")),
                    score=total,
                    section=section,
                    text=text,
                )
            )

        return sorted(results, key=lambda item: item.score, reverse=True)

    def _keyword_score(self, query_terms: set, text: str) -> Tuple[float, List[str]]:
        if not query_terms:
            return 0.0, []
        text_lower = text.lower()
        matched = [term for term in query_terms if term in text_lower]
        score = len(matched) / max(len(query_terms), 1)
        return score, matched

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r"[a-zA-Z']+", text.lower()))

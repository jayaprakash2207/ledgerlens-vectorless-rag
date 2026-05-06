import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QAResult:
    answer: str
    sources: List[str]


class QAEngine:
    def __init__(self, llm_client: Optional[object]) -> None:
        self.llm_client = llm_client

    def answer(self, question: str, chunks: List[str]) -> QAResult:
        if not self.llm_client:
            return QAResult(answer="LLM client not configured.", sources=[])

        if not chunks:
            return QAResult(answer="No relevant context was retrieved.", sources=[])

        context = "\n\n".join(chunks[:6])
        prompt = (
            "You are a financial filings analyst. Answer the user question using only the provided context. "
            "If the context is insufficient, say so. Provide a concise answer.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Return JSON with keys: answer (string), sources (array of short quotes)."
        )
        raw = self.llm_client.generate(prompt)
        if not raw:
            last_error = getattr(self.llm_client, "last_error", None)
            message = "No response from LLM."
            if last_error:
                message = f"No response from LLM ({last_error})."
            return QAResult(answer=message, sources=[])

        try:
            data = json.loads(self._extract_json(raw))
            answer = str(data.get("answer", "")).strip()
            sources = [str(item) for item in data.get("sources", [])]
            if not answer:
                last_error = getattr(self.llm_client, "last_error", None)
                message = "No answer returned by LLM."
                if last_error:
                    message = f"No answer returned by LLM ({last_error})."
                return QAResult(answer=message, sources=sources)
            return QAResult(answer=answer, sources=sources)
        except Exception:
            return QAResult(answer=raw.strip(), sources=[])

    def _extract_json(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return "{}"
        return text[start : end + 1]

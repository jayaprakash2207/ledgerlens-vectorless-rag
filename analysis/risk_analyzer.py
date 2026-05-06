import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RiskAnalysisResult:
    top_risks: List[str]
    risk_categories: Dict[str, List[str]]
    red_flags: List[str]
    confidence_score: int
    summary: str
    risky_sentences: List[str]


class RiskAnalyzer:
    def __init__(self, llm_client: Optional[object] = None, require_llm: bool = False) -> None:
        self.llm_client = llm_client
        self.require_llm = require_llm
        self.risk_terms = ["risk", "uncertain", "volatility", "may", "could", "potential", "threat"]

    def analyze(self, text: str) -> RiskAnalysisResult:
        text = text or ""
        if not text.strip():
            return RiskAnalysisResult(
                top_risks=[],
                risk_categories={"Financial": [], "Operational": [], "Market": [], "Regulatory": []},
                red_flags=[],
                confidence_score=0,
                summary="No text available for analysis.",
                risky_sentences=[],
            )

        if self.llm_client:
            response = self._analyze_with_llm(text)
            if response:
                print("RiskAnalyzer: used LLM response")
                return response
            if self.require_llm:
                raise RuntimeError("LLM did not return a valid response.")

        return self._heuristic_analysis(text)

    def _analyze_with_llm(self, text: str) -> Optional[RiskAnalysisResult]:
        prompt = (
            "You are a financial risk analyst. Extract the top risks, categorize them, "
            "identify red flags, and summarize. Return JSON with keys: top_risks (array), "
            "risk_categories (object with keys Financial, Operational, Market, Regulatory), "
            "red_flags (array), confidence_score (0-100), summary (string). "
            "If unsure, still return best-effort JSON.\n\n"
            f"TEXT:\n{text[:12000]}"
        )
        raw = self.llm_client.generate(prompt)
        if not raw:
            return None

        try:
            data = json.loads(self._extract_json(raw))
            return RiskAnalysisResult(
                top_risks=data.get("top_risks", []),
                risk_categories=data.get("risk_categories", {}),
                red_flags=data.get("red_flags", []),
                confidence_score=int(data.get("confidence_score", 50)),
                summary=data.get("summary", ""),
                risky_sentences=self._find_risky_sentences(text),
            )
        except Exception:
            return None

    def _heuristic_analysis(self, text: str) -> RiskAnalysisResult:
        risky_sentences = self._find_risky_sentences(text)
        top_risks = self._extract_top_risks(text)
        risk_categories = self._categorize_risks(top_risks)
        red_flags = self._find_red_flags(text)
        confidence_score = self._score_confidence(text, red_flags)
        summary = self._build_summary(top_risks, red_flags)

        print(
            "RiskAnalyzer: risky_sentences=%s top_risks=%s red_flags=%s"
            % (len(risky_sentences), len(top_risks), len(red_flags))
        )

        return RiskAnalysisResult(
            top_risks=top_risks,
            risk_categories=risk_categories,
            red_flags=red_flags,
            confidence_score=confidence_score,
            summary=summary,
            risky_sentences=risky_sentences,
        )

    def _find_risky_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        risky = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(term in lower for term in self.risk_terms):
                risky.append(sentence.strip())
        return risky[:50]

    def _extract_top_risks(self, text: str) -> List[str]:
        candidates = []
        for sentence in self._find_risky_sentences(text):
            candidates.append(sentence)
        return candidates[:10]

    def _categorize_risks(self, risks: List[str]) -> Dict[str, List[str]]:
        categories = {"Financial": [], "Operational": [], "Market": [], "Regulatory": []}
        for risk in risks:
            lower = risk.lower()
            if any(word in lower for word in ["liquidity", "debt", "cash", "credit", "financing"]):
                categories["Financial"].append(risk)
            elif any(word in lower for word in ["supply", "operations", "technology", "cyber", "staff"]):
                categories["Operational"].append(risk)
            elif any(word in lower for word in ["competition", "demand", "pricing", "market"]):
                categories["Market"].append(risk)
            elif any(word in lower for word in ["regulation", "compliance", "legal", "policy"]):
                categories["Regulatory"].append(risk)

        return categories

    def _find_red_flags(self, text: str) -> List[str]:
        flags = []
        patterns = [
            r"material weakness",
            r"going concern",
            r"restatement",
            r"significant doubt",
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                flags.append(pattern)
        return flags

    def _score_confidence(self, text: str, red_flags: List[str]) -> int:
        risk_count = len(self._find_risky_sentences(text))
        score = min(100, 40 + risk_count + len(red_flags) * 10)
        return max(0, score)

    def _build_summary(self, risks: List[str], flags: List[str]) -> str:
        if not risks:
            return "No material risks detected from the extracted sections."
        summary = f"Identified {len(risks)} risk statements with {len(flags)} red flags."
        return summary

    def _extract_json(self, text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return "{}"
        return match.group(0)

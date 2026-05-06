import os
from typing import Optional

import requests


class GeminiClient:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.last_error: Optional[str] = None

    def generate(self, prompt: str, temperature: float = 0.2) -> Optional[str]:
        if not self.api_key:
            self.last_error = "GEMINI_API_KEY not set"
            print("GeminiClient: GEMINI_API_KEY not set")
            return None

        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        params = {"key": self.api_key}

        try:
            response = requests.post(url, json=payload, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()
            text = self._extract_text(data)
            if not text:
                self.last_error = "Empty response from Gemini"
                return None
            self.last_error = None
            return text
        except Exception as exc:
            self.last_error = str(exc)
            print(f"GeminiClient: request failed ({exc})")
            return None

    def _extract_text(self, data: dict) -> Optional[str]:
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            return None
        return parts[0].get("text")

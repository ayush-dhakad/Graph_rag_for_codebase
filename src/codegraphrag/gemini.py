"""Minimal Gemini REST client that exposes provider-reported token usage."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class GeminiResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    thought_tokens: int
    cached_tokens: int
    total_tokens: int
    model: str


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str = "gemini-3.6-flash") -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Set GEMINI_API_KEY before using the Gemini benchmark.")
        self.model = model

    def generate(self, prompt: str, retries: int = 3) -> GeminiResult:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        response = None
        for attempt in range(retries + 1):
            response = requests.post(
                url,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json={"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0, "maxOutputTokens": 400}},
                timeout=60,
            )
            if response.status_code not in {429, 503} or attempt == retries:
                break
            time.sleep(2**attempt)
        assert response is not None
        if not response.ok:
            raise RuntimeError(f"Gemini request failed ({response.status_code}): {response.text[:500]}")
        payload = response.json()
        usage = payload.get("usageMetadata", {})
        candidates = payload.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = "".join(part.get("text", "") for part in parts)
        prompt_tokens = int(usage.get("promptTokenCount", 0))
        output_tokens = int(usage.get("candidatesTokenCount", 0))
        thought_tokens = int(usage.get("thoughtsTokenCount", 0))
        cached_tokens = int(usage.get("cachedContentTokenCount", 0))
        total = int(usage.get("totalTokenCount", prompt_tokens + output_tokens))
        return GeminiResult(text, prompt_tokens, output_tokens, thought_tokens, cached_tokens, total, self.model)

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote

import httpx

from portakal_app.data.errors import LLMConfigurationError, LLMRequestError, LLMResponseError
from portakal_app.data.models import AnalysisSuggestion, DatasetSummary
from portakal_app.models import LLMSessionConfig


SYSTEM_PROMPT = (
    "You are Portakal's AI data analysis assistant. "
    "Inspect the dataset summary and produce concise risks and suggestions for data quality, "
    "model readiness, suspicious schema patterns, missing data issues, leakage risks, and next-step actions. "
    "Return valid JSON only with keys 'risks' and 'suggestions'. "
    "Each list item must contain 'title', 'body', and 'severity'. "
    "Allowed severity values: low, medium, high. "
    "Do not wrap the JSON in markdown."
)
DEFAULT_LLM_TIMEOUT_SECONDS = 90.0


class LLMAnalyzer:
    def __init__(self, timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds

    def analyze(self, summary: DatasetSummary, context: str, config: LLMSessionConfig) -> list[AnalysisSuggestion]:
        _ = summary
        model = config.model.strip()
        base_url = config.base_url.strip() or config.default_base_url()
        if not model:
            raise LLMConfigurationError("Select a model before running AI analysis.")
        if not base_url:
            raise LLMConfigurationError("Base URL is required before running AI analysis.")

        if config.provider != "Ollama" and not config.resolved_api_key():
            raise LLMConfigurationError(f"{config.provider} API key is required before running AI analysis.")

        raw_text = self._request_analysis_text(context, config.with_updates(base_url=base_url))
        parsed = self._parse_json_payload(raw_text)
        return self._parse_suggestions(parsed)

    def _request_analysis_text(self, context: str, config: LLMSessionConfig) -> str:
        if config.provider in {"OpenAI", "Qwen"}:
            return self._request_openai_compatible(context, config)
        if config.provider == "Claude":
            return self._request_claude(context, config)
        if config.provider == "Gemini":
            return self._request_gemini(context, config)
        if config.provider == "Ollama":
            return self._request_ollama(context, config)
        raise LLMConfigurationError(f"Unsupported LLM provider: {config.provider}")

    def _request_openai_compatible(self, context: str, config: LLMSessionConfig) -> str:
        response = self._post_json(
            f"{config.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.resolved_api_key()}",
                "Content-Type": "application/json",
            },
            json_body={
                "model": config.model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
            },
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError("Invalid AI response.")
        message = choices[0].get("message", {})
        content = message.get("content")
        text = self._coerce_content_to_text(content)
        if not text:
            raise LLMResponseError("Invalid AI response.")
        return text

    def _request_claude(self, context: str, config: LLMSessionConfig) -> str:
        response = self._post_json(
            f"{config.base_url.rstrip('/')}/v1/messages",
            headers={
                "x-api-key": str(config.resolved_api_key()),
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json_body={
                "model": config.model,
                "max_tokens": 900,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": context}],
            },
        )
        content = response.get("content")
        if not isinstance(content, list) or not content:
            raise LLMResponseError("Invalid AI response.")
        text_chunks = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        text = "\n".join(chunk for chunk in text_chunks if chunk)
        if not text:
            raise LLMResponseError("Invalid AI response.")
        return text

    def _request_gemini(self, context: str, config: LLMSessionConfig) -> str:
        response = self._post_json(
            f"{config.base_url.rstrip('/')}/models/{quote(config.model, safe='')}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": config.resolved_api_key()},
            json_body={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": f"{SYSTEM_PROMPT}\n\nDataset summary:\n{context}",
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                },
            },
        )
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise LLMResponseError("Invalid AI response.")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not isinstance(parts, list) or not parts:
            raise LLMResponseError("Invalid AI response.")
        text = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text"))
        if not text:
            raise LLMResponseError("Invalid AI response.")
        return text

    def _request_ollama(self, context: str, config: LLMSessionConfig) -> str:
        response = self._post_json(
            f"{config.base_url.rstrip('/')}/api/chat",
            headers={"Content-Type": "application/json"},
            json_body={
                "model": config.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
            },
        )
        message = response.get("message", {})
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError("Invalid AI response.")
        return text

    def _post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, object],
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            response = httpx.post(
                url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise LLMRequestError("AI request timed out.") from exc
        except httpx.HTTPError as exc:
            raise LLMRequestError(f"AI request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMResponseError("LLM returned a non-JSON response.") from exc

        if response.status_code >= 400:
            raise LLMRequestError(self._error_message_from_payload(payload, response.status_code))
        return payload

    def _error_message_from_payload(self, payload: dict[str, Any], status_code: int) -> str:
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
        return f"AI request failed with status {status_code}."

    def _coerce_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif item.get("type") == "output_text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
            return "\n".join(part.strip() for part in parts if part and part.strip())
        return ""

    def _parse_json_payload(self, raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        if not text:
            raise LLMResponseError("Invalid AI response.")
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        if not text.startswith("{"):
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match is None:
                raise LLMResponseError("Invalid AI response.")
            text = match.group(0)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMResponseError("Invalid AI response.") from exc
        if not isinstance(payload, dict):
            raise LLMResponseError("Invalid AI response.")
        return payload

    def _parse_suggestions(self, payload: dict[str, Any]) -> list[AnalysisSuggestion]:
        results: list[AnalysisSuggestion] = []
        for kind in ("risks", "suggestions"):
            items = payload.get(kind)
            if items is None:
                continue
            if not isinstance(items, list):
                raise LLMResponseError("Invalid AI response.")
            for item in items:
                if not isinstance(item, dict):
                    raise LLMResponseError("Invalid AI response.")
                title = str(item.get("title", "")).strip()
                body = str(item.get("body", "")).strip()
                severity = self._normalize_severity(str(item.get("severity", "medium")).strip().lower())
                if not title or not body:
                    raise LLMResponseError("Invalid AI response.")
                results.append(AnalysisSuggestion(title=title, body=body, kind=kind[:-1], severity=severity))
        return results

    def _normalize_severity(self, severity: str) -> str:
        if severity in {"low", "medium", "high"}:
            return severity
        return "medium"

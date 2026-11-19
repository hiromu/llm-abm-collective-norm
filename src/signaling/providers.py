"""HTTP clients for OpenAI, DeepSeek and OpenRouter."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    effort: str
    max_tokens: int
    upstream: str | None = None
    quantization: str | None = None


MODELS = {
    "deepseek": ModelConfig("deepseek", "deepseek-flash", "high", 16384),
    "luna": ModelConfig("openai", "gpt-6-luna", "low", 8192),
    "oss": ModelConfig("openrouter", "openai/gpt-oss-120b", "low", 8192, "cerebras", "fp16"),
    "qwen": ModelConfig("openrouter", "qwen/qwen3.5-35b-a3b", "off", 8192, "deepinfra", "fp8"),
}
ENDPOINTS = {
    "openai": "https://api.openai.com/v1/responses",
    "deepseek": "https://api.deepseek.com/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}
KEYS = {"openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


class APIError(RuntimeError):
    def __init__(self, message, raw="", uncertain=True):
        super().__init__(message)
        self.raw, self.uncertain = raw, uncertain


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_body(config, system, observation):
    """Format applicant observations for individual or social learning."""
    user = json.dumps(observation, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if config.provider == "openai":
        return {"model": config.model, "instructions": system, "input": user,
                "reasoning": {"effort": "none" if config.effort == "off" else config.effort},
                "max_output_tokens": config.max_tokens, "text": {"format": {"type": "json_object"}},
                "store": False, "service_tier": "default"}
    body = {"model": config.model, "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}],
            "max_tokens": config.max_tokens, "response_format": {"type": "json_object"}, "stream": False}
    if config.provider == "deepseek":
        body["thinking"] = {"type": "disabled" if config.effort == "off" else "enabled"}
        if config.effort != "off":
            body["reasoning_effort"] = config.effort
    elif config.provider == "openrouter":
        body["reasoning"] = ({"enabled": False} if config.effort == "off"
                             else {"enabled": True, "effort": config.effort})
        body["provider"] = {"only": [config.upstream], "order": [config.upstream],
                            "allow_fallbacks": False, "quantizations": [config.quantization]}
    else:
        raise ValueError(f"Unknown provider: {config.provider}")
    return body


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def parse_decision(raw, provider):
    """Validate complete JSON decisions with technology use in [0, 1]."""
    response = json.loads(raw)
    if not isinstance(response, dict):
        raise ValueError("Expected a provider response object")
    if provider == "openai":
        if response.get("status") != "completed":
            raise ValueError("Incomplete OpenAI response")
        parts = [p["text"] for item in response.get("output", []) if item.get("type") == "message"
                 for p in item.get("content", []) if p.get("type") == "output_text"]
        content = "".join(parts)
    else:
        choices = response.get("choices", [])
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError("Incomplete chat completion")
        content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("No final JSON content")
    decision = json.loads(content, object_pairs_hook=unique_object)
    if not isinstance(decision, dict) or set(decision) != {"technology_use", "reason", "strategy"}:
        raise ValueError("Expected technology_use, reason and strategy")
    value = decision["technology_use"]
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("technology_use must be a finite number in [0,1]")
    if any(not isinstance(decision[k], str) or not decision[k].strip() for k in ("reason", "strategy")):
        raise ValueError("reason and strategy must be nonempty strings")
    return decision


class Client:
    """Send one request per call."""

    def __init__(self, config, env_file=Path(".env"), timeout=180):
        self.config, self.timeout = config, timeout
        name = KEYS[config.provider]
        self._key = os.environ.get(name)
        if not self._key and env_file and env_file.is_file():
            from dotenv import dotenv_values
            self._key = dotenv_values(env_file).get(name)
        if not self._key:
            raise ValueError(f"{name} is missing; set it in the environment or select --env-file")

    def send(self, body, request_id):
        request = Request(ENDPOINTS[self.config.provider],
                          data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode(),
                          headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                                   "X-Client-Request-Id": request_id}, method="POST")
        try:
            with build_opener(NoRedirect()).open(request, timeout=self.timeout) as response:
                return response.read().decode().replace(self._key, "[REDACTED]")
        except HTTPError as error:
            raw = error.read().decode(errors="replace").replace(self._key, "[REDACTED]")
            # Server failures and timeouts may have consumed billable work.
            raise APIError(f"{self.config.provider}: HTTP {error.code}", raw,
                           uncertain=error.code >= 500 or error.code in (408, 409)) from None
        except (URLError, TimeoutError, OSError) as error:
            raise APIError(f"{self.config.provider}: {type(error).__name__}; outcome unknown") from None

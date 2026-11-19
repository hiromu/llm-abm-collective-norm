"""Provider clients for common-pool resource experiments."""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import MODELS

ENDPOINTS = {"openai": "https://api.openai.com/v1/chat/completions",
             "groq": "https://api.groq.com/openai/v1/chat/completions",
             "anthropic": "https://api.anthropic.com/v1/messages"}
KEYS = {"openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def configuration(model, request_model=None, max_tokens=None):
    provider, identifier = MODELS[model]
    return {"provider": provider, "model": request_model or identifier,
            "temperature": .7 if provider == "groq" else None,
            "max_tokens": max_tokens if max_tokens is not None else (1024 if provider == "anthropic" else None)}


def request_body(config, messages):
    body = {"model": config["model"], "stream": False}
    if config["provider"] == "anthropic":
        body.update(system=messages[0]["content"], messages=messages[1:])
    else:
        body["messages"] = messages
    for field in ("temperature", "max_tokens"):
        if config[field] is not None:
            body[field] = config[field]
    return body


def content(raw, provider):
    response = json.loads(raw)
    if provider == "anthropic":
        if response.get("stop_reason") not in ("end_turn", "stop_sequence"):
            raise ValueError("Incomplete Anthropic response")
        value = "".join(c["text"] for c in response["content"] if c["type"] == "text")
    else:
        choices = response.get("choices", [])
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError("Incomplete chat response")
        value = choices[0]["message"]["content"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Empty model response")
    return value


class RequestError(RuntimeError):
    def __init__(self, message, raw="", uncertain=True):
        super().__init__(message)
        self.raw, self.uncertain = raw, uncertain


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Client:
    def __init__(self, config, env_file=Path(".env"), timeout=300):
        self.config, self.timeout = config, timeout
        name = KEYS[config["provider"]]
        self._key = os.environ.get(name)
        if not self._key and env_file and env_file.is_file():
            from dotenv import dotenv_values
            self._key = dotenv_values(env_file).get(name)
        if not self._key:
            raise ValueError(f"Missing {name}; select an existing --env-file or set the environment variable")

    def send(self, body):
        headers = {"Content-Type": "application/json"}
        provider = self.config["provider"]
        if provider == "anthropic":
            headers.update({"x-api-key": self._key, "anthropic-version": "2023-06-01"})
        else:
            headers["Authorization"] = "Bearer " + self._key
        request = Request(ENDPOINTS[provider], data=json.dumps(body, allow_nan=False).encode(),
                          headers=headers, method="POST")
        try:
            with build_opener(NoRedirect()).open(request, timeout=self.timeout) as response:
                return response.read().decode().replace(self._key, "[REDACTED]")
        except HTTPError as e:
            raw = e.read().decode(errors="replace").replace(self._key, "[REDACTED]")
            raise RequestError(f"{provider}: HTTP {e.code}", raw, e.code >= 500 or e.code in (408, 409)) from None
        except (URLError, TimeoutError, OSError) as e:
            raise RequestError(f"{provider}: {type(e).__name__}; request outcome unknown") from None

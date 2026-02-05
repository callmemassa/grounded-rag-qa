from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

from app.config import settings


# -----------------------------
# Output schema (strict)
# -----------------------------

class Citation(BaseModel):
    doc_id: str = Field(min_length=1)
    chunk_id: int = Field(ge=0)
    page: Optional[int] = Field(default=None, ge=1)


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1)
    citations: List[Citation] = Field(default_factory=list)


# -----------------------------
# Result container
# -----------------------------

@dataclass(frozen=True)
class GenerateResult:
    data: GeneratedAnswer
    model: str
    latency_ms: int
    usage: Optional[Dict[str, int]]


# -----------------------------
# Generator
# -----------------------------

class Generator:
    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        temperature: float = 0.2,
        max_output_tokens: int = 400,
        timeout_s: float = 30.0,
        retries: int = 2,
        logger=None,
    ):
        self.client = client
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.timeout_s = float(timeout_s)
        self.retries = max(0, int(retries))
        self.log = logger

    def _extract_text(self, resp: Any) -> str:
        txt = getattr(resp, "output_text", None)
        if txt:
            return str(txt).strip()

        return str(resp).strip()

    def _extract_usage(self, resp: Any) -> Optional[Dict[str, int]]:
        u = getattr(resp, "usage", None)
        if u is None:
            return None


        input_tokens = int(getattr(u, "input_tokens", 0) or 0)
        output_tokens = int(getattr(u, "output_tokens", 0) or 0)
        total_tokens = int(getattr(u, "total_tokens", 0) or (input_tokens + output_tokens))


        if input_tokens == 0 and output_tokens == 0 and total_tokens == 0:
            return None

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def _parse_json(self, text: str) -> GeneratedAnswer:
        obj = json.loads(text)
        return GeneratedAnswer.model_validate(obj)

    def generate(self, messages: List[Dict[str, Any]]) -> GenerateResult:
        t0 = time.perf_counter()
        last_err: Optional[Exception] = None

        for attempt in range(1, self.retries + 2):
            try:
                extra = ""
                if attempt > 1:
                    extra = (
                        "\n\nIMPORTANT: Return ONLY a valid JSON object. "
                        "No markdown, no prose, no code fences, no extra keys."
                    )
                    messages = [
                        messages[0],
                        {"role": "user", "content": (messages[1]["content"] + extra)},
                    ]

                resp = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    timeout=self.timeout_s,
                )

                raw = self._extract_text(resp)

                parsed = self._parse_json(raw)

                latency_ms = int((time.perf_counter() - t0) * 1000)
                usage = self._extract_usage(resp)

                if self.log:
                    self.log.info(
                        "GEN ok | model=%s | attempt=%s | latency_ms=%s | usage=%s",
                        self.model, attempt, latency_ms, usage
                    )

                return GenerateResult(
                    data=parsed,
                    model=self.model,
                    latency_ms=latency_ms,
                    usage=usage,
                )

            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
                if self.log:
                    self.log.info(
                        "GEN invalid json | attempt=%s | err=%s",
                        attempt, f"{type(e).__name__}: {e}"
                    )

                if attempt <= self.retries:
                    time.sleep(0.4 * attempt)
                    continue
                break

            except Exception as e:
                last_err = e
                if self.log:
                    self.log.info(
                        "GEN error | attempt=%s | err=%s",
                        attempt, f"{type(e).__name__}: {e}"
                    )
                if attempt <= self.retries:
                    time.sleep(0.5 * attempt)
                    continue
                break


        raise RuntimeError(f"LLM generation failed after retries: {last_err}") from last_err


def build_generator(client: OpenAI, logger=None) -> Generator:
    return Generator(
        client=client,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        timeout_s=settings.LLM_TIMEOUT_S,
        retries=settings.LLM_RETRIES,
        logger=logger,
    )
from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main_mod
from app.schemas import AskResponse


def test_ask_ok_with_mocked_pipeline(monkeypatch):
    def fake_answer(q: str) -> AskResponse:
        return AskResponse(
            ok=True,
            answer="mock answer",
            sources=[],
            latency_ms=1,
            usage=None,
            cost_usd=0.0,
            request_id="will_be_overwritten",
        )

    monkeypatch.setattr(main_mod, "rag_answer", fake_answer, raising=True)

    client = TestClient(main_mod.app)

    resp = client.post("/ask", json={"question": "text fonts requirements"})
    assert resp.status_code == 200

    data = resp.json()
    assert data["ok"] is True
    assert data["answer"] == "mock answer"
    assert "request_id" in data


def test_ask_422_on_invalid_body():
    client = TestClient(main_mod.app)

    resp = client.post("/ask", json={})
    assert resp.status_code == 422
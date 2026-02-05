from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.rag.pipeline import answer as rag_answer


CASES_PATH_DEFAULT = Path("eval/cases.jsonl")

REFUSAL_TEXT = "I don't know based on the provided documents."


@dataclass
class Case:
    id: str
    question: str
    expectation: str  # "should_answer" | "should_refuse"
    must_include: List[str]
    keywords: List[str]


def load_cases(path: Path) -> List[Case]:
    cases: List[Case] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            cid = str(obj.get("id") or f"case_{line_no}")
            q = str(obj.get("question") or "").strip()
            exp = str(obj.get("expectation") or "").strip()
            must = obj.get("must_include") or []
            kws = obj.get("keywords") or []

            if not q:
                raise ValueError(f"Empty question in {path}:{line_no}")
            if exp not in {"should_answer", "should_refuse"}:
                raise ValueError(f"Bad expectation in {path}:{line_no}: {exp}")

            if not isinstance(must, list):
                raise ValueError(f"must_include must be list in {path}:{line_no}")
            if not isinstance(kws, list):
                raise ValueError(f"keywords must be list in {path}:{line_no}")

            cases.append(
                Case(
                    id=cid,
                    question=q,
                    expectation=exp,
                    must_include=[str(x) for x in must],
                    keywords=[str(x) for x in kws],
                )
            )
    return cases


def _sources_doc_ids(res: Any) -> List[str]:
    out: List[str] = []
    for s in (getattr(res, "sources", None) or []):
        doc_id = getattr(s, "doc_id", None)
        if doc_id:
            out.append(str(doc_id))
    return out


def _sources_count(res: Any) -> int:
    return len(getattr(res, "sources", None) or [])


def _has_must_include(res: Any, must_include: List[str]) -> bool:
    if not must_include:
        return True
    doc_ids = set(_sources_doc_ids(res))
    return any(m in doc_ids for m in must_include)


def _answer_contains_keywords(res: Any, keywords: List[str]) -> bool:
    if not keywords:
        return True
    ans = (getattr(res, "answer", "") or "").lower()
    return all(k.lower() in ans for k in keywords)


def _is_refusal(res: Any) -> bool:
    ans = (getattr(res, "answer", "") or "").strip()
    return ans == REFUSAL_TEXT


def _fmt_bool(x: bool) -> str:
    return "OK" if x else "FAIL"


def main() -> int:
    p = argparse.ArgumentParser(description="RAG Eval Runner")
    p.add_argument("--cases", type=str, default=str(CASES_PATH_DEFAULT), help="Path to eval/cases.jsonl")
    p.add_argument("--print_failures_only", action="store_true", help="Print only failed cases")
    args = p.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        raise RuntimeError(f"Missing cases file: {cases_path}")

    cases = load_cases(cases_path)

    # Metrics counters
    total = len(cases)
    should_answer_total = 0
    should_refuse_total = 0

    hit_k_ok = 0
    grounded_ok = 0
    refusal_ok = 0

    empty_outputs = 0

    latencies: List[int] = []
    costs: List[float] = []

    print("\nEVAL RUN")
    print("--------")
    print(f"cases: {total}\n")

    for c in cases:
        res = None
        err: Optional[str] = None

        try:
            res = rag_answer(c.question)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        # If infra died -> count as fail (but still print)
        if err is not None:
            if not args.print_failures_only:
                print(f"[{c.id}] {c.expectation} | ERROR | {err}")
            continue

        # Collect latency / cost if present
        lat = getattr(res, "latency_ms", None)
        if isinstance(lat, int):
            latencies.append(lat)

        cost = getattr(res, "cost_usd", None)
        if isinstance(cost, (int, float)):
            costs.append(float(cost))

        sources_n = _sources_count(res)
        is_refusal = _is_refusal(res)
        has_must = _has_must_include(res, c.must_include)
        kw_ok = _answer_contains_keywords(res, c.keywords)

        case_ok = True
        extra_notes: List[str] = []

        if c.expectation == "should_answer":
            should_answer_total += 1

            # grounded_rate: must have sources
            grounded = sources_n > 0
            if grounded:
                grounded_ok += 1
            else:
                case_ok = False
                extra_notes.append("no_sources")
                empty_outputs += 1

            # should not refuse
            if is_refusal:
                case_ok = False
                extra_notes.append("refused")
                empty_outputs += 1

            # hit@k: must_include doc_id appears in sources (if provided)
            if c.must_include:
                if has_must:
                    hit_k_ok += 1
                else:
                    case_ok = False
                    extra_notes.append("missed_must_include")

            # optional keyword check (answer must contain all keywords)
            if c.keywords:
                if not kw_ok:
                    case_ok = False
                    extra_notes.append("missing_keywords")

        else:
            should_refuse_total += 1


            if is_refusal and sources_n == 0:
                refusal_ok += 1
            else:
                case_ok = False
                if not is_refusal:
                    extra_notes.append("did_not_refuse")
                if sources_n != 0:
                    extra_notes.append("sources_not_empty")

        if (not args.print_failures_only) or (not case_ok):
            notes = (", ".join(extra_notes) if extra_notes else "-")
            print(
                f"[{c.id}] exp={c.expectation} | {_fmt_bool(case_ok)} | "
                f"sources={sources_n} refusal={is_refusal} must={has_must} kw={kw_ok} | {notes}"
            )

    def _rate(x: int, d: int) -> float:
        return (x / d) if d > 0 else 0.0

    hit_at_k = _rate(hit_k_ok, should_answer_total)
    grounded_rate = _rate(grounded_ok, should_answer_total)
    refusal_quality = _rate(refusal_ok, should_refuse_total)
    empty_rate = _rate(empty_outputs, should_answer_total)

    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else None
    avg_cost = round(sum(costs) / len(costs), 6) if costs else None

    print("\nSUMMARY")
    print("-------")
    print(f"should_answer: {should_answer_total}")
    print(f"should_refuse: {should_refuse_total}")
    print(f"hit@k: {hit_k_ok}/{should_answer_total} = {hit_at_k:.2f}")
    print(f"grounded_rate: {grounded_ok}/{should_answer_total} = {grounded_rate:.2f}")
    print(f"refusal_quality: {refusal_ok}/{should_refuse_total} = {refusal_quality:.2f}")
    print(f"empty_rate (answer-cases): {empty_outputs}/{should_answer_total} = {empty_rate:.2f}")

    if avg_latency is not None:
        print(f"avg_latency_ms: {avg_latency}")
    if avg_cost is not None:
        print(f"avg_cost_usd: {avg_cost}")

    print("\nOK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
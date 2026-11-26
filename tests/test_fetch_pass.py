"""scripts/ops/fetch_pass.py: summary + request-rate arithmetic. Hermetic: fake HTTP, fake clock."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))
import fetch_pass as fp  # noqa: E402
import refresh_feed as rf  # noqa: E402


def test_request_rate_by_domain_uses_the_smallest_start_to_start_gap() -> None:
    log = [
        {"t": 0.0, "domain": "a.example"},
        {"t": 1.0, "domain": "a.example"},
        {"t": 2.5, "domain": "a.example"},
        {"t": 0.2, "domain": "b.example"},
    ]
    rates = fp.request_rate_by_domain(log)
    assert rates["a.example"] == {"requests": 3, "min_gap_s": 1.0, "max_rate_per_s": 1.0}
    assert rates["b.example"]["requests"] == 1  # one request: no gap to measure


def test_run_summarises_one_store_saves_raw_pages_and_respects_the_limiter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [0.0]

    def sleep(s: float) -> None:
        clock[0] += s

    monkeypatch.setattr(rf, "GLOBAL_THROTTLE", rf.DomainThrottle(clock=lambda: clock[0], sleep=sleep))
    monkeypatch.setattr(fp.time, "monotonic", lambda: clock[0])
    page = json.dumps({"products": [{"id": 1}, {"id": 2}]})

    def fake_get(url: str, headers: dict[str, str], timeout: float) -> rf.HttpResponse:
        if url.endswith("/robots.txt"):
            return rf.HttpResponse(200, {}, "User-agent: *\nAllow: /\n")
        return rf.HttpResponse(200, {}, page)

    summary = fp.run(["shopkop"], tmp_path, fake_get)
    row = summary["stores"][0]
    assert row["status"] == rf.OK and row["items"] == 2 and not row["capped"]
    assert summary["requests_total"] == 2 and summary["max_rate_per_s_any_domain"] <= 1.0
    saved = tmp_path / "shopkop__p0001.json.gz"
    assert json.loads(gzip.decompress(saved.read_bytes()))["products"][0]["id"] == 1
    assert "| shopkop | ok |" in fp.to_markdown({k: v for k, v in summary.items() if k != "_log"})

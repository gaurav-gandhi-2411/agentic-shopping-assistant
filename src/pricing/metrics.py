"""In-process metrics for the live price check, in Prometheus text-exposition shape.

The repo has no ``prometheus_client`` dependency and no ``/metrics`` endpoint yet, so this is a
tiny thread-safe registry that uses Prometheus naming (``asa_*``, ``_total`` counters,
``_seconds`` histogram with cumulative ``le`` buckets). When a scrape endpoint is added it can
serve :func:`render_prometheus`; swapping in ``prometheus_client`` later is a one-file change.
The same numbers are also emitted as structured log fields by ``live_check``.
"""

from __future__ import annotations

import threading
from collections import defaultdict

# Batch latency buckets (seconds); the check budget is 1.5 s so 1.5 is an explicit edge.
LATENCY_BUCKETS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5)

_lock = threading.Lock()
_counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(
    lambda: defaultdict(float)
)
_hist_counts: list[int] = [0] * (len(LATENCY_BUCKETS) + 1)  # last slot is +Inf
_hist_sum: float = 0.0

CHECKS = "asa_live_price_checks_total"  # label outcome: ok|cache_hit|timeout|error|...
PRICE_CHANGED = "asa_live_price_changed_total"
UNAVAILABLE = "asa_live_price_unavailable_total"
DURATION = "asa_live_price_check_duration_seconds"


def inc(name: str, amount: float = 1.0, **labels: str) -> None:
    """Increment counter ``name`` for the given label set."""
    with _lock:
        _counters[name][tuple(sorted(labels.items()))] += amount


def observe_duration(seconds: float) -> None:
    """Record one whole-batch check duration in the histogram."""
    global _hist_sum
    with _lock:
        _hist_sum += seconds
        for i, edge in enumerate(LATENCY_BUCKETS):
            if seconds <= edge:
                _hist_counts[i] += 1
                return
        _hist_counts[-1] += 1


def counter_value(name: str, **labels: str) -> float:
    """Current value of one counter series (0.0 when never incremented)."""
    with _lock:
        return _counters[name].get(tuple(sorted(labels.items())), 0.0)


def reset() -> None:
    """Clear all series (tests only)."""
    global _hist_sum
    with _lock:
        _counters.clear()
        _hist_counts[:] = [0] * (len(LATENCY_BUCKETS) + 1)
        _hist_sum = 0.0


def render_prometheus() -> str:
    """Render every series in Prometheus text format."""
    lines: list[str] = []
    with _lock:
        for name in sorted(_counters):
            lines.append(f"# TYPE {name} counter")
            for labels, value in sorted(_counters[name].items()):
                lab = ",".join(f'{k}="{v}"' for k, v in labels)
                lines.append(f"{name}{{{lab}}} {value:g}" if lab else f"{name} {value:g}")
        lines.append(f"# TYPE {DURATION} histogram")
        running = 0
        for edge, n in zip(LATENCY_BUCKETS, _hist_counts, strict=False):
            running += n
            lines.append(f'{DURATION}_bucket{{le="{edge:g}"}} {running}')
        running += _hist_counts[-1]
        lines.append(f'{DURATION}_bucket{{le="+Inf"}} {running}')
        lines.append(f"{DURATION}_sum {_hist_sum:g}")
        lines.append(f"{DURATION}_count {running}")
    return "\n".join(lines) + "\n"

"""Remove store identity from the text judges see (titles and descriptions).

The judge prompt already omits the `store` field, but product titles and descriptions carry the
brand ("VASTRAMAY Men's ... Bundle", "Libas Green Kurta"), which would let a judge infer the store
and bring brand priors into a relevance label. Every store slug, display name and domain label in
the store registry (public and non-public) is scrubbed from title and description text before
prompting. Detection (`leaks`) is deliberately independent of the scrubber: it normalises to
lowercase alphanumerics and does a substring test, so it also catches glued or re-spaced forms.

    python -m evals.retrieval.scrub        # leak rate on the frozen candidates, before / after
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Iterable, Mapping

from src.config.store_registry import STORE_REGISTRY
from src.config.stores import STORE_CONFIG

_ALNUM = re.compile(r"[^a-z0-9]+")
# Terms shorter than this are not matched by the substring detector (false-positive guard);
# every registry term is at least 4 characters once normalised ("hm" is covered by "h&m" below).
_MIN_DETECT = 4


def _norm(text: str) -> str:
    return _ALNUM.sub("", text.lower())


@lru_cache(maxsize=1)
def store_terms() -> tuple[str, ...]:
    """All distinct surface forms of every registry store: slug, display name, domain label,
    each with and without spaces/hyphens. Longest first so "Global Republic" beats "Global"."""
    forms: set[str] = set()
    for slug, rec in STORE_REGISTRY.items():
        names = [slug, rec.display_name]
        if rec.domain:
            host = rec.domain.removeprefix("www.")
            names.append(host.split(".")[0])
        cfg = STORE_CONFIG.get(slug) or {}
        if cfg.get("display_name"):
            names.append(str(cfg["display_name"]))
        for n in names:
            n = n.strip()
            if not n:
                continue
            forms.add(n)
            forms.add(re.sub(r"[-_\s]+", " ", n))
            forms.add(re.sub(r"[-_\s]+", "", n))
    return tuple(sorted(forms, key=lambda s: (-len(s), s)))


@lru_cache(maxsize=1)
def _pattern() -> re.Pattern[str]:
    # Terms may be glued to letters on neither side ("Libas" in "Libaskurta" is not a store
    # mention); spaces inside a term match any run of space/hyphen.
    alts = []
    for t in store_terms():
        alts.append(re.sub(r"(?:\\ |\\-)+", r"[-\\s_]*", re.escape(t)))
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(alts) + r")(?:'s|’s)?(?![A-Za-z0-9])", re.I)


def scrub_stores(text: str) -> str:
    """`text` with every registry store mention removed and whitespace collapsed."""
    out = _pattern().sub(" ", text or "")
    out = re.sub(r"\(\s*\)", " ", out)  # "()" left by a bracketed brand
    return re.sub(r"\s+", " ", out).strip(" -|,")  # no other punctuation edits: text stays as-is


@lru_cache(maxsize=1)
def _detect_terms() -> tuple[str, ...]:
    return tuple(sorted({n for t in store_terms() if len(n := _norm(t)) >= _MIN_DETECT}))


def leaks(text: str) -> list[str]:
    """Normalised store terms still present in `text` (empty = clean). Independent of scrub."""
    norm = _norm(text)
    return [t for t in _detect_terms() if t in norm]


def leak_rate(lines: Iterable[str]) -> tuple[int, int]:
    """(lines with at least one leak, total lines)."""
    n = bad = 0
    for ln in lines:
        n += 1
        bad += bool(leaks(ln))
    return bad, n


def judge_lines(items: Iterable[Mapping[str, Any]], scrub: bool) -> list[str]:
    """The title + description text of each candidate as a judge would be shown it."""
    from evals.retrieval.judge import SNIPPET_CHARS

    out = []
    for c in items:
        title = c.get("display_name") or c.get("title") or ""
        desc = " ".join((c.get("detail_desc") or c.get("snippet") or "").split())
        if scrub:  # same order as judge._fmt_candidate: scrub, then truncate
            title, desc = scrub_stores(title), scrub_stores(desc)
        out.append(f"{title} | {desc[:SNIPPET_CHARS]}")
    return out


def main() -> int:
    from evals.retrieval.common import CANDIDATE_DIR, load_candidates, load_cases

    items = [it for c in load_cases() for it in load_candidates(c["id"], CANDIDATE_DIR)["items"]]
    res = {
        mode: dict(zip(("leaking_lines", "lines"), leak_rate(judge_lines(items, mode == "after"))))
        for mode in ("before", "after")
    }
    for r in res.values():
        r["leak_rate"] = round(r["leaking_lines"] / r["lines"], 4) if r["lines"] else None
    print(json.dumps(res, indent=1))
    return 0 if res["after"]["leaking_lines"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

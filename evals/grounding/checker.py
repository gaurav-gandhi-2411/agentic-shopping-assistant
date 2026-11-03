"""Deterministic grounding checker for assistant text (Wave 1 item W10, E5).

    check(assistant_text, shown_items, user_text="", extra_amounts=()) -> list[Violation]

Pure function, no I/O, no LLM. Every price / size / stock / discount / delivery claim in
`assistant_text` must be backed by a field of an item that was actually SHOWN (card data), or be
one of two narrow exemptions (below). Anything else is a Violation.

Item fields consulted (all optional; a missing field means "no data", so a claim that needs it is
a violation):
    price_inr, display_name / prod_name / detail_desc (name + description text),
    sizes (list[str]), in_stock (bool), stock_count (int), discount_pct (float), mrp_inr (float),
    delivery_days (int | [min, max]), free_shipping (bool).
Today's catalogue carries only price_inr (no sizes, stock, discount or delivery fields), so any
size/stock/discount/delivery claim is a violation today. The fields are honoured so that W11d
(live price/availability) can back claims without changing this checker.

Exemptions (both narrow, both tested):
  1. BUDGET ECHO: a money amount equal (+-1 INR) to an amount the USER stated, in a budget-context
     phrase ("under X", "within X", "your X budget", "between X and Y"). Never covers an amount
     attributed to an item ("the Riviera at X" is still checked against the item's price).
  2. DENIAL ECHO: a claim inside a sentence that disclaims ("I don't have ...", "can't confirm
     ...", "check the product page") is allowed when the claim keyword is bare (no number), or
     when its number/term appears in the user's own text. An invented number inside a denial
     ("we don't have the 40% off deal") is still a violation.

Known limits (see tests/test_grounding_checker.py): prose number words beyond a small vocabulary,
claims split across sentences, and implicit price statements ("pricey") are not detected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

PRICE_TOL = 1.0  # INR; spec: "price within Rs 1"

# --------------------------------------------------------------------------- violations


@dataclass(frozen=True)
class Violation:
    """One ungrounded claim. `kind` is the category used in reports."""

    kind: str  # price | price_wrong_item | percent | discount | size | stock | delivery
    text: str  # the offending span
    start: int
    end: int
    reason: str
    sentence: str = ""


@dataclass
class _Claim:
    kind: str
    start: int
    end: int
    value: float | str | None = None  # amount (INR), percent, or normalised size / stock token
    flavour: str = ""  # extra tag, e.g. "oos", "in", "count", "material"
    budget_ctx: bool = False


# --------------------------------------------------------------------------- number parsing

_NUM = r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_MULT = r"(?:\s?(?:k\b|thousand\b|lakhs?\b|lacs?\b|crores?\b|cr\b|hundred\b))?"
_CUR = r"(?:₹|\bRs\.?|\bINR\b|\bUSD\b|\bRupees?\b|[$€£])"
_CUR_SUFFIX = r"(?:/-|\s*rupees?\b|\s*rs\b\.?|\s*inr\b|\s*bucks\b)"
_MULT_VAL = {
    "k": 1e3, "thousand": 1e3, "hundred": 1e2, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5,
    "lacs": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7,
}
_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}

_PRICE_CUES = (
    r"(?:priced?(?:\s+(?:at|around|about|of|is))?|costs?|costing|worth|pay(?:ing)?|mrp|"
    r"under|below|above|within|upto|up\s+to|budget(?:\s+of)?|around|about|approx(?:imately)?\.?|"
    r"starting\s+(?:at|from)|starts\s+(?:at|from)|less\s+than|more\s+than|at\s+most|at\s+least|"
    r"max(?:imum)?|between|only|just|for|at|from|over)"
)
# Cue words after which an amount is a *constraint* (user-style budget) rather than a price.
_BUDGET_WORDS = re.compile(
    r"\b(?:under|below|within|upto|up\s+to|budget|less\s+than|more\s+than|at\s+most|at\s+least|"
    r"max(?:imum)?|between|limit|cap|ceiling|range|above|over|around|about|approx(?:imately)?)\b",
    re.I,
)
_UNIT_AFTER = re.compile(
    r"^\s*(?:days?|hours?|hrs?|weeks?|months?|years?|minutes?|mins?|cm|mm|inch(?:es)?|in\b|"
    r"meters?|metres?|m\b|kg|gm?s?\b|grams?|pieces?|pcs|items?|options?|colou?rs?|stores?|"
    r"looks?|sizes?|%|percent|per\s*cent|ml|sets?|styles?|results?|products?|pairs?)",
    re.I,
)

_RE_PRE = re.compile(rf"(?P<cur>{_CUR})\s*(?P<num>{_NUM})(?P<mult>{_MULT})", re.I)
_RE_SUF = re.compile(rf"(?<![\w.,])(?P<num>{_NUM})(?P<mult>{_MULT})(?P<suf>{_CUR_SUFFIX})", re.I)
_RE_BARE = re.compile(
    rf"(?<![\w₹$€£])(?P<cue>{_PRICE_CUES})\s*(?:of\s+|is\s+|:\s*)?(?P<num>{_NUM})(?P<mult>{_MULT})"
    r"(?![\w,])",
    re.I,
)
_RE_WORDS = re.compile(
    r"\b(?P<w>(?:" + "|".join(_WORD_NUM) + r"))(?:\s+|-)?(?P<m>hundred|thousand|lakh|lac)\b"
    r"(?:\s+(?:and\s+)?(?P<w2>(?:" + "|".join(_WORD_NUM) + r")))?"
    r"\s*(?:rupees?|rs\b|inr\b|bucks)",
    re.I,
)
_RE_RANGE = re.compile(
    rf"\s*(?P<sep>-|–|—|to|and)\s*(?:{_CUR}\s*)?(?P<num>{_NUM})(?P<mult>{_MULT})"
    rf"(?:{_CUR_SUFFIX})?",
    re.I,
)
_RE_PCT = re.compile(r"(?<![\w.])(?P<n>\d+(?:\.\d+)?)\s*(?:%|percent\b|per\s*cent\b)", re.I)

_FABRIC = (
    r"(?:cotton|polyester|viscose|rayon|silk|linen|wool|nylon|spandex|elastane|acrylic|"
    r"lycra|modal|denim|georgette|chiffon|blend|poly|leather|satin|velvet|crepe)"
)


def _to_float(num: str, mult: str | None) -> float:
    v = float(num.replace(",", ""))
    m = (mult or "").strip().lower()
    return v * _MULT_VAL[m] if m in _MULT_VAL else v


def _is_id_like(num: str) -> bool:
    """Unformatted 7+ digit runs are article/SKU ids, never a bare price."""
    return "," not in num and "." not in num and len(num) >= 7


def _is_year(num: str) -> bool:
    return bool(re.fullmatch(r"(?:19[89]\d|20[0-3]\d)", num))


# --------------------------------------------------------------------------- claim extraction


def _money_claims(text: str) -> list[_Claim]:
    """All INR-style amounts (with ranges), as claims with value in rupees."""
    found: list[_Claim] = []
    taken: list[tuple[int, int]] = []

    def free(s: int, e: int) -> bool:
        return not any(s < te and e > ts for ts, te in taken)

    def add(s: int, e: int, val: float, budget: bool = False) -> None:
        taken.append((s, e))
        found.append(_Claim("price", s, e, val, budget_ctx=budget))

    def with_range(m: re.Match[str], val: float, bare_ok: bool) -> None:
        s, e = m.start(), m.end()
        add(s, e, val)
        r = _RE_RANGE.match(text, e)
        if r and (r.group("sep") != "and" or re.search(r"between\W*$", text[max(0, s - 24):s], re.I)):
            n2 = r.group("num")
            tail_has_cur = bool(re.search(_CUR, r.group(0), re.I) or re.search(_CUR_SUFFIX, r.group(0), re.I))
            rest = text[r.end():r.end() + 12]
            if (bare_ok or tail_has_cur) and not _UNIT_AFTER.match(rest):
                add(r.start("num"), r.end(), _to_float(n2, r.group("mult")))

    for m in _RE_PRE.finditer(text):
        if free(m.start(), m.end()):
            with_range(m, _to_float(m.group("num"), m.group("mult")), bare_ok=True)
    for m in _RE_SUF.finditer(text):
        if free(m.start(), m.end()) and not _UNIT_AFTER.match(text[m.end():m.end() + 10]):
            with_range(m, _to_float(m.group("num"), m.group("mult")), bare_ok=True)
    for m in _RE_BARE.finditer(text):
        num = m.group("num")
        s = m.start("num")
        if not free(s, m.end()) or _UNIT_AFTER.match(text[m.end():m.end() + 10]):
            continue
        val = _to_float(num, m.group("mult"))
        cue = m.group("cue").lower()
        if val < 100 or _is_id_like(num) or (_is_year(num) and cue.split()[0] in {"from", "over"}):
            continue
        if re.match(r"\s*(?:%|percent)", text[m.end():m.end() + 10], re.I):
            continue
        add(s, m.end(), val)
        r = _RE_RANGE.match(text, m.end())
        if r and (r.group("sep") != "and" or cue == "between"):
            v2 = _to_float(r.group("num"), r.group("mult"))
            if v2 >= 100 and not _UNIT_AFTER.match(text[r.end():r.end() + 12]):
                add(r.start("num"), r.end(), v2)
    for m in _RE_WORDS.finditer(text):
        if free(m.start(), m.end()):
            base = _WORD_NUM[m.group("w").lower()] * _MULT_VAL[m.group("m").lower().rstrip("s")]
            if m.group("w2"):
                base += _WORD_NUM[m.group("w2").lower()]
            add(m.start(), m.end(), float(base))
    found.sort(key=lambda c: c.start)
    return found


_SIZE_LETTERS = r"(?:XXXL|XXL|XL|XXS|XS|2XL|3XL|4XL|5XL|S|M|L)"
_RE_SIZE_CUE = re.compile(
    rf"\bsizes?\s*(?:no\.?|number|chart)?\s*[:\-]?\s*(?:is\s+|of\s+|are\s+)?"
    rf"(?P<list>(?:{_SIZE_LETTERS}|\d{{1,2}}(?:\.\d)?)(?:\s*(?:,|/|-|–|to|and|or|&)\s*"
    rf"(?:{_SIZE_LETTERS}|\d{{1,2}}(?:\.\d)?))*)\b",
    re.I,
)
_RE_SIZE_REGION = re.compile(r"\b(?:UK|US|EU|IN|IND)[\s\-]?(?P<n>\d{1,2}(?:\.\d)?)\b")
_RE_SIZE_IN = re.compile(
    rf"\b(?:in|fits?|wear(?:s|ing)?|available\s+in|comes\s+in|go\s+for|order)\s+(?:a\s+)?"
    rf"(?P<s>{_SIZE_LETTERS}(?:\s*(?:,|/|and|or|&)\s*{_SIZE_LETTERS})*)\b(?!['’.]?\w)"
)
_RE_SIZE_MULTI = re.compile(r"(?<![\w-])(?P<s>XXXL|XXL|XL|XXS|XS|[2-5]XL)(?![\w-])", re.I)
_RE_SIZE_FREE = re.compile(r"\b(?:free|one|universal)[\s\-]?size\b", re.I)
_RE_FIT = re.compile(
    r"\b(?:runs?\s+(?:small|large|big|tight|true)|true\s+to\s+size|fits?\s+true|size\s+(?:up|down)|"
    r"order\s+a\s+size|tts)\b",
    re.I,
)
_RE_STOCK = re.compile(
    r"(?P<oos>\bout[\s\-]of[\s\-]stock\b|\bsold[\s\-]out\b|\bunavailable\b|\bnot\s+available\b|"
    r"\bno\s+longer\s+available\b|\bcurrently\s+unavailable\b)"
    r"|(?P<ins>\bin[\s\-]stock\b|\bback\s+in\s+stock\b|\brestock(?:ed|ing)?\b|"
    r"\bready\s+to\s+ship\b|\bcurrently\s+available\b|\bavailable\s+(?:now|right\s+now|today)\b|"
    r"\bstock\s+(?:is\s+)?(?:low|limited|available)\b|\blow\s+stock\b|\blimited\s+stock\b|"
    r"\bselling\s+fast\b|\balmost\s+(?:gone|sold)\b|\bfew\s+(?:pieces?\s+)?left\b|"
    r"\blast\s+few\b|\bhurry\b)"
    r"|(?P<cnt>\b(?:only\s+)?(?<![\w.])(?P<n>\d+)\s+(?:pieces?|pcs|units?|items?|sets?|pairs?)?\s*"
    r"(?:left|remaining|in\s+stock|available)\b|\blast\s+(?P<n2>\d+)\b)",
    re.I,
)
_RE_DISCOUNT = re.compile(
    r"\b(?:discount(?:s|ed)?|on\s+sale|sale\s+price|sale|clearance|markdowns?|coupons?|"
    r"promo(?:\s+code)?|cash\s*back|bogo|buy\s+one\s+get\s+one|price\s+(?:drop|cut)|"
    r"reduced\s+price|deals?|bargains?|offers?\s+(?:on|of)|savings?|save\s+(?:up|big|money)|"
    r"flat\s+\d+|special\s+offer|limited[\s\-]time\s+offer|\d+\s*%?\s*off|"
    r"buy\s+\d+\s+get\s+\d+(?:\s+free)?|b\dg\d)\b"
    r"|(?:save|saves|saving)\s+(?:up\s+to\s+)?(?:rs\.?|₹)\s*\d[\d,]*"
    r"|(?:rs\.?|₹)\s*\d[\d,]*\s*off\b",
    re.I,
)
_RE_DELIVERY = re.compile(
    r"\b(?:free\s+(?:shipping|delivery|returns?|exchange|cod)|complimentary\s+(?:shipping|delivery)|"
    r"cash\s+on\s+delivery|cod\b|deliver(?:y|s|ed|ing)?|ship(?:s|ped|ping|ment)?|dispatch(?:ed|es)?|"
    r"arriv(?:e|es|ing|al)|express|next[\s\-]day|same[\s\-]day|return\s+policy|"
    r"\d+[\s\-]days?\s+returns?|easy\s+returns?|pan[\s\-]india)\b",
    re.I,
)
_RE_DAYS = re.compile(
    r"(?<![\w.])(?P<a>\d+)(?:\s*(?:-|–|to)\s*(?P<b>\d+))?\s*(?:business\s+|working\s+)?"
    r"(?P<u>days?|hours?|hrs?|weeks?)\b",
    re.I,
)

# Words that, in a sentence, make a delivery keyword a delivery *claim* for day counts.
_DELIVERY_CTX = re.compile(r"\b(?:deliver|ship|dispatch|arriv|return|exchange|refund)", re.I)

_DENIAL = re.compile(
    r"\b(?:i\s+(?:do\s*n[o']t|don['’]t|do\s+not|can[’']?t|cannot|can\s+not|couldn['’]t|"
    r"could\s+not|am\s+(?:not\s+able|unable)|have\s+no|lack)|"
    r"(?:do\s*n[o']t|don['’]t|does\s*n[o']t|doesn['’]t|did\s*n[o']t|didn['’]t)\s+"
    r"(?:have|list|show|carry|include|know|track|provide|offer|display|filter)|"
    r"no\s+(?:data|info|information|details|way\s+to)|unable\s+to|not\s+able\s+to|"
    r"(?:isn['’]t|aren['’]t|is\s+not|are\s+not|wasn['’]t)\s+(?:listed|shown|available\s+in\s+the\s+data|"
    r"provided|something\s+i)|not\s+(?:listed|shown)|"
    r"check\s+(?:the\s+)?(?:product|store|brand|retailer|seller|pdp|site|website|page|link)|"
    r"(?:confirm|verify)\s+(?:on|with|at)\s+the)\b",
    re.I,
)


_CLAUSE_SPLIT = re.compile(r"\b(?:but|however|though|although|yet|whereas)\b|;", re.I)


def _clause_denials(sent: str) -> list[tuple[int, int, bool]]:
    """(start, end, has_denial_cue) for each clause of `sent`.

    Clauses are split on but/however/though/yet/;  so "I can't see stock, but it says in stock"
    does not let the second clause borrow the first clause's disclaimer. Computed once per
    sentence (it used to run per claim, which was quadratic on claim-dense input).
    """
    out: list[tuple[int, int, bool]] = []
    lo = 0
    for m in _CLAUSE_SPLIT.finditer(sent):
        out.append((lo, m.start(), bool(_DENIAL.search(sent[lo:m.start()]))))
        lo = m.end()
    out.append((lo, len(sent) + 1, bool(_DENIAL.search(sent[lo:]))))
    return out


def _in_denial(clauses: list[tuple[int, int, bool]], pos: int) -> bool:
    """Whether offset `pos` sits in a clause that carries a denial cue."""
    return any(lo <= pos < hi and den for lo, hi, den in clauses)


def _sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans. Splits on [.!?] before a capital/quote and on newlines.

    Does not split on "Rs. 1,499" or decimals (the next char after the dot is a digit/space+digit).
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"[.!?]+(?=\s+[A-Z\"“*_(\[])|\n+", text):
        if text[: m.start()].rstrip().lower().endswith(("rs", "no", "approx")):
            continue
        spans.append((start, m.end()))
        start = m.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _extract_claims(text: str) -> list[_Claim]:
    """Every price / percent / size / stock / discount / delivery claim in `text`."""
    claims: list[_Claim] = _money_claims(text)
    price_spans = [(c.start, c.end) for c in claims]

    def over_price(s: int, e: int) -> bool:
        return any(s < pe and e > ps for ps, pe in price_spans)

    for m in _RE_PCT.finditer(text):
        tail = text[m.end():m.end() + 14]
        head = text[max(0, m.start() - 14):m.start()]
        n = float(m.group("n"))
        is_disc = bool(
            re.match(r"\s*(?:off|discount|sale|cash\s*back|savings?|less|cheaper|lower|reduction)\b", tail, re.I)
            or re.search(r"(?:flat|upto|up\s+to|save|extra|additional|get|enjoy|min(?:imum)?|"
                         r"maximum|max)\s*$", head, re.I)
        )
        material = bool(re.match(rf"\s*(?:\w+\s+)?{_FABRIC}\b", tail, re.I))
        end = m.end()
        word = re.match(r"\s*(?:off|discount|sale|cash\s*back|savings?)\b", tail, re.I)
        if word:  # "20% off" / "20% discount" is one claim, so the keyword is not reported twice
            end += word.end()
        claims.append(_Claim("percent", m.start(), end, n,
                             flavour="discount" if is_disc else ("material" if material else "other")))
    for m in _RE_SIZE_CUE.finditer(text):
        lst = m.group("list")
        base = m.start("list")
        for t in re.finditer(rf"{_SIZE_LETTERS}|\d{{1,2}}(?:\.\d)?", lst, re.I):
            claims.append(_Claim("size", base + t.start(), base + t.end(), t.group(0).upper()))
    for m in _RE_SIZE_REGION.finditer(text):
        claims.append(_Claim("size", m.start(), m.end(), m.group(0).upper().replace(" ", "")))
    for m in _RE_SIZE_IN.finditer(text):
        for t in re.finditer(_SIZE_LETTERS, m.group("s")):
            claims.append(
                _Claim("size", m.start("s") + t.start(), m.start("s") + t.end(), t.group(0).upper())
            )
    for m in _RE_SIZE_MULTI.finditer(text):
        claims.append(_Claim("size", m.start("s"), m.end("s"), m.group("s").upper()))
    for m in _RE_SIZE_FREE.finditer(text):
        claims.append(_Claim("size", m.start(), m.end(), "FREE", flavour="free"))
    for m in _RE_FIT.finditer(text):
        claims.append(_Claim("size", m.start(), m.end(), m.group(0).lower(), flavour="fit"))
    for m in _RE_STOCK.finditer(text):
        if m.group("cnt"):
            n = int(m.group("n") or m.group("n2"))
            claims.append(_Claim("stock", m.start(), m.end(), float(n), flavour="count"))
        else:
            claims.append(_Claim("stock", m.start(), m.end(), None,
                                 flavour="oos" if m.group("oos") else "in"))
    for m in _RE_DISCOUNT.finditer(text):
        claims.append(_Claim("discount", m.start(), m.end(), m.group(0).lower()))
    for m in _RE_DELIVERY.finditer(text):
        claims.append(_Claim("delivery", m.start(), m.end(), m.group(0).lower()))
    for sent_s, sent_e in _sentences(text):
        sent = text[sent_s:sent_e]
        if _DELIVERY_CTX.search(sent):
            for m in _RE_DAYS.finditer(sent):
                s, e = sent_s + m.start(), sent_s + m.end()
                if not over_price(s, e):
                    claims.append(_Claim("delivery", s, e, m.group(0).lower(), flavour="days"))
    # A discount "N% off" / "flat N" is reported once, as the percent claim.
    pct_spans = [(c.start, c.end) for c in claims if c.kind == "percent"]
    claims = [
        c for c in claims
        if not (c.kind == "discount" and any(c.start < pe and c.end > ps for ps, pe in pct_spans))
    ]
    disc_spans = [(c.start, c.end) for c in claims if c.kind == "discount"]
    uniq: dict[tuple[str, int, int], _Claim] = {}
    for c in claims:
        if c.kind == "price" and any(c.start < de and c.end > ds for ds, de in disc_spans):
            continue
        uniq.setdefault((c.kind, c.start, c.end), c)  # two size regexes can hit the same "XL"
    return sorted(uniq.values(), key=lambda c: c.start)


# --------------------------------------------------------------------------- item helpers


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower()).strip()


def _name_texts(it: dict[str, Any]) -> list[str]:
    return [t for t in (_norm(it.get("display_name")), _norm(it.get("prod_name"))) if t]


def _item_text(it: dict[str, Any]) -> str:
    return " ".join(str(it.get(k) or "") for k in ("display_name", "prod_name", "detail_desc"))


def _mentions(
    sentence: str, items: Sequence[dict[str, Any]]
) -> tuple[list[tuple[int, list[dict[str, Any]]]], list[tuple[int, int]]]:
    """(position, items) groups for items the sentence refers to by name, in text order.

    Candidate phrases per item: the full name, its first 3 words, any 2-word window (>= 8 letters).
    A longer match claims its text span; shorter matches overlapping it are dropped, so
    "MLS Pathani Suit 2PCS" is not also credited to "MLS Pathani Suit (White ...)" through their
    shared 3-word prefix. Items matching the identical span form one group (a genuine tie).
    Also returns the claimed name spans: a keyword inside an item's own name ("Diwali Sale
    Essentials") is the name, not a claim by the assistant.
    """
    low = sentence.lower()
    cands: list[tuple[int, int, int, int]] = []  # (n_words, start, end, item index)
    for idx, it in enumerate(items):
        for name in _name_texts(it):
            words = name.split()
            phrases = {name}
            if len(words) >= 3:
                phrases.add(" ".join(words[:3]))
            phrases.update(
                f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)
                if len(words[i]) + len(words[i + 1]) >= 8
            )
            for ph in phrases:
                pat = r"\b" + r"\W+".join(re.escape(w) for w in ph.split()) + r"\b"
                cands += [(len(ph.split()), m.start(), m.end(), idx) for m in re.finditer(pat, low)]
    claimed: list[tuple[int, int, int]] = []  # (n_words, start, end)
    groups: dict[tuple[int, int], set[int]] = {}
    for n, st, en, idx in sorted(cands, key=lambda c: (-c[0], c[1])):
        if any(cn > n and st < ce and en > cs for cn, cs, ce in claimed):
            continue
        claimed.append((n, st, en))
        groups.setdefault((st, en), set()).add(idx)
    groups_out = [(st, [items[i] for i in sorted(ids)]) for (st, _), ids in sorted(groups.items())]
    return groups_out, [(st, en) for _, st, en in claimed]


def _owner(pos: int, mentions: list[tuple[int, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """The item(s) an amount at `pos` is attributed to: the nearest preceding mention
    (<= 90 chars), else the nearest following one."""
    before = [(pos - p, its) for p, its in mentions if p <= pos and pos - p <= 90]
    if before:
        return min(before, key=lambda t: t[0])[1]
    after = [(p - pos, its) for p, its in mentions if p > pos]
    return min(after, key=lambda t: t[0])[1] if after else []


def _price(it: dict[str, Any]) -> float | None:
    p = it.get("price_inr")
    return float(p) if isinstance(p, (int, float)) and not isinstance(p, bool) else None


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= PRICE_TOL


def _name_number_exempt(text: str, c: _Claim, items: Sequence[dict[str, Any]]) -> bool:
    """A bare number that is part of an item name ("Levi's 511") is a name, not a price."""
    raw = text[c.start:c.end]
    if re.search(_CUR, text[max(0, c.start - 3):c.start], re.I) or re.search(_CUR_SUFFIX, text[c.end:c.end + 8], re.I):
        return False
    m = re.search(_NUM, raw)
    if not m:
        return False
    digits = m.group(0)
    prev = re.findall(r"[A-Za-z']+", text[max(0, c.start - 24):c.start])
    prev_w = _norm(prev[-1]) if prev else ""
    for it in items:
        toks = _norm(" ".join(str(it.get(k) or "") for k in ("display_name", "prod_name"))).split()
        if digits in toks and i_adj(toks, digits, prev_w):
            return True
    return False


def i_adj(toks: list[str], digits: str, prev_w: str) -> bool:
    """True when `digits` follows `prev_w` in the item's name tokens."""
    return any(toks[i] == digits and i > 0 and toks[i - 1] == prev_w for i in range(len(toks)))


def _size_tokens(it: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for s in it.get("sizes") or []:
        t = str(s).upper().replace(" ", "")
        out.add(t)
        if t.isdigit():  # "UK 8" is backed by a plain "8" in the sizes field
            out.update(f"{r}{t}" for r in ("UK", "US", "EU", "IN", "IND"))
    if any(x in out for x in ("FREE", "FREESIZE", "ONESIZE", "ONE")):
        out.add("FREE")
    return out


# --------------------------------------------------------------------------- main entry


def check(
    assistant_text: str,
    shown_items: Iterable[dict[str, Any]],
    user_text: str = "",
    extra_amounts: Iterable[float] = (),
) -> list[Violation]:
    """Return every claim in `assistant_text` not backed by `shown_items`.

    Args:
        assistant_text: the full assistant output for the turn (response + rationale strings).
        shown_items: item dicts exactly as rendered on cards (ItemSummary.model_dump()).
        user_text: the user's message(s); used only for the two narrow echo exemptions.
        extra_amounts: code-computed money figures rendered with the turn (e.g. look totals).
    """
    items = [dict(it) for it in shown_items]
    if not assistant_text or not assistant_text.strip():
        return []
    claims = _extract_claims(assistant_text)
    if not claims:
        return []

    user_amounts = [c.value for c in _money_claims(user_text or "") if isinstance(c.value, float)]
    user_norm = " " + _norm(user_text) + " "
    user_pcts = {c.value for c in _extract_claims(user_text or "") if c.kind == "percent"}
    extras = [float(a) for a in extra_amounts]
    all_prices = [p for p in (_price(i) for i in items) if p is not None]
    out: list[Violation] = []

    for s0, e0 in _sentences(assistant_text):
        sent = assistant_text[s0:e0]
        sc = []
        seen_kw: set[tuple[str, str]] = set()
        mentions, name_spans = _mentions(sent, items)
        for c in claims:
            if not s0 <= c.start < e0:
                continue
            if c.kind != "price" and any(
                ns <= c.start - s0 and c.end - s0 <= ne for ns, ne in name_spans
            ):
                continue  # the words are part of a shown item's own name, not a claim
            if c.kind in {"discount", "delivery"} or (c.kind == "stock" and c.flavour != "count"):
                # keyword claims: one violation per kind per sentence ("next-day delivery" = 1)
                if (c.kind, c.flavour) in seen_kw:
                    continue
                seen_kw.add((c.kind, c.flavour))
            sc.append(c)
        if not sc:
            continue
        clauses = _clause_denials(sent)
        ment = [it for _, its in mentions for it in its]
        scope = ment or items
        total_ctx = bool(re.search(r"\b(?:total|together|combined|altogether|all\s+(?:three|four|five|two)|whole\s+look|entire)\b", sent, re.I))

        def viol(c: _Claim, kind: str, reason: str) -> None:
            out.append(Violation(kind, assistant_text[c.start:c.end], c.start, c.end, reason, sent.strip()))

        for c in sc:
            denial = _in_denial(clauses, c.start - s0)
            if c.kind == "price":
                amt = float(c.value)  # type: ignore[arg-type]
                if _name_number_exempt(assistant_text, c, items):
                    continue
                mp = [p for p in (_price(i) for i in scope) if p is not None]
                backing = mp
                if len(mentions) > 1:  # several items named: the amount must fit the nearest one
                    own = _owner(c.start - s0, mentions)
                    backing = [p for p in (_price(i) for i in own) if p is not None] if own else mp
                backing = backing + [
                    float(i["mrp_inr"]) for i in scope if isinstance(i.get("mrp_inr"), (int, float))
                ]
                if any(_close(amt, p) for p in backing) or any(_close(amt, x) for x in extras):
                    continue
                if total_ctx and all_prices and _close(amt, sum(all_prices)):
                    continue
                if total_ctx and ment and mp and _close(amt, sum(mp)):
                    continue
                head = assistant_text[max(s0, c.start - 40):c.start]
                budget_phrase = bool(_BUDGET_WORDS.search(head[-30:])) or bool(
                    re.match(r"\s*(?:budget|range|limit|cap|ceiling|max|maximum)\b",
                             assistant_text[c.end:c.end + 12], re.I))
                if budget_phrase and any(_close(amt, u) for u in user_amounts):
                    continue
                other = [p for p in all_prices if _close(amt, p)]
                if ment and other:
                    viol(c, "price_wrong_item", f"{amt:g} is the price of a different shown item")
                elif denial and any(_close(amt, u) for u in user_amounts):
                    continue
                else:
                    viol(c, "price", f"{amt:g} INR matches no shown item price")
            elif c.kind == "percent":
                if c.flavour == "material":
                    continue
                if c.flavour == "other" and any(
                    assistant_text[c.start:c.end].replace(" ", "").lower() in _item_text(i).replace(" ", "").lower()
                    for i in scope
                ):
                    continue
                if c.flavour == "discount":
                    if any(isinstance(i.get("discount_pct"), (int, float)) and abs(float(i["discount_pct"]) - float(c.value)) <= 1  # type: ignore[arg-type]
                           for i in scope):
                        continue
                    mrp_ok = False
                    for i in scope:
                        p, mrp = _price(i), i.get("mrp_inr")
                        if p and isinstance(mrp, (int, float)) and mrp > p:
                            mrp_ok = mrp_ok or abs((mrp - p) / mrp * 100 - float(c.value)) <= 1  # type: ignore[arg-type]
                    if mrp_ok:
                        continue
                if denial and c.value in user_pcts:
                    continue
                viol(c, "percent", f"{c.value:g}% ({c.flavour}) not backed by any shown item field")
            elif c.kind == "discount":
                if denial:
                    continue
                if any(i.get("discount_pct") or i.get("mrp_inr") for i in scope):
                    continue
                viol(c, "discount", "discount/sale language with no discount field on any shown item")
            elif c.kind == "size":
                tokens = [_size_tokens(i) for i in scope]
                if c.flavour == "fit":
                    if not denial:
                        viol(c, "size", "fit/size-run claim; no fit data exists")
                    continue
                if denial and (str(c.value).lower() in user_norm or str(c.value).lower().replace("uk", "") in user_norm):
                    continue
                if any(str(c.value) in t for t in tokens):
                    continue
                if c.flavour == "free" and any("FREE" in t for t in tokens):
                    continue
                if denial and c.flavour == "free":
                    continue
                viol(c, "size", f"size {c.value} not in the sizes field of any (mentioned) shown item")
            elif c.kind == "stock":
                if c.flavour == "count":
                    if any(i.get("stock_count") == int(c.value) for i in scope):  # type: ignore[arg-type]
                        continue
                elif c.flavour == "oos":
                    if any(i.get("in_stock") is False for i in scope):
                        continue
                else:
                    if any(i.get("in_stock") is True for i in scope):
                        continue
                if denial:
                    continue
                viol(c, "stock", "stock statement not backed by in_stock/stock_count on a shown item")
            elif c.kind == "delivery":
                if denial:
                    continue
                if c.flavour == "days":
                    nums = [int(x) for x in re.findall(r"\d+", assistant_text[c.start:c.end])[:2]]
                    ok = False
                    for i in scope:
                        d = i.get("delivery_days")
                        dd = [d] if isinstance(d, int) else list(d or [])
                        if dd and nums and all(n in dd or (len(dd) == 2 and dd[0] <= n <= dd[1]) for n in nums):
                            ok = True
                    if ok:
                        continue
                elif re.search(r"free", str(c.value)) and any(i.get("free_shipping") is True for i in scope):
                    continue
                elif any(i.get("delivery_days") is not None or i.get("free_shipping") is not None for i in scope) and not re.search(r"free", str(c.value)):
                    continue
                viol(c, "delivery", "delivery/shipping/return statement with no item field to back it")
    return sorted(out, key=lambda v: v.start)

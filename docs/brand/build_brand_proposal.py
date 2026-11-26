"""Build docs/brand/brand-proposal.html (self-contained) and print the WCAG contrast report.

Docs-only tooling: computes WCAG 2.x contrast ratios from hex values (relative luminance
formula), then renders the proposal page with those numbers embedded. Nothing here is imported
by the product. Run: python docs/brand/build_brand_proposal.py
Exit code 1 if any *intended* pairing of a proposed direction fails AA.
"""
from __future__ import annotations

import colorsys
import html
import sys
from pathlib import Path

OUT = Path(__file__).with_name("brand-proposal.html")


# ---------------------------------------------------------------- contrast maths
def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(max(0, min(1, c)) * 255):02X}" for c in rgb)


def hsl_to_hex(h: float, s: float, light: float) -> str:
    """CSS-style hsl(h s% l%) to hex (the tokens in globals.css are HSL triples)."""
    r, g, b = colorsys.hls_to_rgb(h / 360, light / 100, s / 100)
    return rgb_to_hex((r, g, b))


def lum(hexv: str) -> float:
    def f(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (f(c) for c in hex_to_rgb(hexv))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(fg: str, bg: str) -> float:
    a, b = lum(fg), lum(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def over(fg: str, bg: str, alpha: float) -> str:
    """Composite fg at `alpha` over opaque bg (Tailwind /NN opacity utilities)."""
    f, b = hex_to_rgb(fg), hex_to_rgb(bg)
    return rgb_to_hex(tuple(alpha * x + (1 - alpha) * y for x, y in zip(f, b)))  # type: ignore[arg-type]


def verdict(r: float, kind: str) -> str:
    need = 4.5 if kind == "normal" else 3.0
    return "PASS" if r >= need else "FAIL"


# ---------------------------------------------------------------- current brand (read from
# frontend/app/globals.css; hex from its palette comment, HSL-only tokens converted here)
CREAM, INK, CARD = "#FAF6F1", "#3B3230", "#FFFFFF"
BLUSH50, HAIRLINE = "#FBF1EF", "#E9DFD2"
CHAMP, CHAMP_DEEP, PLUM, PLUM_HOVER, SAGE_TEXT = "#B99A5F", "#A3854B", "#7A4A63", "#64394F", "#7C9082"
MUTED_FG = hsl_to_hex(20.0, 9.5, 42.0)
DESTRUCTIVE = hsl_to_hex(3.5, 51.1, 46.5)
DARK_BG = hsl_to_hex(340.0, 10.8, 16.3)
DARK_MUTED = hsl_to_hex(33.9, 20.0, 74.0)
DARK_SECONDARY = hsl_to_hex(323.1, 10.4, 24.5)
OWNED_BADGE = "#E8A33D"  # hard-coded in OutfitBoard.tsx (magic value, not a token)
OG_TAGLINE = "#6F6259"  # hard-coded in opengraph-image.tsx

# (label, fg, bg, kind, where used)
CURRENT: list[tuple[str, str, str, str, str]] = [
    ("champagne on cream (KNOWN FAILURE)", CHAMP, CREAM, "normal", "Logo.tsx wordmark 'Maitri' (16px semibold), text-champagne"),
    ("champagne on white card", CHAMP, CARD, "normal", "same wordmark when the lockup sits on a card"),
    ("champagne on dark banner", CHAMP, DARK_BG, "normal", "DemoBanner.tsx link (.dark scope, text-xs)"),
    ("champagne-deep on cream", CHAMP_DEEP, CREAM, "normal", "token defined; no text-champagne-deep usage found"),
    ("sage-text on cream", SAGE_TEXT, CREAM, "large", "token defined ('large text only'); no usage found in tsx"),
    ("champagne mark stroke on cream", CHAMP, CREAM, "ui", "LogoMark in MessageList / saved-looks / favicon tile (graphic)"),
    ("hairline border on cream (inputs)", HAIRLINE, CREAM, "ui", "--border / --input: input and card boundaries"),
    ("champagne/40 chip border on cream", over(CHAMP, CREAM, 0.4), CREAM, "ui", "MessageList + OccasionChips + MessageBubble chips"),
    ("muted-foreground on cream", MUTED_FG, CREAM, "normal", "97 usages (text-muted-foreground)"),
    ("muted-foreground on white card", MUTED_FG, CARD, "normal", "card captions"),
    ("muted-foreground on blush-50", MUTED_FG, BLUSH50, "normal", "assistant bubble captions"),
    ("muted-foreground/70 on cream (11px)", over(MUTED_FG, CREAM, 0.7), CREAM, "normal", "BodyShapeUpload 'For her/him', OutfitBoard slot hint"),
    ("plum on cream (links)", PLUM, CREAM, "normal", "text-primary links"),
    ("white on plum (CTA)", "#FFFFFF", PLUM, "normal", "primary buttons"),
    ("white on plum-hover", "#FFFFFF", PLUM_HOVER, "normal", "button hover"),
    ("plum on plum/10 pill over white", PLUM, over(PLUM, CARD, 0.10), "normal", "look/[id] + OutfitBoard price pill (text-xs)"),
    ("plum/80 on plum/10 (9px)", over(PLUM, over(PLUM, CARD, 0.10), 0.8), over(PLUM, CARD, 0.10), "normal", "OutfitBoard 9px badge"),
    ("ink on cream", INK, CREAM, "normal", "body text"),
    ("destructive on cream", DESTRUCTIVE, CREAM, "normal", "error text (text-destructive)"),
    ("destructive on destructive/10 over cream", DESTRUCTIVE, over(DESTRUCTIVE, CREAM, 0.10), "normal", "ChatThread / embed error banner (text-xs)"),
    ("white on destructive", "#FFFFFF", DESTRUCTIVE, "normal", "destructive button token"),
    ("white on owned-badge #E8A33D", "#FFFFFF", OWNED_BADGE, "normal", "OutfitBoard 'owned' label (10px, hard-coded hex)"),
    ("OG tagline #6F6259 on cream (32px)", OG_TAGLINE, CREAM, "large", "opengraph-image.tsx tagline"),
    ("focus ring plum on cream", PLUM, CREAM, "ui", "--ring"),
    ("dark: cream on plum-charcoal", CREAM, DARK_BG, "normal", ".dark foreground (DemoBanner)"),
    ("dark: muted-foreground on dark bg", DARK_MUTED, DARK_BG, "normal", ".dark token"),
]

# ---------------------------------------------------------------- proposed directions
DIRS = [
    {
        "slug": "a", "name": "A. Marigold and Plum (refine)", "short": "Refine",
        "why": ("Keeps the cream/plum/marigold-knot equity already in product, but fixes the one "
                "structural flaw: gold was used as TEXT. Gold becomes a darkened mark colour and a "
                "separate darker text gold. Lowest migration cost; reads as soft, airy, bridal "
                "pastel for sangeet / mehendi / haldi browsing."),
        "display": "Fraunces, 'Iowan Old Style', Georgia, serif", "body": "Inter, system-ui, 'Segoe UI', Roboto, sans-serif",
        "fonts_note": "Fraunces + Inter are already bundled via next/font (no new font).",
        "L": dict(bg="#FAF6F1", surface="#FFFFFF", wash="#FBF1EF", ink="#3B3230", muted="#665955", primary="#7A4A63",
                  primaryFg="#FFFFFF", hover="#64394F", accent="#E8A33D", accentText="#7A5C1E", mark="#8F6F2E",
                  border="#857870", error="#A3302A", errWash="#F8E6E3"),
        "D": dict(bg="#2E2528", surface="#3A2F33", wash="#3F3236", ink="#FAF6F1", muted="#D3C8BC", primary="#D9BE82",
                  primaryFg="#2E2528", hover="#E8D09A", accent="#E8A33D", accentText="#E3C887", mark="#D9BE82",
                  border="#9B8D93", error="#F29A90", errWash="#4A2C2C"),
    },
    {
        "slug": "b", "name": "B. Sindoor and Ivory (bold)", "short": "Bold",
        "why": ("Deep sindoor-maroon CTAs with marigold accents: the colours of the bridal lehenga and "
                "the mandap, higher chroma than today. Photography still leads, but the brand feels "
                "ceremonial rather than pastel; stronger on a phone in sunlight (higher contrast "
                "headroom). Bigger visual change, so higher migration risk."),
        "display": "Georgia, 'Iowan Old Style', 'Times New Roman', serif", "body": "system-ui, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif",
        "fonts_note": "System/web-safe only: removes both font downloads (no CLS from swap).",
        "L": dict(bg="#FFF8EF", surface="#FFFFFF", wash="#FDEFE0", ink="#2A1F1E", muted="#5E4A47", primary="#8A1C3C",
                  primaryFg="#FFFFFF", hover="#6E1530", accent="#E8A33D", accentText="#7D5200", mark="#9A6A12",
                  border="#85746B", error="#B3261E", errWash="#FBE4E1"),
        "D": dict(bg="#241618", surface="#33201F", wash="#3A2524", ink="#FFF3E4", muted="#D9C3B8", primary="#F2A7B8",
                  primaryFg="#241618", hover="#F7C0CD", accent="#F0B54F", accentText="#F0B54F", mark="#F0B54F",
                  border="#9A8580", error="#FF9C94", errWash="#4A2624"),
    },
    {
        "slug": "c", "name": "C. Peacock and Pearl (modern)", "short": "Modern",
        "why": ("Peacock teal with pearl and rose-gold: a cooler, contemporary-luxe palette that "
                "separates from the pink/red wedding-site sea and from Indian fast-fashion marketplaces. "
                "Teal-on-pearl gives the strongest contrast; rose-gold is decoration only. Largest "
                "departure from the existing knot-and-plum identity."),
        "display": "'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif", "body": "system-ui, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif",
        "fonts_note": "System/web-safe only; Palatino resolves to Palatino Linotype / URW Palladio / Georgia.",
        "L": dict(bg="#F6F4EF", surface="#FFFFFF", wash="#E8F1EF", ink="#1E2B2C", muted="#4D5F60", primary="#0E5A5F",
                  primaryFg="#FFFFFF", hover="#0A4549", accent="#C98B76", accentText="#8A4A38", mark="#A8604A",
                  border="#788785", error="#A8322D", errWash="#F7E3E0"),
        "D": dict(bg="#122426", surface="#1B3336", wash="#1F3A3D", ink="#F3F1EA", muted="#B8C9C7", primary="#6FCBC9",
                  primaryFg="#122426", hover="#92DAD8", accent="#E0A894", accentText="#E8B5A2", mark="#E0A894",
                  border="#6E8A88", error="#FF9A92", errWash="#40201F"),
    },
]


def pairs_for(p: dict[str, str]) -> list[tuple[str, str, str, str]]:
    """(label, fg, bg, kind) for every intended pairing of a palette."""
    return [
        ("ink on bg", p["ink"], p["bg"], "normal"),
        ("ink on surface", p["ink"], p["surface"], "normal"),
        ("muted on bg", p["muted"], p["bg"], "normal"),
        ("muted on surface", p["muted"], p["surface"], "normal"),
        ("muted on wash (bubble)", p["muted"], p["wash"], "normal"),
        ("primary text/link on bg", p["primary"], p["bg"], "normal"),
        ("primary text on wash", p["primary"], p["wash"], "normal"),
        ("button label on primary", p["primaryFg"], p["primary"], "normal"),
        ("button label on hover", p["primaryFg"], p["hover"], "normal"),
        ("accent text (gold) on bg", p["accentText"], p["bg"], "normal"),
        ("ink on accent fill (badge)", p["ink"] if p["bg"][1:3] > "80" else p["bg"], p["accent"], "normal"),
        ("error text on bg", p["error"], p["bg"], "normal"),
        ("error text on error wash", p["error"], p["errWash"], "normal"),
        ("logo mark on bg (graphic)", p["mark"], p["bg"], "ui"),
        ("input/chip border on bg", p["border"], p["bg"], "ui"),
        ("focus ring (primary) on bg", p["primary"], p["bg"], "ui"),
    ]


def table_text(rows: list[tuple[str, str, str, str]]) -> tuple[str, list[str]]:
    lines, fails = [], []
    for label, fg, bg, kind in rows:
        r = ratio(fg, bg)
        v = verdict(r, kind)
        if v == "FAIL":
            fails.append(label)
        need = "4.5" if kind == "normal" else "3.0"
        lines.append(f"{label:<40} {fg} on {bg}  {r:6.2f}:1  need {need}  {v}")
    return "\n".join(lines), fails


def build_report() -> tuple[str, list[str], list[tuple[str, str, str, str, str, float, str]]]:
    out: list[str] = []
    cur_rows: list[tuple[str, str, str, str, str, float, str]] = []
    out.append("CURRENT BRAND (hex from frontend/app/globals.css; alpha utilities composited)")
    for label, fg, bg, kind, where in CURRENT:
        r = ratio(fg, bg)
        v = verdict(r, kind)
        cur_rows.append((label, fg, bg, kind, where, r, v))
        need = "4.5" if kind == "normal" else "3.0"
        out.append(f"{label:<44} {fg} on {bg}  {r:6.2f}:1  need {need}  {v}")
    nfail = sum(1 for x in cur_rows if x[6] == "FAIL")
    out.append(f"-> {nfail} of {len(cur_rows)} current pairs FAIL")
    bad: list[str] = []
    for d in DIRS:
        for mode in ("L", "D"):
            out.append("")
            out.append(f"PROPOSED {d['name']} [{'light' if mode == 'L' else 'dark'}]")
            txt, fails = table_text(pairs_for(d[mode]))
            out.append(txt)
            bad += [f"{d['slug']}/{mode}: {f}" for f in fails]
    out.append("")
    out.append("ALL PROPOSED INTENDED PAIRINGS PASS AA" if not bad else "PROPOSED FAILURES: " + "; ".join(bad))
    return "\n".join(out), bad, cur_rows


# ---------------------------------------------------------------- SVG pieces
MARK_PATH = ("M14.50,0.00 C14.50,6.60 4.30,7.30 0.00,3.00 C-4.30,7.30 -14.50,6.60 -14.50,0.00 "
             "C-14.50,-6.60 -4.30,-7.30 0.00,-3.00 C4.30,-7.30 14.50,-6.60 14.50,0.00 Z")
SYMBOLS = (
    '<svg width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false"><defs>'
    '<symbol id="knot" viewBox="0 0 48 48"><g transform="translate(24,24)" fill="none" stroke="currentColor" '
    'stroke-width="3.8" stroke-linecap="round" stroke-linejoin="round">'
    f'<path d="{MARK_PATH}"/><path d="{MARK_PATH}" transform="rotate(90)"/></g></symbol>'
    # Direction B mark: sindoor dot inside a marigold ring (sketch)
    '<symbol id="ring" viewBox="0 0 48 48"><g fill="none" stroke="currentColor" stroke-width="3.8" '
    'stroke-linecap="round"><circle cx="24" cy="24" r="16"/><circle cx="24" cy="24" r="6" fill="currentColor"/>'
    '<path d="M24 4v6M24 38v6M4 24h6M38 24h6"/></g></symbol>'
    # Direction C mark: peacock-feather eye (sketch)
    '<symbol id="eye" viewBox="0 0 48 48"><g fill="none" stroke="currentColor" stroke-width="3.8" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M24 6 C38 14 38 34 24 42 C10 34 10 14 24 6 Z"/>'
    '<circle cx="24" cy="25" r="5" fill="currentColor"/></g></symbol>'
    "</defs></svg>"
)
MARK_ID = {"a": "knot", "b": "ring", "c": "eye"}
CUR_FONT = "Fraunces, Georgia, serif"


def wordmark_svg(d: dict, mode: str = "L", width: int = 240) -> str:
    p = d[mode]
    mid = MARK_ID[d["slug"]]
    f = html.escape(d["display"], quote=True)
    return (
        f'<svg viewBox="0 0 240 48" width="{width}" role="img" aria-label="Style Maitri wordmark sketch">'
        f'<use href="#{mid}" x="0" y="4" width="40" height="40" style="color:{p["mark"]}"/>'
        f'<text x="50" y="32" font-family="{f}" font-size="26" font-weight="600" fill="{p["ink"]}">Style '
        f'<tspan fill="{p["accentText"]}">Maitri</tspan></text></svg>'
    )


def favicon_svg(d: dict) -> str:
    p = d["L"]
    mid = MARK_ID[d["slug"]]
    return (
        f'<svg viewBox="0 0 48 48" width="64" role="img" aria-label="Favicon sketch">'
        f'<rect width="48" height="48" rx="10" fill="{p["primary"]}"/>'
        f'<use href="#{mid}" x="6" y="6" width="36" height="36" style="color:{p["primaryFg"]}"/></svg>'
    )


def og_svg(d: dict) -> str:
    p = d["L"]
    mid = MARK_ID[d["slug"]]
    f = html.escape(d["display"], quote=True)
    b = html.escape(d["body"], quote=True)
    return (
        f'<svg viewBox="0 0 1200 630" role="img" aria-label="OG card sketch" style="width:100%;height:auto;border-radius:8px">'
        f'<rect width="1200" height="630" fill="{p["bg"]}"/><rect y="590" width="1200" height="40" fill="{p["primary"]}"/>'
        f'<use href="#{mid}" x="535" y="90" width="130" height="130" style="color:{p["mark"]}"/>'
        f'<text x="600" y="340" text-anchor="middle" font-family="{f}" font-size="84" font-weight="600" fill="{p["ink"]}">Style '
        f'<tspan fill="{p["accentText"]}">Maitri</tspan></text>'
        f'<text x="600" y="420" text-anchor="middle" font-family="{b}" font-size="36" fill="{p["muted"]}">'
        f'Your AI stylist for every occasion</text>'
        f'<text x="600" y="520" text-anchor="middle" font-family="{b}" font-size="28" fill="{p["primary"]}">'
        f'Sangeet · Mehendi · Haldi · Reception</text></svg>'
    )


def sample(d: dict, mode: str) -> str:
    p = d[mode]
    style = ";".join(f"--{k}:{v}" for k, v in p.items())
    badge_fg = p["ink"] if p["bg"][1:3] > "80" else p["bg"]  # same rule as pairs_for()
    style += f";--badgeFg:{badge_fg};--fd:{d['display']};--fb:{d['body']}"
    style = html.escape(style, quote=True)
    label = "Light" if mode == "L" else "Dark"
    return f"""<div class="smp" style="{style}" role="group" aria-label="{html.escape(d['short'])} {label} samples">
<p class="lbl">{label} surface</p>
<div class="chatcard">
 <div class="bubble">For a daytime haldi, this yellow kurta set stays breathable and photographs well.</div>
 <div class="prod"><div class="img" aria-hidden="true"></div><div><b class="t">Mustard chikankari kurta set</b><br><span class="m">Rs 4,299 · Ajio</span><br><span class="badge">Owned</span></div></div>
 <div class="row"><button class="btn pri">Add to look</button><button class="btn sec">Save</button><button class="btn gh">Share</button></div>
 <div class="row"><span class="chip">Sangeet</span><span class="chip">Mehendi</span><span class="chip on">Haldi</span></div>
</div>
<div class="tiles">
 <div class="tile"><span class="sk" aria-hidden="true"></span><span class="sk s2" aria-hidden="true"></span><small>Loading: finding looks...</small></div>
 <div class="tile"><span class="ic" aria-hidden="true">+</span><small>Empty: no saved looks yet. Try a Sangeet chip.</small></div>
 <div class="tile err" role="alert"><span class="ic" aria-hidden="true">!</span><small>Error: could not load items. Retry.</small></div>
</div>
</div>"""


CSS = """
:root{--pbg:#FBFAF8;--ptx:#2B2523;--pmu:#5F5550;--pcard:#FFFFFF;--pbd:#D8D0C8;--pac:#7A4A63;--pfail:#A3302A;--pok:#2F6B3F}
@media (prefers-color-scheme:dark){:root{--pbg:#1C1819;--ptx:#F1EAE4;--pmu:#BFB3AB;--pcard:#272122;--pbd:#4A4144;--pac:#E2B7CF;--pfail:#FF9C94;--pok:#8FD3A0}}
*{box-sizing:border-box}
body{margin:0;background:var(--pbg);color:var(--ptx);font:16px/1.55 system-ui,'Segoe UI',Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:16px}
h1{font:600 2rem/1.2 Georgia,serif;margin:.4em 0}h2{font:600 1.5rem/1.3 Georgia,serif;margin:1.8em 0 .4em;border-bottom:1px solid var(--pbd);padding-bottom:.2em}
h3{font-size:1.1rem;margin:1.4em 0 .3em}a{color:var(--pac)}
.banner{border:2px solid var(--pfail);border-radius:8px;padding:10px 14px;background:var(--pcard)}
.pill{display:inline-block;border:1px solid var(--pac);color:var(--pac);border-radius:999px;padding:1px 10px;font-weight:600;font-size:.85rem}
table{border-collapse:collapse;width:100%;font-size:.88rem;background:var(--pcard)}th,td{border:1px solid var(--pbd);padding:5px 7px;text-align:left;vertical-align:top}
.wrap{overflow-x:auto}td.sw span{display:inline-block;width:3.4em;padding:2px 4px;border:1px solid var(--pbd);border-radius:4px;font:.75rem monospace}
.f{color:var(--pfail);font-weight:700}.ok{color:var(--pok);font-weight:700}
pre{background:var(--pcard);border:1px solid var(--pbd);padding:10px;overflow:auto;font-size:.72rem;line-height:1.4}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
.card{background:var(--pcard);border:1px solid var(--pbd);border-radius:10px;padding:12px}
.demo{padding:14px;border-radius:8px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.chips span{display:inline-block;margin:2px 6px 2px 0;padding:2px 9px;border:1px solid var(--pbd);border-radius:999px;font:.78rem monospace}
.dir{border-top:4px solid var(--pac);margin-top:2.2em;padding-top:.5em}
.smp{background:var(--bg);color:var(--ink);font-family:var(--fb);padding:12px;border-radius:10px;border:1px solid var(--border)}
.smp .lbl{margin:0 0 6px;color:var(--muted);font-size:.78rem}
.smp .chatcard{display:grid;gap:10px}
.smp .bubble{background:var(--wash);color:var(--ink);border-radius:12px;padding:8px 10px;font-size:.88rem}
.smp .prod{display:flex;gap:10px;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:8px}
.smp .img{width:64px;height:80px;border-radius:6px;background:linear-gradient(135deg,var(--accent),var(--primary))}
.smp .t{font-family:var(--fd);font-weight:600}.smp .m{color:var(--muted);font-size:.8rem}
.smp .badge{display:inline-block;margin-top:4px;background:var(--accent);color:var(--badgeFg);font-size:.72rem;font-weight:600;padding:1px 7px;border-radius:4px}
.smp .row{display:flex;gap:8px;flex-wrap:wrap}
.smp .btn{min-height:44px;padding:0 16px;border-radius:10px;font:600 .9rem var(--fb);cursor:pointer}
.smp .pri{background:var(--primary);color:var(--primaryFg);border:0}.smp .pri:hover{background:var(--hover)}
.smp .sec{background:var(--surface);color:var(--ink);border:1px solid var(--border)}
.smp .gh{background:transparent;color:var(--primary);border:0;text-decoration:underline}
.smp .btn:focus-visible{outline:3px solid var(--primary);outline-offset:2px}
.smp .chip{min-height:44px;display:inline-flex;align-items:center;padding:0 14px;border-radius:999px;border:1px solid var(--border);background:var(--surface);font-size:.85rem}
.smp .chip.on{background:var(--primary);color:var(--primaryFg);border-color:var(--primary)}
.smp .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-top:10px}
.smp .tile{border:1px dashed var(--border);border-radius:10px;padding:8px;background:var(--surface);color:var(--muted);font-size:.78rem}
.smp .tile.err{background:var(--errWash);color:var(--error);border:1px solid var(--error)}
.smp .ic{display:block;font:700 1.2rem var(--fd);color:var(--primary)}.smp .tile.err .ic{color:var(--error)}
.smp .sk{display:block;height:12px;border-radius:6px;background:var(--wash);margin-bottom:6px;animation:p 1.4s ease-in-out infinite}.smp .sk.s2{width:60%}
@keyframes p{50%{opacity:.4}}@media (prefers-reduced-motion:reduce){.smp .sk{animation:none}}
.logos{display:flex;gap:18px;flex-wrap:wrap;align-items:center}.logos .b{padding:10px;border-radius:8px}
"""


def swatch_row(p: dict[str, str]) -> str:
    return "".join(
        f'<span style="background:{v};color:{"#000" if lum(v) > 0.4 else "#FFF"}">{k} {v}</span> ' for k, v in p.items()
    )


def cur_table(rows: list[tuple[str, str, str, str, str, float, str]], only_fail: bool) -> str:
    h = ['<div class="wrap"><table><thead><tr><th>Pair</th><th>fg / bg</th><th>Ratio</th><th>Need</th><th>Result</th><th>Where</th></tr></thead><tbody>']
    for label, fg, bg, kind, where, r, v in rows:
        if only_fail and v != "FAIL":
            continue
        need = "4.5" if kind == "normal" else "3.0"
        cls = "f" if v == "FAIL" else "ok"
        h.append(
            f'<tr><td>{html.escape(label)}</td><td class="sw"><span style="background:{fg};color:{bg}">fg {fg}</span> '
            f'<span style="background:{bg};color:{fg}">bg {bg}</span></td><td>{r:.2f}:1</td><td>{need}:1 ({kind})</td>'
            f'<td class="{cls}">{v}</td><td>{html.escape(where)}</td></tr>'
        )
    h.append("</tbody></table></div>")
    return "".join(h)


def dir_table(p: dict[str, str]) -> str:
    h = ['<div class="wrap"><table><thead><tr><th>Intended pairing</th><th>fg / bg</th><th>Ratio</th><th>Need</th><th>Result</th></tr></thead><tbody>']
    for label, fg, bg, kind in pairs_for(p):
        r = ratio(fg, bg)
        v = verdict(r, kind)
        need = "4.5" if kind == "normal" else "3.0"
        h.append(
            f'<tr><td>{label}</td><td class="sw"><span style="background:{fg};color:{bg}">{fg}</span> on '
            f'<span style="background:{bg};color:{fg}">{bg}</span></td><td>{r:.2f}:1</td><td>{need}:1</td>'
            f'<td class="{"ok" if v == "PASS" else "f"}">{v}</td></tr>'
        )
    h.append("</tbody></table></div>")
    return "".join(h)


def build_html(report: str, cur_rows: list[tuple[str, str, str, str, str, float, str]]) -> str:
    known = next(x for x in cur_rows if x[0].startswith("champagne on cream"))
    fails = [x for x in cur_rows if x[6] == "FAIL"]
    parts: list[str] = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<meta name="color-scheme" content="light dark">',
        "<title>Style Maitri brand proposal (V1, nothing applied)</title>",
        f"<style>{CSS}</style></head><body>{SYMBOLS}<main>",
        "<h1>Style Maitri: brand proposal (V1)</h1>",
        '<p><span class="pill">PENDING GG DECISION</span> Proposal only. Nothing here is applied to the product; '
        "no frontend file, logo, favicon, token or copy was changed. This page is not served by the site. "
        "Generated by <code>docs/brand/build_brand_proposal.py</code>; all ratios are computed (WCAG 2.x relative luminance), "
        "not user-tested.</p>",
        "<h2>1. Current brand, as it is</h2>",
        "<p>Source: <code>frontend/app/globals.css</code> tokens, <code>tailwind.config.ts</code>, <code>components/Logo.tsx</code>, "
        "<code>public/brand/*.svg</code>, <code>app/icon.svg</code>, <code>app/opengraph-image.tsx</code>.</p>",
        '<div class="grid"><div class="card"><h3>Colour</h3>'
        f'<div class="chips"><span style="background:{CREAM}">cream {CREAM}</span><span style="background:{INK};color:#fff">ink {INK}</span>'
        f'<span style="background:{PLUM};color:#fff">plum {PLUM}</span><span style="background:{CHAMP}">champagne {CHAMP}</span>'
        f'<span style="background:{BLUSH50}">blush-50 {BLUSH50}</span><span style="background:{HAIRLINE}">hairline {HAIRLINE}</span>'
        f'<span style="background:{SAGE_TEXT}">sage-text {SAGE_TEXT}</span></div></div>',
        '<div class="card"><h3>Type</h3><p>Fraunces 500/600 (display + wordmark) and Inter (body), loaded by next/font/google in '
        "<code>app/layout.tsx</code>. This offline page falls back to Georgia / system-ui, so glyphs differ slightly from product.</p></div>",
        '<div class="card"><h3>Logo usage</h3>'
        '<div class="demo" style="background:#FAF6F1;color:#3B3230"><svg width="40" height="40" style="color:#B99A5F"><use href="#knot"/></svg>'
        '<span style="font:600 22px Fraunces,Georgia,serif">Style <span style="color:#B99A5F">Maitri</span></span></div>'
        "<p>Marigold Knot mark in champagne <code>#B99A5F</code>; wordmark 'Maitri' in champagne on cream. "
        "Favicon = knot on a cream tile; OG card = cream, knot, wordmark, taupe tagline.</p></div></div>",
        "<h3>Known failure: champagne on cream</h3>",
        f'<div class="banner"><b class="f">FAIL</b> Wordmark "Maitri" (<code>text-champagne</code>) in <code>Logo.tsx</code>: foreground <b>{known[1]}</b> '
        f"on background <b>{known[2]}</b> = <b>{known[5]:.2f}:1</b> (need 4.5:1 for 16px semibold text; the task brief expected about 2.49:1"
        f"{', matches' if abs(known[5] - 2.49) < 0.01 else ', MY computed number differs: ' + format(known[5], '.2f')}). "
        '<div class="demo" style="background:#FAF6F1"><span style="font:600 22px Fraunces,Georgia,serif;color:#3B3230">Style '
        '<span style="color:#B99A5F">Maitri</span></span></div></div>',
        f"<h3>Every current pair that fails AA ({len(fails)} of {len(cur_rows)} checked)</h3>",
        cur_table(cur_rows, True),
        "<p>Unused tokens (champagne-deep, sage-text) are listed because they are defined and invite misuse. "
        "Ring/border pairs use the 3:1 non-text rule (WCAG 1.4.11); decorative hairlines that do not identify a control are exempt "
        "but <code>--input</code> reuses the same hex for input boundaries. Brand-injected pages (pdp-demo white-on-brand header) depend on partner colours and were not checked.</p>",
        "<h3>All current pairs</h3>",
        cur_table(cur_rows, False),
        "<h2>2. Proposal: three directions</h2>",
        "<p>Shared principles: gold is never used as small text; every text pairing is at least 4.5:1, graphics and control borders at least 3:1; "
        "one tokens source (the CSS variables in <code>globals.css</code>) feeds wordmark, favicon and OG card; the OG route keeps hex literals only because Satori cannot read CSS variables, "
        "so the build script/tokens should be the single place those hex values come from.</p>",
    ]
    for d in DIRS:
        parts += [
            f'<section class="dir"><h3>{html.escape(d["name"])}</h3><p>{html.escape(d["why"])}</p>',
            f"<p><b>Type:</b> display <code>{html.escape(d['display'])}</code>; body <code>{html.escape(d['body'])}</code>. {html.escape(d['fonts_note'])}</p>",
            f'<div class="chips">{swatch_row(d["L"])}</div><div class="chips">{swatch_row(d["D"])}</div>',
            "<h4>Contrast (light)</h4>", dir_table(d["L"]), "<h4>Contrast (dark)</h4>", dir_table(d["D"]),
            "<h4>Wordmark, favicon, OG sketches</h4>",
            f'<div class="logos"><div class="b" style="background:{d["L"]["bg"]}">{wordmark_svg(d, "L")}</div>'
            f'<div class="b" style="background:{d["D"]["bg"]}">{wordmark_svg(d, "D")}</div>{favicon_svg(d)}</div>',
            f'<div style="max-width:520px;margin-top:10px">{og_svg(d)}</div>',
            '<h4>Chat card, buttons, chips, states</h4><div class="grid">' + sample(d, "L") + sample(d, "D") + "</div></section>",
        ]
    parts += [
        "<h2>3. Recommendation</h2>",
        "<p><b>Direction A (Marigold and Plum, refine).</b> It fixes every current AA failure that is caused by the palette "
        "(gold-as-text, hairline inputs, hard-coded badge) while keeping the equity of the knot mark, plum CTAs and the already-bundled Fraunces/Inter. "
        "Directions B and C are stronger visually but are a rebrand, which the evidence (no user research) cannot yet justify.</p>",
        "<h3>Trade-offs</h3><ul><li>A: lowest risk, least differentiation from other pastel wedding sites.</li>"
        "<li>B: strongest ceremonial identity and no font downloads; highest recolour surface and the red/maroon family is crowded in Indian wedding retail.</li>"
        "<li>C: most distinctive; teal clashes with some product photography (green/blue outfits) and discards the marigold story.</li></ul>",
        "<h3>What would change if A were adopted (files/tokens only; nothing done)</h3><ul>"
        "<li><code>frontend/app/globals.css</code>: add <code>--gold-mark</code> and <code>--gold-text</code>; set <code>--muted-foreground</code> to #665955; raise <code>--border</code>/<code>--input</code> to #857870 (keep a separate decorative hairline token); add dark-mode primary #D9BE82 and wire a theme switch only if desired.</li>"
        "<li><code>frontend/tailwind.config.ts</code>: map the new tokens; deprecate <code>champagne</code> / <code>champagne-deep</code> / <code>sage-text</code> for text.</li>"
        "<li><code>frontend/components/Logo.tsx</code>: wordmark 'Maitri' uses gold-text, mark uses gold-mark.</li>"
        "<li><code>frontend/public/brand/mark*.svg</code>, <code>favicon*.svg</code>, <code>app/icon.svg</code>: stroke #B99A5F to #8F6F2E.</li>"
        "<li><code>frontend/app/opengraph-image.tsx</code>: hex literals for knot, 'Maitri', tagline.</li>"
        "<li><code>components/chat/OutfitBoard.tsx</code>: owned badge #E8A33D with white text becomes token + ink text; 9px primary/80 badge raised in size or opacity.</li>"
        "<li><code>components/DemoBanner.tsx</code>, <code>components/chat/MessageList.tsx</code>, <code>MessageBubble.tsx</code>, <code>components/landing/OccasionChips.tsx</code>: champagne border/text usages.</li>"
        "<li><code>frontend/e2e</code> and component tests asserting class names (e.g. StreamNotice.test.tsx) if tokens are renamed.</li></ul>",
        "<h3>Migration risk</h3><p>Low-medium: colour-only, no layout change, but gold darkens from #B99A5F to roughly #8F6F2E so the brand reads warmer/heavier; "
        "OG image caches on WhatsApp/social persist for days; shared hex duplicates (OG route, SVG files) can drift if not generated from tokens.</p>",
        "<h3>Screenshots / rule 15c plan</h3><p>Before/after screenshots of landing, chat empty state, outfit board, saved looks, error banner and the OG card (light only, since dark is not toggled today), "
        "committed under <code>reports/screenshots/&lt;pr&gt;/</code> on the adopting PR's branch and embedded via raw.githubusercontent.com URLs. "
        "A contrast unit test over the token hex values should gate CI (extends rule 15b/70a gate 5).</p>",
        "<h2>4. Honest limits</h2><ul>"
        "<li>No user research: preference between directions is a judgment, not evidence.</li>"
        "<li>Contrast is computed from hex values, not user-tested; WCAG ratios do not capture thin Fraunces strokes, sunlight glare, or colour-vision differences.</li>"
        "<li>Fonts: Fraunces/Inter are not loaded here (offline page), so renders use fallback fonts; Palatino/Georgia availability varies by OS.</li>"
        "<li>Alpha utilities were composited over the assumed background; real stacking (images under badges) was not measured. Partner-brand pages were not checked.</li>"
        "<li>Logo marks in B and C are quick sketches, not designed marks. Only usages found by grep of <code>app/</code> and <code>components/</code> were listed; dynamically built class names could be missed.</li>"
        "<li>The 'known 2.49:1' claim was re-derived here, see Section 1.</li></ul>",
        "<h2>Appendix: contrast script output</h2><pre>" + html.escape(report) + "</pre>",
        "</main></body></html>",
    ]
    return "".join(parts)


def main() -> int:
    report, bad, cur_rows = build_report()
    print(report)
    OUT.write_text(build_html(report, cur_rows), encoding="utf-8")
    print(f"\nwrote {OUT} ({OUT.stat().st_size} bytes)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

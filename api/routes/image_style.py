"""POST /style/from-image — visual look-composition from an uploaded image.

Flow
----
1. Demo-guard checks (same as POST /chat).
2. Validate content-type + file size (hard 15 MB cap).
3. Decode bytes in-memory via PIL (HEIC via pillow-heif); reject non-images.
4. CLIP-encode the image → 512-d vector.
5. **Drop image bytes immediately** — they are never written to disk/DB/log.
6. FAISS nearest-neighbour search in the brand's CLIP-text index.
7. Pick the top valid article_id as anchor → call compose_outfit_variants.
8. Generate grounded rationales + build cart links.
9. Return the same shape as POST /chat outfit responses.

Privacy guarantee
-----------------
- ``upload_bytes`` is a local variable; it is explicitly deleted (``del``) and
  Python's reference counting ensures the buffer is freed before any I/O-bound
  work starts.
- ``pil_img`` is similarly cleared after ``encode_image`` returns.
- No image data appears in any log statement (structlog or stdlib).
- The ``max_request_body_size="never"`` Sentry option in main.py already
  prevents request bodies from reaching Sentry, but we add no Sentry
  breadcrumbs here either.

Feature flag
-----------
``features.image_input_enabled`` in config.yaml controls the feature.  The
env var ``ENABLE_IMAGE_INPUT=false`` overrides the yaml value to off.
When off (or when the brand CLIP index is missing) the endpoint returns 404.
"""
from __future__ import annotations

import io
import logging
import os
import uuid
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

import api.deps as deps
from api.auth import get_current_user_id_or_demo
from api.schemas import ItemLink, ItemSummary, OutfitVariant
from src.agents.intent_parser import parse_intent
from src.agents.outfit.cart_links import build_cart_action
from src.agents.outfit.composer import compose_outfit_variants
from src.agents.outfit.image_anchor import find_anchor_from_image
from src.agents.outfit.rationale import generate_rationales, template_rationale
from src.retrieval.index_store import UNIFIED_BRAND

logger = logging.getLogger(__name__)
router = APIRouter(tags=["style"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024  # 15 MB hard cap

_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
    }
)

_DEFAULT_OCCASION: str = "casual"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _feature_enabled(config: dict) -> bool:
    """Return True when the image-input feature is active.

    Precedence: env var ENABLE_IMAGE_INPUT → config.yaml features.image_input_enabled.
    """
    env_val = os.environ.get("ENABLE_IMAGE_INPUT", "").lower()
    if env_val in ("0", "false", "no"):
        return False
    if env_val in ("1", "true", "yes"):
        return True
    return bool((config.get("features") or {}).get("image_input_enabled", True))


def _clip_model_id(config: dict) -> str:
    """Return the configured CLIP model id."""
    return str((config.get("clip") or {}).get("model", "clip-ViT-B-32"))


def _brand_index_exists(brand: str, config: dict) -> bool:
    """Return True when the brand's CLIP index files are present on disk."""
    from pathlib import Path

    clip_base = (
        Path(config.get("clip", {}).get("index_dir", "data/processed/clip"))
    )
    if not clip_base.is_absolute():
        from pathlib import Path as _P
        clip_base = _P(__file__).parent.parent.parent / clip_base
    idx_dir = clip_base / brand
    return (idx_dir / "clip.faiss").exists() and (idx_dir / "clip_article_ids.npy").exists()


def _resolve_gender(
    intent_gender: str | None,
    catalogue_df: pd.DataFrame,
    anchor_id: str,
    brand_cfg: Any,
) -> str:
    """Resolve which gender to steer outfit composition with.

    Precedence:
      1. Gender parsed from the user's free-text message (``intent.gender``),
         when it is an unambiguous "men" or "women".
      2. The anchor item's own catalogue ``gender`` column, when it is
         "men" or "women".
      3. The brand's configured default (``brand_cfg.gender_default``).

    "unisex"/"mixed" values are never returned directly — composition needs a
    concrete men/women slice, so "mixed" is coerced to "women" only as the
    final fallback (steps 1 and 2 are skipped for those values, falling
    through to the next precedence level).
    """
    if intent_gender in ("men", "women"):
        return intent_gender

    if "gender" in catalogue_df.columns and "article_id" in catalogue_df.columns:
        match = catalogue_df.loc[catalogue_df["article_id"] == anchor_id, "gender"]
        if not match.empty and match.iloc[0] is not None:
            anchor_gender = str(match.iloc[0]).lower()
            if anchor_gender in ("men", "women"):
                return anchor_gender

    brand_default = (brand_cfg.gender_default if brand_cfg else "women") or "women"
    return "women" if brand_default == "mixed" else brand_default


def _build_variant_response(variant: dict, brand: str) -> OutfitVariant:
    """Convert a compose_outfit look dict into an OutfitVariant schema object."""
    seed = variant.get("seed_item")
    complements = variant.get("complements") or []
    all_items = ([seed] if seed else []) + complements
    item_summaries = [ItemSummary.from_agent_item(it) for it in all_items]

    cart_action = build_cart_action(all_items, brand)
    cart_url = cart_action.get("cart_url")
    raw_links = cart_action.get("item_links") or []
    item_links: list[ItemLink] | None = (
        [
            ItemLink(
                article_id=lk["article_id"],
                name=lk["name"],
                buy_url=lk["buy_url"],
            )
            for lk in raw_links
        ]
        or None
    )

    return OutfitVariant(
        variant_id=variant.get("look_id") or "",
        label=variant.get("variant_label") or "Base",
        rationale=variant.get("outfit_rationale") or variant.get("rationale") or "",
        items=item_summaries,
        occasion=variant.get("occasion"),
        budget_total_inr=variant.get("budget_total_inr"),
        cart_url=cart_url,
        item_links=item_links,
    )


# ---------------------------------------------------------------------------
# POST /style/from-image
# ---------------------------------------------------------------------------


@router.post("/style/from-image")
async def post_style_from_image(
    request: Request,
    file: UploadFile,
    user_id: Annotated[str, Depends(get_current_user_id_or_demo)],
    conversation_id: Annotated[str | None, Form()] = None,
    message: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Compose a look from an uploaded fashion image.

    Multipart upload: field name ``file``.
    Accepted types: image/jpeg, image/png, image/webp, image/heic, image/heif.
    Max size: 15 MB.
    Optional ``message`` text field is parsed for a budget cap ("under 2000")
    and an occasion ("for a party") using the same deterministic intent
    parser as POST /chat; both are echoed back as ``user_text`` in the
    response and used to steer ``compose_outfit_variants``.

    Returns the same payload shape as a chat outfit response:
    ``look_id``, ``occasion``, ``outfit_rationale``, ``outfit_variants``,
    ``items`` (seed + complements), ``cart_url``, ``item_links``,
    ``budget_total_inr``, ``user_text``.

    Returns 404 when image_input_enabled=false or the brand CLIP index is absent.
    Returns 400 for invalid content-type, non-image bytes, or unreadable files.
    Returns 413 for files exceeding the 15 MB limit.
    """
    # Normalise blank/whitespace-only text to None so downstream parsing and
    # the echoed "user_text" field are consistent regardless of how the
    # client sent an empty form field.
    message = message.strip() or None if message else None

    config = deps.get_config()
    # Resolve the active brand the same way api/main.py does at startup:
    # unified mode (BRAND unset or "unified" or UNIFIED=1) → "unified".
    # This ensures image search queries the cross-store CLIP index rather
    # than falling back to the legacy H&M-only index.
    _unified_flag = os.environ.get("UNIFIED", "").lower() in ("1", "true", "yes")
    _brand_env = os.environ.get("BRAND", "")
    if _unified_flag or _brand_env == UNIFIED_BRAND or not _brand_env:
        brand = UNIFIED_BRAND
    else:
        brand = _brand_env

    # ── Demo guards (same as POST /chat) ──────────────────────────────────────
    if user_id.startswith("anon:"):
        client_ip = request.client.host if request.client else "0.0.0.0"
        engine = deps.get_db_engine()
        if engine is not None:
            from api.demo.guards import (
                check_daily_cap,
                check_daily_cost,
                check_ip_rate_limit,
                record_request,
            )

            ip_ok, ip_retry = check_ip_rate_limit(client_ip, brand, engine)
            if not ip_ok:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(ip_retry)},
                )
            if not check_daily_cap(brand, engine) or not check_daily_cost(brand):
                raise HTTPException(
                    status_code=429,
                    detail="Demo limit reached for today — try again tomorrow.",
                )
            record_request(brand, engine)

    # ── Feature flag ───────────────────────────────────────────────────────────
    if not _feature_enabled(config):
        raise HTTPException(status_code=404, detail="Image style feature is disabled.")

    # ── Content-type validation ────────────────────────────────────────────────
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type '{content_type}'. "
                f"Accepted: {sorted(_ALLOWED_CONTENT_TYPES)}"
            ),
        )

    # ── Read upload bytes with size guard ─────────────────────────────────────
    upload_bytes = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(upload_bytes) > _MAX_UPLOAD_BYTES:
        del upload_bytes
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    # ── Validate actual bytes as an image (PIL decode) ────────────────────────
    try:
        # Register HEIF opener before PIL decode attempt so .heic files work.
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError:
            pass  # HEIF support optional; non-HEIC formats still work

        from PIL import Image

        pil_img = Image.open(io.BytesIO(upload_bytes))
        pil_img.verify()  # checks header without fully decoding
        # Re-open because verify() exhausts the stream
        pil_img = Image.open(io.BytesIO(upload_bytes))
    except Exception as exc:
        del upload_bytes
        logger.debug("Image validation failed: %s", exc)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    # ── Privacy: discard raw bytes immediately after PIL opened the stream ────
    del upload_bytes  # buffer freed; PIL holds no reference to it after open()

    # ── Feature: brand CLIP index must exist ──────────────────────────────────
    if not _brand_index_exists(brand, config):
        del pil_img
        raise HTTPException(
            status_code=404,
            detail=f"Image style feature is not available for brand '{brand}' (index not built).",
        )

    # ── CLIP encode + anchor retrieval ────────────────────────────────────────
    try:
        model_id = _clip_model_id(config)
        candidate_ids = find_anchor_from_image(pil_img, brand, top_k=5, model_id=model_id)
    except RuntimeError as exc:
        logger.error("CLIP encoder error: %s", exc)
        raise HTTPException(status_code=503, detail="Image encoder not available.") from exc
    finally:
        del pil_img  # ensure image bytes are freed regardless of outcome

    if not candidate_ids:
        raise HTTPException(
            status_code=404,
            detail="No matching catalogue items found for the uploaded image.",
        )

    anchor_id = candidate_ids[0]

    # ── Compose outfit variants around the anchor ─────────────────────────────
    catalogue_df = deps.get_catalogue_df()
    retriever = deps.get_retriever()

    from src.config.brand import get_brand_config

    try:
        brand_cfg = get_brand_config()
    except Exception:
        brand_cfg = None

    # Parse optional free-text message for a budget cap, an occasion, and a
    # gender using the same deterministic intent parser as POST /chat — no
    # duplicated regex/keyword logic here.
    occasion = _DEFAULT_OCCASION
    budget_max_inr: int | None = None
    intent_gender: str | None = None
    if message:
        intent = parse_intent(message)
        if intent.occasion is not None:
            occasion = intent.occasion
        budget_max_inr = intent.budget_max_inr
        intent_gender = intent.gender

    gender = _resolve_gender(intent_gender, catalogue_df, anchor_id, brand_cfg)

    # "Owned anchor" feature: the seed resolved from an uploaded photo is an item
    # the user already OWNS, never a catalogue item for sale. owned_anchor=True
    # stamps _owned on the seed in every variant; complements stay shoppable.
    variants = compose_outfit_variants(
        catalogue_df,
        retriever,
        seed_article_id=anchor_id,
        occasion_slug=occasion,
        gender=gender,
        budget_inr=budget_max_inr,
        brand_gender_default=gender,
        owned_anchor=True,
    )

    # ── Grounded rationale generation ─────────────────────────────────────────
    llm = deps.get_llm()
    try:
        rationales = generate_rationales(variants, llm, occasion=occasion, gender=gender)
        for look, rat in zip(variants, rationales):
            look["outfit_rationale"] = rat
    except Exception as exc:
        logger.warning("Rationale generation failed (%s); using template rationale.", exc)
        for look in variants:
            if not look.get("outfit_rationale"):
                look["outfit_rationale"] = template_rationale(look)

    # ── Build response payload ────────────────────────────────────────────────
    outfit_variants = [_build_variant_response(v, brand) for v in variants]
    base = variants[0] if variants else {}
    base_seed = base.get("seed_item")
    base_complements = base.get("complements") or []
    all_base_items = ([base_seed] if base_seed else []) + base_complements

    cart_action = build_cart_action(all_base_items, brand)

    # ── Persist session so follow-up chat turns can resume context ────────────
    # Generate a fresh conversation_id when the client did not supply one.
    cid = conversation_id or str(uuid.uuid4())
    _is_anon = user_id.startswith("anon:")
    try:
        from api.routes.chat import _DEMO_SESSIONS

        store = deps.get_session_store()
        # Compose a minimal session state in the same shape as _persist_result
        # in chat.py so that a follow-up POST /chat or WS /chat/stream can load
        # it and resume with the image outfit as context.
        if _is_anon:
            existing = _DEMO_SESSIONS.get(cid) or {}
        else:
            existing = store.get(cid, user_id) or {}

        existing.update(
            {
                "retrieved_items": all_base_items,
                "look_id": base.get("look_id"),
                "occasion": occasion,
                "look_gender": gender,
                "budget_total_inr": base.get("budget_total_inr"),
                "outfit_rationale": base.get("outfit_rationale"),
                "filters": {"gender": gender},
                "new_items_this_turn": True,
                "anchor_article_id": anchor_id,
                # "Owned anchor": image uploads are always owned by default — a
                # follow-up "Style this" / re-compose must not re-tag this item
                # as buyable. See graph.py outfit_node and src/agents/state.py.
                "anchor_is_owned": True,
            }
        )
        if message:
            existing["last_user_text"] = message
        # Ensure ConversationMemory exists so _build_invoke_state in chat.py
        # does not KeyError on session["_memory"] when resuming this session.
        if "_memory" not in existing:
            from src.memory.conversation import ConversationMemory

            llm = deps.get_llm()
            config = deps.get_config()
            existing["_memory"] = ConversationMemory(llm, config)
        if "messages" not in existing:
            existing["messages"] = []

        if _is_anon:
            _DEMO_SESSIONS[cid] = existing
        else:
            store.set(cid, existing, user_id)
    except Exception as _sess_exc:
        # Session persistence is best-effort; never fail the image response.
        logger.warning("image session persist failed: %s", _sess_exc)

    payload = {
        "conversation_id": cid,
        "anchor_article_id": anchor_id,
        "look_id": base.get("look_id"),
        "occasion": occasion,
        "gender": gender,
        "outfit_rationale": base.get("outfit_rationale"),
        "outfit_variants": [v.model_dump() for v in outfit_variants],
        "items": [ItemSummary.from_agent_item(it).model_dump() for it in all_base_items],
        "cart_url": cart_action.get("cart_url"),
        "item_links": cart_action.get("item_links"),
        "budget_total_inr": base.get("budget_total_inr"),
        "user_text": message,
    }

    return JSONResponse(content=payload)

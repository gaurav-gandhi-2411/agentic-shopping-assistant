from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import api.deps as deps
from api.demo.guards import check_daily_cap, check_daily_cost

router = APIRouter(prefix="/demo", tags=["demo"])

DEMO_CAPPED_MSG = "Demo limit reached for today — try again tomorrow."


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class DemoSessionResponse(BaseModel):
    session_token: str
    ws_ticket: str
    expires_in: int  # seconds (always 3600)
    brand: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/session", response_model=DemoSessionResponse)
async def create_demo_session(request: Request) -> DemoSessionResponse:
    """Create an anonymous demo session token and a paired WebSocket ticket.

    Returns HTTP 404 when DEMO_MODE is not enabled so the endpoint is invisible
    in production deployments.  Returns HTTP 429 when the daily request or cost
    cap for the brand has been reached.
    """
    if os.environ.get("DEMO_MODE", "").lower() not in ("1", "true", "yes"):
        raise HTTPException(status_code=404)

    # client_ip is not used here; per-IP rate limit is enforced at the chat endpoint.
    # The Request object is kept in the signature for future middleware use.
    brand = os.environ.get("BRAND", "hm")
    engine = deps.get_db_engine()

    if engine is not None:
        if not check_daily_cap(brand, engine):
            raise HTTPException(status_code=429, detail=DEMO_CAPPED_MSG)
        if not check_daily_cost(brand):
            raise HTTPException(status_code=429, detail=DEMO_CAPPED_MSG)

    anon_id = f"anon:{uuid.uuid4()}"

    from api.auth import mint_ws_ticket
    from api.demo.session import create_demo_token

    ws_ticket = mint_ws_ticket(anon_id)
    session_token = create_demo_token(anon_id, brand)

    return DemoSessionResponse(
        session_token=session_token,
        ws_ticket=ws_ticket,
        expires_in=3600,
        brand=brand,
    )

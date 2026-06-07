"""Auth routes: POST /auth/ws-ticket.

Issues a short-lived (60 s) single-use nonce that the frontend passes as
``?ticket=<nonce>`` when opening a WebSocket connection.  This avoids
exposing the full JWT in server access logs.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import get_current_user_id_or_demo, mint_ws_ticket

router = APIRouter(tags=["auth"])


@router.post("/auth/ws-ticket")
async def get_ws_ticket(user_id: str = Depends(get_current_user_id_or_demo)) -> dict[str, str]:
    """Mint a 60-second WebSocket authentication ticket.

    Requires a valid Bearer JWT in the Authorization header.  Returns a
    ``ticket`` nonce that may be passed as ``?ticket=<nonce>`` when opening
    ``/chat/stream``.  The ticket is single-use and expires after 60 seconds.
    """
    nonce = mint_ws_ticket(user_id)
    return {"ticket": nonce}

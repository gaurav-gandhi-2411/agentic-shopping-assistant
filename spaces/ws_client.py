"""Async WebSocket client for the /chat/stream endpoint.

Exports:
  stream_turn()  — async generator, one frame dict per yield
  iter_frames()  — synchronous bridge for Streamlit event handlers
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import AsyncIterator, Iterator

import websockets
import websockets.exceptions


async def stream_turn(
    backend_url: str,
    conversation_id: str | None,
    message: str,
) -> AsyncIterator[dict]:
    """Connect to /chat/stream, send a user_message frame, yield server frames.

    Terminates after the terminal frame (done / cancelled / error) or on
    connection close.
    """
    ws_url = backend_url.rstrip("/") + "/chat/stream"
    async with websockets.connect(ws_url) as ws:
        await ws.send(
            json.dumps({
                "type": "user_message",
                "message": message,
                "conversation_id": conversation_id,
            })
        )
        while True:
            try:
                raw = await ws.recv()
            except websockets.exceptions.ConnectionClosed:
                break
            frame = json.loads(raw)
            yield frame
            if frame.get("type") in ("done", "cancelled", "error"):
                break


def iter_frames(
    backend_url: str,
    conversation_id: str | None,
    message: str,
) -> Iterator[dict]:
    """Synchronous bridge around stream_turn for Streamlit's threading model.

    Runs stream_turn in a daemon thread via asyncio.run() and passes frames
    through a Queue so the Streamlit main thread can iterate without blocking
    the event loop.
    """
    q: queue.Queue[dict | None] = queue.Queue()

    async def _run() -> None:
        try:
            async for frame in stream_turn(backend_url, conversation_id, message):
                q.put(frame)
        except Exception as exc:
            q.put({"type": "error", "message": str(exc), "code": "client_error"})
        finally:
            q.put(None)  # sentinel

    t = threading.Thread(target=asyncio.run, args=(_run(),), daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item
    t.join()

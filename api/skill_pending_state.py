"""Mirror staged skill writes into a dedicated SSE event.

When ``skills.write_approval`` is on, ``skill_manage`` does not write — it
stages the write and returns ``{"success": true, "staged": true, ...}``. The
turn continues normally, so without a signal the browser has no idea anything
is waiting for review until the user happens to open the Skills panel.

This module is the live half of that signal, modelled on
:mod:`api.todo_state`: ``api.streaming`` calls :func:`emit_skill_pending_state`
from its ``tool_complete`` paths, and the Skills panel updates mid-turn.

It covers foreground writes only. The background self-improvement review stages
its writes *after* the turn, on a thread with no stream attached — the client
also refreshes on turn end for exactly that case.

Side-effect free by design: it parses a tool result and calls a caller-supplied
``put``. Never raises — a failed mirror must not break tool delivery.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Bumped only for non-additive payload changes.
VERSION = 1

# Single source of truth for the SSE event name; the frontend registers the
# same string in its run-journal event list.
EVENT_NAME = "skill_pending"

# The only tool whose results can stage a skill write.
TOOL_NAME = "skill_manage"


def parse_staged_result(function_result: Any) -> Optional[dict]:
    """Return the staged-write fields from a ``skill_manage`` result, else None.

    Returns None for every non-staged outcome — a normal successful write, an
    error, a non-JSON result, or a multimodal (list) result. Only the gate's
    staging branch sets ``staged: true``.
    """
    if function_result is None:
        return None
    payload: Any = function_result
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8")
        except Exception:
            return None
    if isinstance(payload, str):
        text = payload.strip()
        if not text.startswith("{"):
            return None
        try:
            payload = json.loads(text)
        except Exception:
            return None
    if not isinstance(payload, dict):
        return None
    if payload.get("staged") is not True:
        return None
    pending_id = payload.get("pending_id")
    if not isinstance(pending_id, str) or not pending_id:
        return None
    return {
        "pending_id": pending_id,
        "gist": str(payload.get("gist", "") or ""),
        "version": VERSION,
    }


def emit_skill_pending_state(
    put: Callable[[str, dict], Any],
    *,
    name: Optional[str],
    function_result: Any,
    session_id: Optional[str],
    stream_id: Optional[str],
    source: str = "tool",
) -> bool:
    """Emit a ``skill_pending`` SSE event when a skill write was staged.

    Returns True if an event was emitted. Swallows internal errors — emission
    must never break tool delivery, which is the caller's primary contract.

    The payload carries the pending id and one-line gist only, never the staged
    SKILL.md: the review UI fetches the full diff on demand, so a burst of
    staged writes cannot push large bodies through the stream.
    """
    if name != TOOL_NAME:
        return False
    try:
        staged = parse_staged_result(function_result)
        if staged is None:
            return False
        put(EVENT_NAME, {
            "session_id": session_id,
            "stream_id": stream_id,
            "source": source,
            "ts": time.time(),
            **staged,
        })
        return True
    except Exception:
        logger.debug("Failed to emit skill_pending event", exc_info=True)
        return False

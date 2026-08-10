"""Bridge to the agent's write-approval gate (``skills`` / ``memory``).

Hermes Agent stages agent-authored writes behind ``<subsystem>.write_approval``
(``tools/write_approval.py``). The gate is enforced inside the tool itself, so
it already fires for WebUI sessions — but until this module existed the WebUI
had no way to list, diff, approve, or reject what it staged. Turning the gate on
while using only the browser silently swallowed every skill write.

This module is the read/act half. It deliberately owns no state: the pending
store is a directory of JSON records under ``<hermes_home>/pending/<subsystem>/``
owned by the agent, and the CLI (``/skills pending``) reads the same files. The
semantics here mirror ``hermes_cli/write_approval_commands.py`` exactly —
notably **approve discards a record only when the replay succeeded**, so a write
that fails to apply stays staged instead of vanishing.

Profile scoping is the caller's job, and it is not optional. Both
``write_approval._pending_dir()`` and ``skill_manager_tool._skills_dir()``
resolve through ``hermes_constants.get_hermes_home()`` at call time, which
without a context override returns the *process* home rather than the
per-request profile. Every route in ``api.routes`` that reaches these helpers
wraps the call in ``api.profiles.profile_env_for_active_request``. Getting this
wrong does not raise — it silently reads one profile's pending queue and applies
it to another profile's skills directory.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Write-approval support is optional: a WebUI deployment without the
# hermes-agent source mounted must degrade to "unavailable", not 500. Mirrors
# the graceful-import shape used by api/route_approvals.py.
try:
    from tools import write_approval as _wa

    MEMORY = _wa.MEMORY
    SKILLS = _wa.SKILLS
    _AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without the agent source
    _wa = None
    MEMORY = "memory"
    SKILLS = "skills"
    _AVAILABLE = False

SUBSYSTEMS = (SKILLS, MEMORY)

# Pending ids are minted by the agent as ``uuid4().hex[:8]``. The agent's
# get_pending/discard_pending build a path as ``pending_dir / f"{id}.json"``
# with no sanitisation, which is safe for a CLI-typed id and NOT safe for one
# that arrived in an HTTP body. Validate at the boundary, before any agent call.
_PENDING_ID_RE = re.compile(r"^[0-9a-f]{8}$")


def is_available() -> bool:
    """True when the agent's write-approval module could be imported."""
    return _AVAILABLE


def valid_pending_id(pending_id: Any) -> bool:
    """True when ``pending_id`` is a well-formed agent-minted pending id."""
    return bool(isinstance(pending_id, str) and _PENDING_ID_RE.match(pending_id))


def valid_subsystem(subsystem: Any) -> bool:
    return subsystem in SUBSYSTEMS


def gate_enabled(subsystem: str) -> bool:
    """Whether ``<subsystem>.write_approval`` is on for the active profile.

    Reads through the agent so the WebUI can never disagree with the value the
    tool itself gates on. Fails closed to ``False`` (gate off) to match the
    agent's own default for an unset/unreadable config.
    """
    if not _AVAILABLE or not valid_subsystem(subsystem):
        return False
    try:
        return bool(_wa.write_approval_enabled(subsystem))
    except Exception:
        logger.debug("write_approval_enabled(%s) failed", subsystem, exc_info=True)
        return False


def summarize(record: dict) -> dict:
    """Project a stored record down to the fields the browser needs.

    The full ``payload`` (which can hold an entire SKILL.md) is deliberately
    withheld — the diff endpoint serves that on demand, so listing N pending
    writes does not ship N skill bodies to the client.
    """
    payload = record.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "id": record.get("id", ""),
        "action": record.get("action", "") or payload.get("action", ""),
        "name": payload.get("name", ""),
        "file_path": payload.get("file_path", "") or "",
        "summary": record.get("summary", ""),
        "origin": record.get("origin", "foreground"),
        "created_at": record.get("created_at", 0),
    }


def list_pending(subsystem: str) -> list[dict]:
    """Return browser-safe summaries of every staged write, oldest first."""
    if not _AVAILABLE or not valid_subsystem(subsystem):
        return []
    try:
        return [summarize(r) for r in _wa.list_pending(subsystem)]
    except Exception:
        logger.warning("Failed to list pending %s writes", subsystem, exc_info=True)
        return []


def pending_count(subsystem: str) -> int:
    """Count staged writes.

    Derived from ``list_pending`` rather than the agent's ``pending_count``:
    that helper globs the directory blindly while ``list_pending`` skips
    records it cannot parse, so the two disagree when a record is corrupt.
    The badge must never claim work the list cannot show.
    """
    return len(list_pending(subsystem))


def get_pending(subsystem: str, pending_id: str) -> Optional[dict]:
    """Return the raw stored record, or None. Caller must validate the id."""
    if not _AVAILABLE or not valid_subsystem(subsystem):
        return None
    if not valid_pending_id(pending_id):
        return None
    try:
        return _wa.get_pending(subsystem, pending_id)
    except Exception:
        logger.warning("Failed to read pending %s/%s", subsystem, pending_id, exc_info=True)
        return None


def pending_diff(record: dict) -> str:
    """Full diff/content for a staged skill write.

    Note the agent's ``skill_pending_diff`` is not uniformly a diff: ``create``
    returns the whole new SKILL.md, ``delete``/``remove_file`` return a single
    sentence, and only ``edit``/``patch``/``write_file`` return a unified diff.
    The client branches on ``action`` to render each correctly.
    """
    if not _AVAILABLE:
        return ""
    try:
        return _wa.skill_pending_diff(record) or ""
    except Exception:
        logger.warning("Failed to build pending skill diff", exc_info=True)
        return ""


def discard_pending(subsystem: str, pending_id: str) -> bool:
    """Delete a staged record. False when it was already gone."""
    if not _AVAILABLE or not valid_subsystem(subsystem):
        return False
    if not valid_pending_id(pending_id):
        return False
    try:
        return bool(_wa.discard_pending(subsystem, pending_id))
    except Exception:
        logger.warning("Failed to discard pending %s/%s", subsystem, pending_id, exc_info=True)
        return False


def _apply_one(subsystem: str, record: dict) -> tuple[bool, str]:
    """Replay one staged write. Returns ``(ok, error_message)``.

    Mirrors ``hermes_cli/write_approval_commands.py:_apply_one``. Memory is
    intentionally not wired yet: its store is owned by the live agent session,
    so a browser-side apply needs a resolved memory store that this bridge does
    not have. The subsystem plumbing is in place for that follow-up.
    """
    payload = record.get("payload") or {}
    if subsystem != SKILLS:
        return False, f"{subsystem} approval is not supported from the WebUI yet"
    try:
        import json

        from tools.skill_manager_tool import apply_skill_pending

        result = json.loads(apply_skill_pending(payload))
        return bool(result.get("success")), str(result.get("error", "") or "")
    except Exception as e:
        logger.warning("Failed to apply pending %s write", subsystem, exc_info=True)
        return False, str(e)


def approve(subsystem: str, pending_id: str) -> dict:
    """Apply one staged write, or all of them when ``pending_id == 'all'``.

    Returns ``{applied, failures, remaining}``. A record whose replay failed is
    **kept staged** (the CLI does the same) — the skill it targets may have been
    renamed or deleted since staging, and dropping the record would lose the
    change with no way to retry.
    """
    if not _AVAILABLE or not valid_subsystem(subsystem):
        return {"applied": 0, "failures": [], "remaining": 0}

    bulk = isinstance(pending_id, str) and pending_id.lower() == "all"
    if bulk:
        targets = _wa.list_pending(subsystem)
    else:
        record = get_pending(subsystem, pending_id)
        targets = [record] if record else []

    applied = 0
    failures: list[dict] = []
    for record in targets:
        rid = record.get("id", "")
        ok, message = _apply_one(subsystem, record)
        if ok:
            discard_pending(subsystem, rid)
            applied += 1
        else:
            failures.append({"id": rid, "error": message or "apply failed"})

    return {
        "applied": applied,
        "failures": failures,
        "remaining": pending_count(subsystem),
    }


def reject(subsystem: str, pending_id: str) -> dict:
    """Discard one staged write without applying it, or all of them."""
    if not _AVAILABLE or not valid_subsystem(subsystem):
        return {"rejected": 0, "remaining": 0}

    if isinstance(pending_id, str) and pending_id.lower() == "all":
        ids = [r.get("id", "") for r in _wa.list_pending(subsystem)]
    else:
        ids = [pending_id]

    rejected = sum(1 for rid in ids if discard_pending(subsystem, rid))
    return {"rejected": rejected, "remaining": pending_count(subsystem)}


def state(subsystem: str, *, include_items: bool = True) -> dict:
    """Bundle everything a client needs for one round trip."""
    available = _AVAILABLE and valid_subsystem(subsystem)
    items = list_pending(subsystem) if (available and include_items) else []
    payload = {
        "available": available,
        "gate_enabled": gate_enabled(subsystem),
        "count": len(items) if include_items else pending_count(subsystem),
    }
    if include_items:
        payload["items"] = items
    return payload


def with_profile(purpose: str) -> Callable:
    """Return the profile context manager these calls must run inside.

    Split out so the reason is stated once: without it, ``get_hermes_home()``
    resolves the process home instead of the request's profile.
    """
    from api.profiles import profile_env_for_active_request

    return profile_env_for_active_request(purpose, logger_override=logger)

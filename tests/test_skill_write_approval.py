"""Skill write-approval review surface in the WebUI.

Hermes Agent stages agent-authored skill writes behind ``skills.write_approval``
(``tools/write_approval.py``). The gate lives inside ``skill_manage`` itself, so
it already fired for WebUI sessions — but the WebUI had no way to list, diff,
approve, or reject what it staged, which meant turning the gate on while using
only the browser silently swallowed every skill write.

Covers:
  - the bridge's list/count/diff/approve/reject semantics, including the two
    that are easy to get wrong: a failed apply KEEPS its record staged, and a
    path-traversal id never reaches the filesystem;
  - profile isolation (records staged under profile A must be invisible from B);
  - the config setter writing the same key the CLI writes;
  - the ``skill_pending`` SSE emitter, and a static assertion that BOTH
    ``tool_complete`` sites in api/streaming.py call it;
  - frontend wiring and locale parity for the new copy.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.conftest import HERMES_AGENT  # noqa: E402

# APPENDED, never prepended: the hermes-agent checkout has its own top-level
# `tests/` package, so putting it ahead of REPO_ROOT would let a later
# `tests.*` import in any other module resolve into the wrong checkout. The
# agent-side modules this file needs (`tools.*`) have no WebUI counterpart, so
# the tail of sys.path is enough to reach them without shadowing anything.
if HERMES_AGENT and str(HERMES_AGENT) not in sys.path:
    sys.path.append(str(HERMES_AGENT))

try:
    from tools import write_approval as _wa  # noqa: F401

    AGENT_AVAILABLE = True
except Exception:
    AGENT_AVAILABLE = False

requires_agent = pytest.mark.skipif(
    not AGENT_AVAILABLE,
    reason="hermes-agent source not importable; write-approval is optional by design",
)


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Bridge behaviour (needs the agent source)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    """Point the agent's home at a temp dir for the duration of one test.

    Also resets ``skill_manager_tool.SKILLS_DIR`` to its import-time value.
    ``_skills_dir()`` honours an explicitly patched module global in preference
    to ``get_hermes_home()``, so an earlier test in the same session that
    patched it would otherwise silently redirect writes made here — this test
    module must not depend on suite ordering.
    """
    home = tmp_path / "hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    if AGENT_AVAILABLE:
        from tools import skill_manager_tool

        monkeypatch.setattr(
            skill_manager_tool, "SKILLS_DIR",
            skill_manager_tool._SKILLS_DIR_AT_IMPORT, raising=False,
        )
    return home


def _stage(subsystem, payload, summary="staged write", origin="foreground"):
    from tools import write_approval as wa

    return wa.stage_write(subsystem, payload, summary=summary, origin=origin)


def _write_skill(home: pathlib.Path, name: str, body: str) -> pathlib.Path:
    d = home / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    md = d / "SKILL.md"
    md.write_text(
        f"---\nname: {name}\ndescription: test skill\n---\n\n{body}\n", encoding="utf-8"
    )
    return md


@requires_agent
class TestBridgeRoundTrip:
    def test_staged_write_is_listed_counted_and_diffable(self, hermes_home):
        from api import write_approval_bridge as wab

        assert wab.pending_count(wab.SKILLS) == 0
        record = _stage(
            "skills",
            {"action": "create", "name": "demo", "content": "---\nname: demo\ndescription: d\n---\n\nBody\n"},
            summary="create 'demo'",
        )

        items = wab.list_pending(wab.SKILLS)
        assert [i["id"] for i in items] == [record["id"]]
        assert items[0]["action"] == "create"
        assert items[0]["name"] == "demo"
        assert items[0]["summary"] == "create 'demo'"
        assert wab.pending_count(wab.SKILLS) == 1

        # create returns the whole new SKILL.md, not a diff -- the client
        # branches on action for exactly this reason.
        diff = wab.pending_diff(wab.get_pending(wab.SKILLS, record["id"]))
        assert "name: demo" in diff

    def test_list_never_leaks_the_staged_body(self, hermes_home):
        from api import write_approval_bridge as wab

        secret = "SUPER-SECRET-SKILL-BODY"
        _stage("skills", {"action": "create", "name": "demo", "content": secret})

        assert secret not in json.dumps(wab.list_pending(wab.SKILLS))

    def test_approve_applies_then_discards(self, hermes_home):
        from api import write_approval_bridge as wab

        content = "---\nname: fresh\ndescription: made by approval\n---\n\nHello\n"
        record = _stage("skills", {"action": "create", "name": "fresh", "content": content})

        result = wab.approve(wab.SKILLS, record["id"])

        assert result["applied"] == 1
        assert result["failures"] == []
        assert result["remaining"] == 0
        # Observable behaviour, not the response text: the skill now exists.
        assert (hermes_home / "skills" / "fresh" / "SKILL.md").exists()

    def test_failed_approve_keeps_the_record_staged(self, hermes_home):
        from api import write_approval_bridge as wab

        # patch a skill that does not exist -> apply fails
        record = _stage(
            "skills",
            {"action": "patch", "name": "ghost", "old_string": "a", "new_string": "b"},
        )

        result = wab.approve(wab.SKILLS, record["id"])

        assert result["applied"] == 0
        assert len(result["failures"]) == 1
        assert result["failures"][0]["id"] == record["id"]
        # The point of the test: a change that could not be applied must remain
        # available to retry, not vanish.
        assert result["remaining"] == 1
        assert wab.get_pending(wab.SKILLS, record["id"]) is not None

    def test_reject_discards_without_applying(self, hermes_home):
        from api import write_approval_bridge as wab

        _write_skill(hermes_home, "keeper", "original body")
        record = _stage(
            "skills",
            {"action": "edit", "name": "keeper", "content": "---\nname: keeper\ndescription: x\n---\n\nREPLACED\n"},
        )

        result = wab.reject(wab.SKILLS, record["id"])

        assert result["rejected"] == 1
        assert result["remaining"] == 0
        on_disk = (hermes_home / "skills" / "keeper" / "SKILL.md").read_text(encoding="utf-8")
        assert "original body" in on_disk
        assert "REPLACED" not in on_disk

    def test_approve_all_is_bulk_and_reports_partial_failure(self, hermes_home):
        from api import write_approval_bridge as wab

        good_a = _stage("skills", {"action": "create", "name": "aa", "content": "---\nname: aa\ndescription: d\n---\n\nA\n"})
        bad = _stage("skills", {"action": "patch", "name": "ghost", "old_string": "a", "new_string": "b"})
        good_b = _stage("skills", {"action": "create", "name": "bb", "content": "---\nname: bb\ndescription: d\n---\n\nB\n"})

        result = wab.approve(wab.SKILLS, "all")

        assert result["applied"] == 2
        assert [f["id"] for f in result["failures"]] == [bad["id"]]
        # Partial failure is never reported as clean success.
        assert result["remaining"] == 1
        assert wab.get_pending(wab.SKILLS, good_a["id"]) is None
        assert wab.get_pending(wab.SKILLS, good_b["id"]) is None

    def test_reject_all_clears_the_queue(self, hermes_home):
        from api import write_approval_bridge as wab

        for n in ("one", "two", "three"):
            _stage("skills", {"action": "create", "name": n, "content": f"---\nname: {n}\ndescription: d\n---\n\nX\n"})

        result = wab.reject(wab.SKILLS, "all")

        assert result["rejected"] == 3
        assert result["remaining"] == 0

    def test_resolving_an_already_resolved_record_is_not_an_error(self, hermes_home):
        from api import write_approval_bridge as wab

        record = _stage("skills", {"action": "create", "name": "gone", "content": "---\nname: gone\ndescription: d\n---\n\nX\n"})
        wab.reject(wab.SKILLS, record["id"])

        # Simulates the CLI resolving it first: nothing applied, nothing failed.
        again = wab.approve(wab.SKILLS, record["id"])
        assert again["applied"] == 0
        assert again["failures"] == []


@requires_agent
class TestBridgeGuards:
    @pytest.mark.parametrize(
        "bad_id",
        ["../../etc/passwd", "..%2f..%2fetc", "a" * 9, "ABCDEF12", "", "12345678/x", "1234567"],
    )
    def test_path_traversal_and_malformed_ids_are_rejected(self, hermes_home, bad_id):
        from api import write_approval_bridge as wab

        assert wab.valid_pending_id(bad_id) is False
        assert wab.get_pending(wab.SKILLS, bad_id) is None
        assert wab.discard_pending(wab.SKILLS, bad_id) is False

    def test_wellformed_id_shape_is_accepted(self, hermes_home):
        from api import write_approval_bridge as wab

        record = _stage("skills", {"action": "create", "name": "ok", "content": "x"})
        assert wab.valid_pending_id(record["id"]) is True

    def test_unknown_subsystem_is_inert(self, hermes_home):
        from api import write_approval_bridge as wab

        assert wab.list_pending("bogus") == []
        assert wab.pending_count("bogus") == 0
        assert wab.gate_enabled("bogus") is False
        assert wab.approve("bogus", "all")["applied"] == 0

    def test_count_agrees_with_list_when_a_record_is_corrupt(self, hermes_home):
        from api import write_approval_bridge as wab

        _stage("skills", {"action": "create", "name": "good", "content": "x"})
        corrupt = hermes_home / "pending" / "skills" / "deadbeef.json"
        corrupt.write_text("{not json", encoding="utf-8")

        # The agent's own pending_count() globs blindly and would say 2 here;
        # the badge must never claim work the list cannot show.
        assert wab.pending_count(wab.SKILLS) == len(wab.list_pending(wab.SKILLS)) == 1


@requires_agent
class TestProfileIsolation:
    def test_records_staged_in_one_home_are_invisible_from_another(self, tmp_path, monkeypatch):
        from api import write_approval_bridge as wab

        home_a = tmp_path / "a"
        home_b = tmp_path / "b"
        for h in (home_a, home_b):
            (h / "skills").mkdir(parents=True)

        monkeypatch.setenv("HERMES_HOME", str(home_a))
        record = _stage("skills", {"action": "create", "name": "only-in-a", "content": "x"})
        assert wab.pending_count(wab.SKILLS) == 1

        monkeypatch.setenv("HERMES_HOME", str(home_b))
        assert wab.pending_count(wab.SKILLS) == 0
        assert wab.get_pending(wab.SKILLS, record["id"]) is None
        # ...and it cannot be resolved from the wrong profile either.
        assert wab.approve(wab.SKILLS, record["id"])["applied"] == 0

        monkeypatch.setenv("HERMES_HOME", str(home_a))
        assert wab.pending_count(wab.SKILLS) == 1


@requires_agent
class TestConfigSetter:
    def test_setter_writes_the_key_the_agent_gates_on(self, hermes_home, monkeypatch):
        import api.config as api_config
        from tools import write_approval as wa

        config_path = hermes_home / "config.yaml"
        monkeypatch.setenv("HERMES_CONFIG_PATH", str(config_path))

        assert wa.write_approval_enabled("skills") is False
        api_config.set_write_approval("skills", True)

        import yaml

        written = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert written["skills"]["write_approval"] is True
        # The agent must observe the change, not just the file.
        assert wa.write_approval_enabled("skills") is True

        api_config.set_write_approval("skills", False)
        assert wa.write_approval_enabled("skills") is False

    def test_setter_preserves_sibling_keys_in_the_section(self, hermes_home, monkeypatch):
        import api.config as api_config
        import yaml

        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            yaml.safe_dump({"skills": {"disabled": ["a", "b"], "guard_agent_created": True}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_CONFIG_PATH", str(config_path))

        api_config.set_write_approval("skills", True)

        written = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert written["skills"]["disabled"] == ["a", "b"]
        assert written["skills"]["guard_agent_created"] is True
        assert written["skills"]["write_approval"] is True

    def test_unknown_subsystem_raises_rather_than_writing_a_dead_key(self, hermes_home, monkeypatch):
        import api.config as api_config

        monkeypatch.setenv("HERMES_CONFIG_PATH", str(hermes_home / "config.yaml"))
        with pytest.raises(ValueError):
            api_config.set_write_approval("skillz", True)


# ═══════════════════════════════════════════════════════════════════════════
# 2. SSE emitter (pure parsing -- no agent source needed)
# ═══════════════════════════════════════════════════════════════════════════

class TestSkillPendingEmitter:
    def _capture(self):
        events = []
        return events, lambda name, data: events.append((name, data))

    def test_emits_for_a_staged_skill_manage_result(self):
        from api.skill_pending_state import EVENT_NAME, emit_skill_pending_state

        events, put = self._capture()
        result = json.dumps({
            "success": True, "staged": True, "pending_id": "a1b2c3d4",
            "gist": "create 'demo' — a demo skill (2 KB)",
            "message": "Staged for approval",
        })

        assert emit_skill_pending_state(
            put, name="skill_manage", function_result=result,
            session_id="s1", stream_id="st1",
        ) is True

        assert len(events) == 1
        name, data = events[0]
        assert name == EVENT_NAME
        assert data["pending_id"] == "a1b2c3d4"
        assert data["gist"].startswith("create 'demo'")
        assert data["session_id"] == "s1"
        assert data["stream_id"] == "st1"

    def test_payload_never_carries_the_skill_body(self):
        from api.skill_pending_state import emit_skill_pending_state

        events, put = self._capture()
        emit_skill_pending_state(
            put, name="skill_manage",
            function_result=json.dumps({
                "staged": True, "pending_id": "a1b2c3d4", "gist": "g",
                "content": "---\nname: x\n---\nHUGE BODY",
            }),
            session_id=None, stream_id=None,
        )
        assert "HUGE BODY" not in json.dumps(events[0][1])

    @pytest.mark.parametrize("name", ["todo", "terminal", "skills_list", None, ""])
    def test_ignores_every_other_tool(self, name):
        from api.skill_pending_state import emit_skill_pending_state

        events, put = self._capture()
        assert emit_skill_pending_state(
            put, name=name,
            function_result=json.dumps({"staged": True, "pending_id": "a1b2c3d4"}),
            session_id=None, stream_id=None,
        ) is False
        assert events == []

    @pytest.mark.parametrize("result", [
        json.dumps({"success": True, "name": "demo"}),          # normal write
        json.dumps({"success": False, "error": "nope"}),        # tool error
        json.dumps({"staged": False, "pending_id": "a1b2c3d4"}),
        json.dumps({"staged": True}),                            # no id
        "Skill 'demo' created.",                                 # plain prose
        "{truncated preview...",                                 # old-build preview
        None,
        [{"type": "text", "text": "multimodal"}],
    ])
    def test_non_staged_results_emit_nothing(self, result):
        from api.skill_pending_state import emit_skill_pending_state

        events, put = self._capture()
        assert emit_skill_pending_state(
            put, name="skill_manage", function_result=result,
            session_id=None, stream_id=None,
        ) is False
        assert events == []

    def test_a_failing_put_never_breaks_tool_delivery(self):
        from api.skill_pending_state import emit_skill_pending_state

        def boom(_name, _data):
            raise RuntimeError("SSE queue is gone")

        assert emit_skill_pending_state(
            boom, name="skill_manage",
            function_result=json.dumps({"staged": True, "pending_id": "a1b2c3d4"}),
            session_id=None, stream_id=None,
        ) is False


# ═══════════════════════════════════════════════════════════════════════════
# 3. Wiring (static analysis -- no agent source, no server)
# ═══════════════════════════════════════════════════════════════════════════

class TestStreamingWiring:
    def test_both_tool_complete_paths_mirror_staged_skill_writes(self):
        """api/streaming.py has two tool_complete emitters.

        emit_todo_state is called from both; hooking only one would make the
        live signal fire on the structured path and vanish on the legacy one.
        """
        src = _read("api/streaming.py")
        # Both imports are bare names (no call parens), so these count call
        # sites only.
        todo_calls = src.count("emit_todo_state(")
        skill_calls = src.count("emit_skill_pending_state(")
        assert todo_calls >= 2
        assert skill_calls == todo_calls, (
            f"emit_skill_pending_state wired at {skill_calls} sites but "
            f"emit_todo_state at {todo_calls}: a tool_complete path was missed"
        )

    def test_emitter_is_imported(self):
        assert "from api.skill_pending_state import emit_skill_pending_state" in _read("api/streaming.py")


class TestRoutesWiring:
    ROUTES = None

    @classmethod
    def setup_class(cls):
        cls.ROUTES = _read("api/routes.py")

    @pytest.mark.parametrize("path", [
        "/api/skills/pending",
        "/api/skills/pending/count",
        "/api/skills/pending/diff",
        "/api/skills/pending/approve",
        "/api/skills/pending/reject",
        "/api/skills/write-approval",
    ])
    def test_endpoint_is_registered(self, path):
        assert f'parsed.path == "{path}"' in self.ROUTES

    def test_every_pending_route_runs_inside_the_profile_context(self):
        """Both the pending store and the skills dir resolve through
        get_hermes_home() at call time, so a route that skips the profile
        context reads one profile's queue and writes another's skills."""
        src = self.ROUTES
        for marker in (
            '"/api/skills/pending"',
            '"/api/skills/pending/count"',
            '"/api/skills/pending/diff"',
        ):
            start = src.index(marker)
            block = src[start:start + 900]
            assert "with wab.with_profile(" in block, f"{marker} escapes the profile context"

        for handler in ("_handle_skill_pending_resolve", "_handle_skill_write_approval_toggle"):
            start = src.index(f"def {handler}(")
            block = src[start:start + 2600]
            assert "with wab.with_profile(" in block, f"{handler} escapes the profile context"

    def test_diff_route_validates_the_id_before_touching_disk(self):
        src = self.ROUTES
        start = src.index('"/api/skills/pending/diff"')
        block = src[start:start + 700]
        assert block.index("valid_pending_id") < block.index("get_pending")

    def test_approve_handler_validates_the_id(self):
        src = self.ROUTES
        start = src.index("def _handle_skill_pending_resolve(")
        block = src[start:start + 2600]
        assert "valid_pending_id" in block
        assert "is_available()" in block


class TestSessionYoloDoesNotBypassTheGate:
    def test_no_yolo_reference_in_the_write_approval_path(self):
        """Session YOLO belongs to the dangerous-command approval queue. This
        gate is config-level and also covers background writes that have no
        session, so YOLO must never short-circuit it."""
        bridge = _read("api/write_approval_bridge.py")
        emitter = _read("api/skill_pending_state.py")
        assert "yolo" not in bridge.lower()
        assert "yolo" not in emitter.lower()


class TestFrontendWiring:
    @classmethod
    def setup_class(cls):
        cls.INDEX = _read("static/index.html")
        cls.PANELS = _read("static/panels.js")
        cls.MESSAGES = _read("static/messages.js")
        cls.CSS = _read("static/style.css")

    def test_pending_group_is_a_sibling_of_the_skills_list(self):
        """renderSkills() clears #skillsList on every search keystroke, so a
        pending group nested inside it would be destroyed by typing."""
        idx = self.INDEX
        pending_at = idx.index('id="skillsPending"')
        list_at = idx.index('id="skillsList"')
        assert pending_at < list_at
        # ...and the list render must not touch the pending container.
        start = self.PANELS.index("function renderSkills(skills)")
        body = self.PANELS[start:start + 2600]
        assert "skillsPending" not in body

    def test_search_filter_does_not_rerender_the_pending_group(self):
        start = self.PANELS.index("function filterSkills()")
        body = self.PANELS[start:start + 200]
        assert "renderSkillsPending" not in body

    def test_pending_rows_open_in_the_shared_main_view(self):
        panels = self.PANELS
        assert "function openSkillPending(" in panels
        assert "_renderSkillPendingDetail" in panels
        assert "skillDetailBody" in panels[panels.index("function _renderSkillPendingDetail("):]

    def test_header_buttons_have_a_pending_mode(self):
        panels = self.PANELS
        start = panels.index("function _setSkillHeaderButtons(")
        block = panels[start:start + 1200]
        assert "mode === 'pending'" in block
        assert "show(approveBtn)" in block
        assert "show(rejectBtn)" in block
        assert 'id="btnApproveSkillPending"' in self.INDEX
        assert 'id="btnRejectSkillPending"' in self.INDEX

    def test_create_renders_as_markdown_and_patch_renders_as_a_diff(self):
        """skill_pending_diff() is not uniformly a diff: create returns the
        whole new SKILL.md. Routing both through one branch would show a
        create as a broken diff."""
        panels = self.PANELS
        start = panels.index("function _renderSkillPendingDetail(")
        block = panels[start:start + 2600]
        assert "action === 'create'" in block
        assert "_skillMarkdownHtml" in block
        assert "_colorDiffLines" in block
        assert "action === 'delete'" in block

    def test_deletion_shows_what_would_be_lost(self):
        panels = self.PANELS
        assert "_loadSkillPendingLostContent" in panels
        start = panels.index("async function _loadSkillPendingLostContent(")
        block = panels[start:start + 900]
        assert "/api/skills/content?name=" in block

    def test_both_selection_sets_are_cleared_together(self):
        """Real skills and staged writes share one main view; clearing only
        one selection set leaves two rows looking selected."""
        panels = self.PANELS
        assert "function _clearSkillSelection()" in panels
        clear = panels[panels.index("function _clearSkillSelection()"):][:400]
        assert ".skill-item.active" in clear
        assert ".skill-pending-item.active" in clear
        for fn in ("async function openSkill(", "async function openSkillPending("):
            block = panels[panels.index(fn):][:600]
            assert "_clearSkillSelection()" in block

    def test_approving_refreshes_the_skill_list(self):
        panels = self.PANELS
        start = panels.index("async function resolveSkillPending(")
        block = panels[start:start + 2600]
        assert "_skillsData = null" in block
        assert "loadSkills()" in block
        assert "loadSkillsPending()" in block

    def test_failed_approve_does_not_claim_success(self):
        panels = self.PANELS
        start = panels.index("async function resolveSkillPending(")
        block = panels[start:start + 2600]
        assert "failures" in block
        # the success/clear path must be gated on there being no failures
        assert "!failures.length" in block

    def test_profile_switch_drops_the_previous_queue(self):
        panels = self.PANELS
        start = panels.index("async function _profileSwitchPanelLoad()")
        block = panels[start:start + 1400]
        assert "_skillsPendingState = null" in block
        assert "loadSkillsPending()" in block

    def test_live_event_listener_and_journal_registration(self):
        messages = self.MESSAGES
        assert "source.addEventListener('skill_pending'" in messages
        journal_start = messages.index("for(const _runJournalEventName of [")
        journal = messages[journal_start:journal_start + 700]
        assert "'skill_pending'" in journal

    def test_turn_end_refresh_covers_background_review_writes(self):
        """Background-review writes are staged after the turn on a thread with
        no stream, so the live event never fires for them."""
        messages = self.MESSAGES
        start = messages.index("source.addEventListener('done'")
        block = messages[start:start + 1400]
        assert "loadSkillsPending()" in block

    def test_badges_exist_on_both_desktop_and_mobile_tabs(self):
        idx = self.INDEX
        assert 'id="skillsPendingBadgeDesktop"' in idx
        assert 'id="skillsPendingBadgeMobile"' in idx
        assert ".skills-pending-badge{" in self.CSS
        panels = self.PANELS
        badge = panels[panels.index("function _updateSkillsPendingBadge()"):][:700]
        assert "skillsPendingBadgeDesktop" in badge
        assert "skillsPendingBadgeMobile" in badge

    def test_settings_toggle_hides_itself_when_the_gate_is_unavailable(self):
        panels = self.PANELS
        start = panels.index("async function _initSkillWriteApprovalToggle()")
        block = panels[start:start + 1600]
        assert "state.available" in block
        assert "display = 'none'" in block
        assert "/api/skills/write-approval" in block
        assert 'id="settingsSkillWriteApproval"' in self.INDEX


# ═══════════════════════════════════════════════════════════════════════════
# 4. Locale parity
# ═══════════════════════════════════════════════════════════════════════════

NEW_KEYS = [
    "skill_pending_heading",
    "skill_pending_approve",
    "skill_pending_reject",
    "skill_pending_approve_all",
    "skill_pending_reject_all",
    "skill_pending_origin_agent",
    "skill_pending_origin_background",
    "skill_pending_badge_title",
    "skill_pending_action_create",
    "skill_pending_action_edit",
    "skill_pending_action_patch",
    "skill_pending_action_delete",
    "skill_pending_action_write_file",
    "skill_pending_action_remove_file",
    "skill_pending_delete_warning",
    "skill_pending_no_change",
    "skill_pending_current_content",
    "skill_pending_staged_toast",
    "skill_pending_approved",
    "skill_pending_rejected",
    "skill_pending_failed",
    "skill_write_approval_on",
    "skill_write_approval_off",
    "skill_write_approval_failed",
    "settings_label_skill_write_approval",
    "settings_desc_skill_write_approval",
]


class TestLocaleParity:
    @classmethod
    def setup_class(cls):
        cls.SRC = _read("static/i18n.js")
        # Count locale blocks off an anchor key that predates this change.
        cls.BLOCKS = len(re.findall(r"^\s*skill_metadata:", cls.SRC, re.MULTILINE))

    def test_repo_still_has_the_expected_number_of_locale_blocks(self):
        assert self.BLOCKS >= 14

    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_key_is_present_in_every_locale_block(self, key):
        found = len(re.findall(rf"^\s*{key}:", self.SRC, re.MULTILINE))
        assert found == self.BLOCKS, (
            f"{key} appears in {found} of {self.BLOCKS} locale blocks"
        )

    @pytest.mark.parametrize("key", [
        "skill_pending_heading",
        "skill_pending_badge_title",
        "skill_pending_staged_toast",
        "skill_pending_approved",
        "skill_pending_rejected",
        "skill_pending_failed",
        "skill_write_approval_failed",
    ])
    def test_placeholder_keys_keep_their_placeholder_in_every_locale(self, key):
        values = re.findall(rf"^\s*{key}:\s*'(.*)',$", self.SRC, re.MULTILINE)
        assert len(values) == self.BLOCKS
        for value in values:
            assert "{0}" in value, f"{key} lost its {{0}} placeholder: {value!r}"


class TestAffectedNames:
    def test_batch_lists_each_skill_once_in_op_order(self):
        from api import write_approval_bridge as wab

        item = wab.summarize({
            "id": "x",
            "action": "batch",
            "summary": "batch(3 ops: patch, create, patch) on alpha, beta",
            "payload": {"action": "batch", "operations": [
                {"action": "patch", "name": "beta"},
                {"action": "create", "name": "alpha"},
                {"action": "patch", "name": "beta"},
            ]},
        })
        assert item["names"] == ["beta", "alpha"]
        assert item["name"] == ""

    def test_batch_op_without_name_uses_legacy_top_level_name(self):
        from api import write_approval_bridge as wab

        item = wab.summarize({"payload": {"action": "batch", "name": "legacy", "operations": [
            {"action": "patch"}, {"action": "patch", "name": "other"},
        ]}})
        assert item["names"] == ["legacy", "other"]

    def test_single_op_names_its_skill(self):
        from api import write_approval_bridge as wab

        assert wab.summarize({"payload": {"action": "patch", "name": "demo"}})["names"] == ["demo"]
        assert wab.summarize({"payload": "not-a-dict"})["names"] == []

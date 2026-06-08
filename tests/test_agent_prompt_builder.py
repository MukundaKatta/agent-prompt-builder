"""Tests for agent_prompt_builder."""

from __future__ import annotations

import pytest

from agent_prompt_builder import AgentPromptBuilder, PromptSection

# ---------------------------------------------------------------------------
# PromptSection
# ---------------------------------------------------------------------------


def test_section_defaults():
    s = PromptSection(name="role", content="You are helpful.")
    assert s.enabled is True
    assert s.order == 0
    assert s.metadata == {}


def test_section_to_dict():
    s = PromptSection(name="role", content="text", enabled=False, order=5)
    d = s.to_dict()
    assert d["name"] == "role"
    assert d["content"] == "text"
    assert d["enabled"] is False
    assert d["order"] == 5


def test_section_to_dict_metadata_copy():
    s = PromptSection(name="x", content="y", metadata={"k": "v"})
    d = s.to_dict()
    d["metadata"]["k"] = "mutated"
    assert s.metadata["k"] == "v"


# ---------------------------------------------------------------------------
# Empty builder
# ---------------------------------------------------------------------------


def test_empty_render():
    assert AgentPromptBuilder().render() == ""


def test_empty_count():
    assert AgentPromptBuilder().count() == 0


def test_empty_enabled_count():
    assert AgentPromptBuilder().enabled_count() == 0


def test_empty_names():
    assert AgentPromptBuilder().names() == []


def test_empty_enabled_names():
    assert AgentPromptBuilder().enabled_names() == []


def test_empty_len():
    assert len(AgentPromptBuilder()) == 0


def test_repr_empty():
    r = repr(AgentPromptBuilder())
    assert "sections=0" in r
    assert "enabled=0" in r


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------


def test_add_single():
    b = AgentPromptBuilder().add("role", "You are helpful.")
    assert b.has("role")
    assert b.render() == "You are helpful."


def test_add_multiple_renders_in_insertion_order_when_same_order():
    b = AgentPromptBuilder()
    b.add("a", "A", order=0)
    b.add("b", "B", order=0)
    b.add("c", "C", order=0)
    assert b.render() == "A\n\nB\n\nC"


def test_add_order_respected():
    b = AgentPromptBuilder()
    b.add("late", "Late", order=10)
    b.add("early", "Early", order=1)
    assert b.render() == "Early\n\nLate"


def test_add_auto_order_increases():
    b = AgentPromptBuilder()
    b.add("a", "A")
    b.add("b", "B")
    b.add("c", "C")
    assert b.render() == "A\n\nB\n\nC"


def test_add_duplicate_raises():
    b = AgentPromptBuilder().add("role", "text")
    with pytest.raises(ValueError, match="already exists"):
        b.add("role", "other")


def test_add_returns_self():
    b = AgentPromptBuilder()
    result = b.add("role", "text")
    assert result is b


def test_add_with_metadata():
    b = AgentPromptBuilder().add("x", "y", metadata={"source": "user"})
    assert b.get("x").metadata == {"source": "user"}


def test_add_metadata_is_copy():
    meta = {"k": "v"}
    b = AgentPromptBuilder().add("x", "y", metadata=meta)
    meta["extra"] = True
    assert "extra" not in b.get("x").metadata


# ---------------------------------------------------------------------------
# add_or_replace()
# ---------------------------------------------------------------------------


def test_add_or_replace_new():
    b = AgentPromptBuilder().add_or_replace("x", "Hello")
    assert b.render() == "Hello"


def test_add_or_replace_existing():
    b = AgentPromptBuilder().add("x", "Old").add_or_replace("x", "New")
    assert b.render() == "New"


def test_add_or_replace_preserves_order_position():
    b = AgentPromptBuilder()
    b.add("a", "A", order=0)
    b.add("b", "B", order=5)
    b.add_or_replace("a", "A2")  # replaces, keeps order=0
    assert b.names() == ["a", "b"]
    assert b.render() == "A2\n\nB"


def test_add_or_replace_returns_self():
    b = AgentPromptBuilder()
    result = b.add_or_replace("x", "y")
    assert result is b


# ---------------------------------------------------------------------------
# set_content()
# ---------------------------------------------------------------------------


def test_set_content():
    b = AgentPromptBuilder().add("role", "Old")
    b.set_content("role", "New")
    assert b.render() == "New"


def test_set_content_missing_raises():
    b = AgentPromptBuilder()
    with pytest.raises(KeyError):
        b.set_content("missing", "x")


def test_set_content_returns_self():
    b = AgentPromptBuilder().add("x", "y")
    assert b.set_content("x", "z") is b


# ---------------------------------------------------------------------------
# set_order()
# ---------------------------------------------------------------------------


def test_set_order():
    b = AgentPromptBuilder()
    b.add("a", "A", order=0)
    b.add("b", "B", order=1)
    b.set_order("b", -1)
    assert b.render() == "B\n\nA"


def test_set_order_missing_raises():
    b = AgentPromptBuilder()
    with pytest.raises(KeyError):
        b.set_order("missing", 5)


# ---------------------------------------------------------------------------
# enable() / disable() / set_enabled()
# ---------------------------------------------------------------------------


def test_disable_excludes_from_render():
    b = AgentPromptBuilder().add("a", "A").add("b", "B")
    b.disable("b")
    assert b.render() == "A"


def test_enable_after_disable():
    b = AgentPromptBuilder().add("a", "A").add("b", "B")
    b.disable("b")
    b.enable("b")
    assert "B" in b.render()


def test_set_enabled_false():
    b = AgentPromptBuilder().add("x", "X")
    b.set_enabled("x", False)
    assert b.render() == ""


def test_set_enabled_true():
    b = AgentPromptBuilder().add("x", "X", enabled=False)
    b.set_enabled("x", True)
    assert b.render() == "X"


def test_disable_missing_raises():
    b = AgentPromptBuilder()
    with pytest.raises(KeyError):
        b.disable("missing")


def test_enable_missing_raises():
    b = AgentPromptBuilder()
    with pytest.raises(KeyError):
        b.enable("missing")


def test_enabled_count():
    b = AgentPromptBuilder()
    b.add("a", "A")
    b.add("b", "B", enabled=False)
    b.add("c", "C")
    assert b.enabled_count() == 2


def test_enabled_names_excludes_disabled():
    b = AgentPromptBuilder()
    b.add("a", "A", order=0)
    b.add("b", "B", order=1, enabled=False)
    b.add("c", "C", order=2)
    assert b.enabled_names() == ["a", "c"]


# ---------------------------------------------------------------------------
# get() / has()
# ---------------------------------------------------------------------------


def test_get_existing():
    b = AgentPromptBuilder().add("role", "text")
    s = b.get("role")
    assert s is not None
    assert s.content == "text"


def test_get_missing_returns_none():
    assert AgentPromptBuilder().get("missing") is None


def test_has_existing():
    b = AgentPromptBuilder().add("x", "y")
    assert b.has("x") is True


def test_has_missing():
    assert AgentPromptBuilder().has("missing") is False


# ---------------------------------------------------------------------------
# names()
# ---------------------------------------------------------------------------


def test_names_in_order():
    b = AgentPromptBuilder()
    b.add("c", "C", order=2)
    b.add("a", "A", order=0)
    b.add("b", "B", order=1)
    assert b.names() == ["a", "b", "c"]


def test_names_includes_disabled():
    b = AgentPromptBuilder()
    b.add("a", "A")
    b.add("b", "B", enabled=False)
    assert "b" in b.names()


# ---------------------------------------------------------------------------
# substitute()
# ---------------------------------------------------------------------------


def test_substitute_single():
    b = AgentPromptBuilder().add("role", "Hello {{name}}, you are {{role}}.")
    b.substitute("role", name="Alice", role="helper")
    assert b.render() == "Hello Alice, you are helper."


def test_substitute_with_spaces_in_placeholder():
    b = AgentPromptBuilder().add("x", "{{ key }}")
    b.substitute("x", key="VALUE")
    assert b.render() == "VALUE"


def test_substitute_unmatched_placeholder_unchanged():
    b = AgentPromptBuilder().add("x", "Hello {{unknown}}")
    b.substitute("x", name="Alice")
    assert b.render() == "Hello {{unknown}}"


def test_substitute_missing_section_raises():
    b = AgentPromptBuilder()
    with pytest.raises(KeyError):
        b.substitute("missing", key="val")


def test_substitute_returns_self():
    b = AgentPromptBuilder().add("x", "{{k}}")
    assert b.substitute("x", k="v") is b


def test_substitute_value_with_backslash_group_reference():
    # Regression: replacement values must be inserted literally, not
    # interpreted as regex replacement templates (\1 is an invalid group
    # reference and previously raised re.error).
    b = AgentPromptBuilder().add("x", "Hello {{name}}")
    b.substitute("x", name=r"\1 World $0")
    assert b.render() == "Hello \\1 World $0"


def test_substitute_value_with_backslashes():
    b = AgentPromptBuilder().add("x", "Path: {{p}}")
    b.substitute("x", p=r"C:\temp\new")
    assert b.render() == "Path: C:\\temp\\new"


def test_substitute_value_with_placeholder_of_already_done_key():
    # A replacement value that contains a placeholder for a key that was
    # already substituted must not be re-scanned (no infinite/retroactive
    # substitution). Here ``b`` is processed before ``a``, so the ``{{b}}``
    # introduced by ``a``'s value stays literal.
    b = AgentPromptBuilder().add("x", "{{a}}")
    b.substitute("x", b="DONE", a="{{b}}")
    assert b.render() == "{{b}}"


# ---------------------------------------------------------------------------
# set_metadata()
# ---------------------------------------------------------------------------


def test_set_metadata():
    b = AgentPromptBuilder().add("x", "y")
    b.set_metadata("x", {"k": "v"})
    assert b.get("x").metadata == {"k": "v"}


def test_set_metadata_stores_copy():
    meta = {"k": "v"}
    b = AgentPromptBuilder().add("x", "y")
    b.set_metadata("x", meta)
    meta["extra"] = True
    assert "extra" not in b.get("x").metadata


def test_set_metadata_missing_raises():
    b = AgentPromptBuilder()
    with pytest.raises(KeyError):
        b.set_metadata("missing", {"k": "v"})


def test_set_metadata_returns_self():
    b = AgentPromptBuilder().add("x", "y")
    assert b.set_metadata("x", {}) is b


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


def test_render_custom_separator():
    b = AgentPromptBuilder()
    b.add("a", "A")
    b.add("b", "B")
    assert b.render(separator="\n---\n") == "A\n---\nB"


def test_render_only_enabled():
    b = AgentPromptBuilder()
    b.add("a", "A")
    b.add("b", "B", enabled=False)
    b.add("c", "C")
    assert b.render() == "A\n\nC"


def test_render_section():
    b = AgentPromptBuilder().add("role", "You are helpful.")
    assert b.render_section("role") == "You are helpful."


def test_render_section_missing():
    assert AgentPromptBuilder().render_section("missing") is None


# ---------------------------------------------------------------------------
# remove() / clear()
# ---------------------------------------------------------------------------


def test_remove_existing():
    b = AgentPromptBuilder().add("a", "A").add("b", "B")
    b.remove("a")
    assert not b.has("a")
    assert b.render() == "B"


def test_remove_missing_is_noop():
    b = AgentPromptBuilder().add("a", "A")
    b.remove("nonexistent")
    assert b.count() == 1


def test_remove_returns_self():
    b = AgentPromptBuilder().add("x", "y")
    assert b.remove("x") is b


def test_clear():
    b = AgentPromptBuilder().add("a", "A").add("b", "B")
    b.clear()
    assert b.count() == 0
    assert b.render() == ""


def test_clear_returns_self():
    b = AgentPromptBuilder()
    assert b.clear() is b


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------


def test_to_dict_keys():
    b = AgentPromptBuilder().add("role", "R").add("rules", "X")
    d = b.to_dict()
    assert "role" in d
    assert "rules" in d


def test_to_dict_insertion_order():
    b = AgentPromptBuilder()
    b.add("z", "Z", order=2)
    b.add("a", "A", order=0)
    d = b.to_dict()
    # to_dict uses insertion order, not render order
    assert list(d.keys()) == ["z", "a"]


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr_with_entries():
    b = AgentPromptBuilder().add("a", "A").add("b", "B", enabled=False)
    r = repr(b)
    assert "sections=2" in r
    assert "enabled=1" in r


# ---------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------


def test_full_chain():
    prompt = (
        AgentPromptBuilder()
        .add("role", "You are {{name}}.", order=0)
        .add("rules", "Be concise.", order=1)
        .add("context", "User is a dev.", order=2, enabled=False)
        .substitute("role", name="Claude")
        .enable("context")
        .render()
    )
    assert "You are Claude." in prompt
    assert "Be concise." in prompt
    assert "User is a dev." in prompt

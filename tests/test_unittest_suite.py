"""Standard-library ``unittest`` suite for :mod:`agent_prompt_builder`.

This mirrors the pytest suite in ``test_agent_prompt_builder.py`` but uses
only the Python standard library so it can run anywhere with::

    python3 -m unittest discover -s tests

(no third-party test runner required).
"""

from __future__ import annotations

import os
import sys
import unittest

# Make ``src/`` importable when running the suite directly from a checkout
# (e.g. ``python3 -m unittest discover -s tests``) without an editable install.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agent_prompt_builder import AgentPromptBuilder, PromptSection  # noqa: E402


class PromptSectionTests(unittest.TestCase):
    def test_defaults(self) -> None:
        s = PromptSection(name="role", content="You are helpful.")
        self.assertTrue(s.enabled)
        self.assertEqual(s.order, 0)
        self.assertEqual(s.metadata, {})

    def test_to_dict(self) -> None:
        s = PromptSection(name="role", content="text", enabled=False, order=5)
        d = s.to_dict()
        self.assertEqual(d["name"], "role")
        self.assertEqual(d["content"], "text")
        self.assertIs(d["enabled"], False)
        self.assertEqual(d["order"], 5)

    def test_to_dict_metadata_is_deep_copy(self) -> None:
        s = PromptSection(name="x", content="y", metadata={"k": "v"})
        d = s.to_dict()
        d["metadata"]["k"] = "mutated"
        self.assertEqual(s.metadata["k"], "v")


class EmptyBuilderTests(unittest.TestCase):
    def test_render_empty(self) -> None:
        self.assertEqual(AgentPromptBuilder().render(), "")

    def test_count_zero(self) -> None:
        self.assertEqual(AgentPromptBuilder().count(), 0)

    def test_enabled_count_zero(self) -> None:
        self.assertEqual(AgentPromptBuilder().enabled_count(), 0)

    def test_names_empty(self) -> None:
        self.assertEqual(AgentPromptBuilder().names(), [])

    def test_enabled_names_empty(self) -> None:
        self.assertEqual(AgentPromptBuilder().enabled_names(), [])

    def test_len_zero(self) -> None:
        self.assertEqual(len(AgentPromptBuilder()), 0)

    def test_repr(self) -> None:
        r = repr(AgentPromptBuilder())
        self.assertIn("sections=0", r)
        self.assertIn("enabled=0", r)


class AddTests(unittest.TestCase):
    def test_add_single(self) -> None:
        b = AgentPromptBuilder().add("role", "You are helpful.")
        self.assertTrue(b.has("role"))
        self.assertEqual(b.render(), "You are helpful.")

    def test_insertion_order_when_same_order(self) -> None:
        b = AgentPromptBuilder()
        b.add("a", "A", order=0)
        b.add("b", "B", order=0)
        b.add("c", "C", order=0)
        self.assertEqual(b.render(), "A\n\nB\n\nC")

    def test_explicit_order_respected(self) -> None:
        b = AgentPromptBuilder()
        b.add("late", "Late", order=10)
        b.add("early", "Early", order=1)
        self.assertEqual(b.render(), "Early\n\nLate")

    def test_auto_order_increases(self) -> None:
        b = AgentPromptBuilder()
        b.add("a", "A")
        b.add("b", "B")
        b.add("c", "C")
        self.assertEqual(b.render(), "A\n\nB\n\nC")

    def test_auto_order_after_explicit_high_order(self) -> None:
        # After an explicit high order, the next auto order must come after it.
        b = AgentPromptBuilder()
        b.add("first", "First", order=100)
        b.add("second", "Second")  # auto order should be > 100
        self.assertEqual(b.render(), "First\n\nSecond")
        self.assertGreater(b.get("second").order, 100)

    def test_duplicate_raises(self) -> None:
        b = AgentPromptBuilder().add("role", "text")
        with self.assertRaises(ValueError):
            b.add("role", "other")

    def test_returns_self(self) -> None:
        b = AgentPromptBuilder()
        self.assertIs(b.add("role", "text"), b)

    def test_metadata_stored(self) -> None:
        b = AgentPromptBuilder().add("x", "y", metadata={"source": "user"})
        self.assertEqual(b.get("x").metadata, {"source": "user"})

    def test_metadata_is_copy(self) -> None:
        meta = {"k": "v"}
        b = AgentPromptBuilder().add("x", "y", metadata=meta)
        meta["extra"] = True
        self.assertNotIn("extra", b.get("x").metadata)

    def test_disabled_section_not_rendered(self) -> None:
        b = AgentPromptBuilder().add("x", "X", enabled=False)
        self.assertEqual(b.render(), "")


class AddOrReplaceTests(unittest.TestCase):
    def test_new(self) -> None:
        b = AgentPromptBuilder().add_or_replace("x", "Hello")
        self.assertEqual(b.render(), "Hello")

    def test_existing_overwrites(self) -> None:
        b = AgentPromptBuilder().add("x", "Old").add_or_replace("x", "New")
        self.assertEqual(b.render(), "New")

    def test_preserves_order_position(self) -> None:
        b = AgentPromptBuilder()
        b.add("a", "A", order=0)
        b.add("b", "B", order=5)
        b.add_or_replace("a", "A2")  # keeps order=0
        self.assertEqual(b.names(), ["a", "b"])
        self.assertEqual(b.render(), "A2\n\nB")

    def test_does_not_duplicate_insertion_entry(self) -> None:
        # Replacing must not append a second entry to the insertion list.
        b = AgentPromptBuilder().add("a", "A")
        b.add_or_replace("a", "A2")
        self.assertEqual(b.count(), 1)
        self.assertEqual(b.names(), ["a"])

    def test_returns_self(self) -> None:
        b = AgentPromptBuilder()
        self.assertIs(b.add_or_replace("x", "y"), b)


class SetContentOrderTests(unittest.TestCase):
    def test_set_content(self) -> None:
        b = AgentPromptBuilder().add("role", "Old")
        b.set_content("role", "New")
        self.assertEqual(b.render(), "New")

    def test_set_content_missing_raises(self) -> None:
        with self.assertRaises(KeyError):
            AgentPromptBuilder().set_content("missing", "x")

    def test_set_content_returns_self(self) -> None:
        b = AgentPromptBuilder().add("x", "y")
        self.assertIs(b.set_content("x", "z"), b)

    def test_set_order_reorders(self) -> None:
        b = AgentPromptBuilder()
        b.add("a", "A", order=0)
        b.add("b", "B", order=1)
        b.set_order("b", -1)
        self.assertEqual(b.render(), "B\n\nA")

    def test_set_order_missing_raises(self) -> None:
        with self.assertRaises(KeyError):
            AgentPromptBuilder().set_order("missing", 5)


class EnableDisableTests(unittest.TestCase):
    def test_disable_excludes(self) -> None:
        b = AgentPromptBuilder().add("a", "A").add("b", "B")
        b.disable("b")
        self.assertEqual(b.render(), "A")

    def test_enable_after_disable(self) -> None:
        b = AgentPromptBuilder().add("a", "A").add("b", "B")
        b.disable("b").enable("b")
        self.assertIn("B", b.render())

    def test_set_enabled_false(self) -> None:
        b = AgentPromptBuilder().add("x", "X")
        b.set_enabled("x", False)
        self.assertEqual(b.render(), "")

    def test_set_enabled_true(self) -> None:
        b = AgentPromptBuilder().add("x", "X", enabled=False)
        b.set_enabled("x", True)
        self.assertEqual(b.render(), "X")

    def test_disable_missing_raises(self) -> None:
        with self.assertRaises(KeyError):
            AgentPromptBuilder().disable("missing")

    def test_enable_missing_raises(self) -> None:
        with self.assertRaises(KeyError):
            AgentPromptBuilder().enable("missing")

    def test_enabled_count(self) -> None:
        b = AgentPromptBuilder()
        b.add("a", "A")
        b.add("b", "B", enabled=False)
        b.add("c", "C")
        self.assertEqual(b.enabled_count(), 2)

    def test_enabled_names_excludes_disabled(self) -> None:
        b = AgentPromptBuilder()
        b.add("a", "A", order=0)
        b.add("b", "B", order=1, enabled=False)
        b.add("c", "C", order=2)
        self.assertEqual(b.enabled_names(), ["a", "c"])


class GetHasTests(unittest.TestCase):
    def test_get_existing(self) -> None:
        b = AgentPromptBuilder().add("role", "text")
        s = b.get("role")
        self.assertIsNotNone(s)
        self.assertEqual(s.content, "text")

    def test_get_missing_returns_none(self) -> None:
        self.assertIsNone(AgentPromptBuilder().get("missing"))

    def test_has_existing(self) -> None:
        self.assertTrue(AgentPromptBuilder().add("x", "y").has("x"))

    def test_has_missing(self) -> None:
        self.assertFalse(AgentPromptBuilder().has("missing"))


class NamesTests(unittest.TestCase):
    def test_names_in_render_order(self) -> None:
        b = AgentPromptBuilder()
        b.add("c", "C", order=2)
        b.add("a", "A", order=0)
        b.add("b", "B", order=1)
        self.assertEqual(b.names(), ["a", "b", "c"])

    def test_names_includes_disabled(self) -> None:
        b = AgentPromptBuilder()
        b.add("a", "A")
        b.add("b", "B", enabled=False)
        self.assertIn("b", b.names())


class SubstituteTests(unittest.TestCase):
    def test_single_and_multiple(self) -> None:
        b = AgentPromptBuilder().add("role", "Hello {{name}}, you are {{role}}.")
        b.substitute("role", name="Alice", role="helper")
        self.assertEqual(b.render(), "Hello Alice, you are helper.")

    def test_spaces_in_placeholder(self) -> None:
        b = AgentPromptBuilder().add("x", "{{ key }}")
        b.substitute("x", key="VALUE")
        self.assertEqual(b.render(), "VALUE")

    def test_unmatched_placeholder_unchanged(self) -> None:
        b = AgentPromptBuilder().add("x", "Hello {{unknown}}")
        b.substitute("x", name="Alice")
        self.assertEqual(b.render(), "Hello {{unknown}}")

    def test_missing_section_raises(self) -> None:
        with self.assertRaises(KeyError):
            AgentPromptBuilder().substitute("missing", key="val")

    def test_returns_self(self) -> None:
        b = AgentPromptBuilder().add("x", "{{k}}")
        self.assertIs(b.substitute("x", k="v"), b)

    def test_value_with_regex_backreference_is_literal(self) -> None:
        # Regression guard: replacement values must be inserted literally and
        # never interpreted as regex replacement templates (``\1`` is an
        # invalid group reference and would otherwise raise re.error).
        b = AgentPromptBuilder().add("x", "Hello {{name}}")
        b.substitute("x", name=r"\1 World $0")
        self.assertEqual(b.render(), "Hello \\1 World $0")

    def test_value_with_backslashes_is_literal(self) -> None:
        b = AgentPromptBuilder().add("x", "Path: {{p}}")
        b.substitute("x", p=r"C:\temp\new")
        self.assertEqual(b.render(), "Path: C:\\temp\\new")

    def test_value_containing_placeholder_not_rescanned(self) -> None:
        # A value introducing ``{{b}}`` must not be retroactively substituted.
        b = AgentPromptBuilder().add("x", "{{a}}")
        b.substitute("x", b="DONE", a="{{b}}")
        self.assertEqual(b.render(), "{{b}}")

    def test_repeated_placeholder_all_replaced(self) -> None:
        b = AgentPromptBuilder().add("x", "{{k}} and {{k}} again")
        b.substitute("x", k="V")
        self.assertEqual(b.render(), "V and V again")


class MetadataTests(unittest.TestCase):
    def test_set_metadata(self) -> None:
        b = AgentPromptBuilder().add("x", "y")
        b.set_metadata("x", {"k": "v"})
        self.assertEqual(b.get("x").metadata, {"k": "v"})

    def test_set_metadata_stores_copy(self) -> None:
        meta = {"k": "v"}
        b = AgentPromptBuilder().add("x", "y")
        b.set_metadata("x", meta)
        meta["extra"] = True
        self.assertNotIn("extra", b.get("x").metadata)

    def test_set_metadata_missing_raises(self) -> None:
        with self.assertRaises(KeyError):
            AgentPromptBuilder().set_metadata("missing", {"k": "v"})

    def test_set_metadata_returns_self(self) -> None:
        b = AgentPromptBuilder().add("x", "y")
        self.assertIs(b.set_metadata("x", {}), b)


class RenderTests(unittest.TestCase):
    def test_custom_separator(self) -> None:
        b = AgentPromptBuilder()
        b.add("a", "A")
        b.add("b", "B")
        self.assertEqual(b.render(separator="\n---\n"), "A\n---\nB")

    def test_only_enabled(self) -> None:
        b = AgentPromptBuilder()
        b.add("a", "A")
        b.add("b", "B", enabled=False)
        b.add("c", "C")
        self.assertEqual(b.render(), "A\n\nC")

    def test_render_section(self) -> None:
        b = AgentPromptBuilder().add("role", "You are helpful.")
        self.assertEqual(b.render_section("role"), "You are helpful.")

    def test_render_section_missing(self) -> None:
        self.assertIsNone(AgentPromptBuilder().render_section("missing"))


class RemoveClearTests(unittest.TestCase):
    def test_remove_existing(self) -> None:
        b = AgentPromptBuilder().add("a", "A").add("b", "B")
        b.remove("a")
        self.assertFalse(b.has("a"))
        self.assertEqual(b.render(), "B")

    def test_remove_then_readd(self) -> None:
        # Removing must fully clear internal bookkeeping so re-adding works.
        b = AgentPromptBuilder().add("a", "A")
        b.remove("a")
        b.add("a", "A2")  # must not raise "already exists"
        self.assertEqual(b.render(), "A2")

    def test_remove_missing_is_noop(self) -> None:
        b = AgentPromptBuilder().add("a", "A")
        b.remove("nonexistent")
        self.assertEqual(b.count(), 1)

    def test_remove_returns_self(self) -> None:
        b = AgentPromptBuilder().add("x", "y")
        self.assertIs(b.remove("x"), b)

    def test_clear(self) -> None:
        b = AgentPromptBuilder().add("a", "A").add("b", "B")
        b.clear()
        self.assertEqual(b.count(), 0)
        self.assertEqual(b.render(), "")

    def test_clear_returns_self(self) -> None:
        b = AgentPromptBuilder()
        self.assertIs(b.clear(), b)


class ToDictTests(unittest.TestCase):
    def test_keys_present(self) -> None:
        b = AgentPromptBuilder().add("role", "R").add("rules", "X")
        d = b.to_dict()
        self.assertIn("role", d)
        self.assertIn("rules", d)

    def test_insertion_order_not_render_order(self) -> None:
        b = AgentPromptBuilder()
        b.add("z", "Z", order=2)
        b.add("a", "A", order=0)
        d = b.to_dict()
        self.assertEqual(list(d.keys()), ["z", "a"])

    def test_values_are_section_dicts(self) -> None:
        b = AgentPromptBuilder().add("role", "R", order=3, metadata={"m": 1})
        d = b.to_dict()
        self.assertEqual(d["role"]["content"], "R")
        self.assertEqual(d["role"]["order"], 3)
        self.assertEqual(d["role"]["metadata"], {"m": 1})


class ReprAndChainTests(unittest.TestCase):
    def test_repr_with_entries(self) -> None:
        b = AgentPromptBuilder().add("a", "A").add("b", "B", enabled=False)
        r = repr(b)
        self.assertIn("sections=2", r)
        self.assertIn("enabled=1", r)

    def test_full_chain(self) -> None:
        prompt = (
            AgentPromptBuilder()
            .add("role", "You are {{name}}.", order=0)
            .add("rules", "Be concise.", order=1)
            .add("context", "User is a dev.", order=2, enabled=False)
            .substitute("role", name="Claude")
            .enable("context")
            .render()
        )
        self.assertIn("You are Claude.", prompt)
        self.assertIn("Be concise.", prompt)
        self.assertIn("User is a dev.", prompt)


if __name__ == "__main__":
    unittest.main()

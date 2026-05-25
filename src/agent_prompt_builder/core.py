"""Composable system prompt builder for LLM agents.

Assemble a system prompt from named sections, control their order,
enable/disable them at runtime, and substitute template variables.

Example::

    from agent_prompt_builder import AgentPromptBuilder

    prompt = (
        AgentPromptBuilder()
        .add("role", "You are a helpful assistant.")
        .add("format", "Always respond in JSON.", order=10)
        .add("context", "User is a software engineer.", order=5)
        .disable("format")
        .render()
    )
    # Sections render in order=0 then order=5 (context skipped: format disabled).
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptSection:
    """A named section of a system prompt.

    Attributes:
        name:    Unique section identifier.
        content: Text content of the section.
        enabled: Whether the section is included when rendering.
        order:   Render order — lower values render first. Sections with
                 the same order value render in insertion order.
        metadata: Arbitrary extra metadata dict.
    """

    name: str
    content: str
    enabled: bool = True
    order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return section as a plain dict."""
        return {
            "name": self.name,
            "content": self.content,
            "enabled": self.enabled,
            "order": self.order,
            "metadata": deepcopy(self.metadata),
        }


class AgentPromptBuilder:
    """Build a system prompt from composable named sections.

    Sections are rendered in ascending *order* value. Sections with the
    same order render in insertion order.

    Example::

        builder = AgentPromptBuilder()
        builder.add("role", "You are a helpful assistant.")
        builder.add("rules", "Never reveal secrets.", order=1)
        print(builder.render())
    """

    def __init__(self) -> None:
        # Keep insertion order separately for stable tie-breaking.
        self._sections: dict[str, PromptSection] = {}
        self._insertion_order: list[str] = []
        self._next_auto_order: int = 0

    # ------------------------------------------------------------------
    # Add / update
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        content: str,
        *,
        enabled: bool = True,
        order: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentPromptBuilder:
        """Add a new section.

        Raises :class:`ValueError` if *name* already exists. Use
        :meth:`add_or_replace` to overwrite.

        Args:
            name:     Section identifier (must be unique).
            content:  Text content.
            enabled:  Include in render output (default ``True``).
            order:    Render order (lower = earlier). Auto-increments if
                      omitted so sections render in insertion order.
            metadata: Optional extra dict stored on the section.

        Returns:
            ``self`` for chaining.
        """
        if name in self._sections:
            raise ValueError(f"Section {name!r} already exists; use add_or_replace()")
        effective_order = order if order is not None else self._next_auto_order
        self._next_auto_order = max(self._next_auto_order, effective_order) + 1
        self._sections[name] = PromptSection(
            name=name,
            content=content,
            enabled=enabled,
            order=effective_order,
            metadata=deepcopy(metadata) if metadata else {},
        )
        self._insertion_order.append(name)
        return self

    def add_or_replace(
        self,
        name: str,
        content: str,
        *,
        enabled: bool = True,
        order: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentPromptBuilder:
        """Add a section, or replace it if it already exists.

        When replacing, the original insertion position is preserved.

        Returns:
            ``self`` for chaining.
        """
        if name in self._sections:
            old = self._sections[name]
            effective_order = order if order is not None else old.order
            self._sections[name] = PromptSection(
                name=name,
                content=content,
                enabled=enabled,
                order=effective_order,
                metadata=deepcopy(metadata) if metadata else {},
            )
        else:
            self.add(
                name, content, enabled=enabled, order=order, metadata=metadata
            )
        return self

    def set_content(self, name: str, content: str) -> AgentPromptBuilder:
        """Replace the content of an existing section.

        Raises :class:`KeyError` if *name* does not exist.

        Returns:
            ``self`` for chaining.
        """
        self._get_or_raise(name).content = content
        return self

    def set_order(self, name: str, order: int) -> AgentPromptBuilder:
        """Change the render order of an existing section.

        Raises :class:`KeyError` if *name* does not exist.

        Returns:
            ``self`` for chaining.
        """
        self._get_or_raise(name).order = order
        return self

    def set_metadata(
        self, name: str, metadata: dict[str, Any]
    ) -> AgentPromptBuilder:
        """Replace the metadata dict of an existing section.

        Raises :class:`KeyError` if *name* does not exist.

        Returns:
            ``self`` for chaining.
        """
        self._get_or_raise(name).metadata = deepcopy(metadata)
        return self

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable(self, name: str) -> AgentPromptBuilder:
        """Enable a section by name.

        Raises :class:`KeyError` if *name* does not exist.

        Returns:
            ``self`` for chaining.
        """
        self._get_or_raise(name).enabled = True
        return self

    def disable(self, name: str) -> AgentPromptBuilder:
        """Disable a section by name (excluded from render output).

        Raises :class:`KeyError` if *name* does not exist.

        Returns:
            ``self`` for chaining.
        """
        self._get_or_raise(name).enabled = False
        return self

    def set_enabled(self, name: str, value: bool) -> AgentPromptBuilder:
        """Set enabled state of a section.

        Returns:
            ``self`` for chaining.
        """
        self._get_or_raise(name).enabled = value
        return self

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, name: str) -> PromptSection | None:
        """Return the :class:`PromptSection` for *name*, or ``None``."""
        return self._sections.get(name)

    def has(self, name: str) -> bool:
        """``True`` if a section with *name* exists."""
        return name in self._sections

    def names(self) -> list[str]:
        """Section names in render order (disabled sections included)."""
        return [s.name for s in self._sorted_sections()]

    def enabled_names(self) -> list[str]:
        """Names of enabled sections in render order."""
        return [s.name for s in self._sorted_sections() if s.enabled]

    def count(self) -> int:
        """Total number of sections (enabled + disabled)."""
        return len(self._sections)

    def enabled_count(self) -> int:
        """Number of enabled sections."""
        return sum(1 for s in self._sections.values() if s.enabled)

    # ------------------------------------------------------------------
    # Substitution
    # ------------------------------------------------------------------

    def substitute(self, section: str, **variables: str) -> AgentPromptBuilder:
        """Substitute ``{{key}}`` placeholders in a section's content.

        Raises :class:`KeyError` if *section* does not exist.

        Args:
            section:     Section name to update.
            **variables: Mapping of placeholder key → replacement value.

        Returns:
            ``self`` for chaining.
        """
        sec = self._get_or_raise(section)
        content = sec.content
        for key, value in variables.items():
            content = re.sub(r"\{\{\s*" + re.escape(key) + r"\s*\}\}", value, content)
        sec.content = content
        return self

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, *, separator: str = "\n\n") -> str:
        """Render all enabled sections joined by *separator*.

        Sections render in ascending :attr:`~PromptSection.order`; ties
        break by insertion order.

        Args:
            separator: String placed between consecutive sections.

        Returns:
            The assembled prompt string.
        """
        parts = [s.content for s in self._sorted_sections() if s.enabled]
        return separator.join(parts)

    def render_section(self, name: str) -> str | None:
        """Return the content of a single section, or ``None`` if missing."""
        section = self._sections.get(name)
        return section.content if section is not None else None

    # ------------------------------------------------------------------
    # Remove / clear
    # ------------------------------------------------------------------

    def remove(self, name: str) -> AgentPromptBuilder:
        """Remove a section by name (no-op if not present).

        Returns:
            ``self`` for chaining.
        """
        if name in self._sections:
            del self._sections[name]
            self._insertion_order.remove(name)
        return self

    def clear(self) -> AgentPromptBuilder:
        """Remove all sections.

        Returns:
            ``self`` for chaining.
        """
        self._sections.clear()
        self._insertion_order.clear()
        return self

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a snapshot of all sections as a plain dict.

        Keys are section names; values are the dicts from
        :meth:`PromptSection.to_dict`.
        """
        return {name: self._sections[name].to_dict() for name in self._insertion_order}

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return (
            f"AgentPromptBuilder(sections={self.count()}, "
            f"enabled={self.enabled_count()})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sorted_sections(self) -> list[PromptSection]:
        """Return sections sorted by (order, insertion_index)."""
        indexed = [
            (self._insertion_order.index(name), section)
            for name, section in self._sections.items()
        ]
        return [s for _, s in sorted(indexed, key=lambda x: (x[1].order, x[0]))]

    def _get_or_raise(self, name: str) -> PromptSection:
        if name not in self._sections:
            raise KeyError(f"Section {name!r} not found")
        return self._sections[name]

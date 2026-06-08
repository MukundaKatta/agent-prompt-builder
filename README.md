# agent-prompt-builder

Composable system prompt builder for LLM agents. Assemble prompts from named sections, control render order, enable/disable sections at runtime, and substitute template variables.

## Install

```bash
pip install agent-prompt-builder
```

## Usage

```python
from agent_prompt_builder import AgentPromptBuilder

prompt = (
    AgentPromptBuilder()
    .add("role", "You are {{name}}, a helpful assistant.")
    .add("rules", "Always respond in JSON.", order=10)
    .add("context", "The user is a software engineer.", order=5)
    .substitute("role", name="Claude")
    .render()
)
print(prompt)
```

Output:

```
You are Claude, a helpful assistant.

The user is a software engineer.

Always respond in JSON.
```

## Methods

| Method | Description |
|--------|-------------|
| `.add(name, content, *, enabled, order, metadata)` | Add a new section (raises if name exists) |
| `.add_or_replace(name, content, ...)` | Add or overwrite a section |
| `.set_content(name, content)` | Replace content of an existing section |
| `.set_order(name, order)` | Change the render order of a section |
| `.set_metadata(name, metadata)` | Replace the metadata dict of a section |
| `.enable(name)` / `.disable(name)` | Toggle section visibility |
| `.set_enabled(name, value)` | Set enabled state explicitly |
| `.substitute(name, **vars)` | Replace `{{key}}` placeholders in a section |
| `.render(separator="\n\n")` | Render all enabled sections |
| `.render_section(name)` | Return content of a single section |
| `.remove(name)` | Remove a section (no-op if missing) |
| `.clear()` | Remove all sections |

### Inspection

```python
builder.has("role")         # True if section exists
builder.get("role")         # PromptSection or None
builder.names()             # names in render order
builder.enabled_names()     # names of enabled sections
builder.count()             # total sections
builder.enabled_count()     # enabled sections
builder.to_dict()           # snapshot as plain dict
```

## Behavior notes

- **Render order.** Sections render in ascending `order`; ties break by
  insertion order. When `order` is omitted, it auto-increments so sections
  render in the order they were added.
- **Disabled sections** are kept (and appear in `names()` / `count()`) but are
  excluded from `render()` and `enabled_names()`.
- **Substitution is literal and single-pass.** `{{key}}` placeholders are
  replaced with the exact string value, so values containing `\1`, `$0`, or
  backslashes are inserted verbatim and never interpreted as regex
  replacement templates. A value that itself contains a placeholder is not
  re-scanned.
- **Defensive copies.** `metadata` passed to `add`/`set_metadata` is
  deep-copied in, and `to_dict()` deep-copies out, so external mutation can't
  corrupt a builder's state.

## Typing

This package ships inline type hints and a [`py.typed`](https://peps.python.org/pep-0561/)
marker, so type checkers such as `mypy` and `pyright` pick up its types when
you depend on it.

## Development

The library has no runtime dependencies. The test suite uses only the Python
standard library (`unittest`), so no test framework needs to be installed:

```bash
python -m unittest discover -s tests
```

Linting (optional) uses [Ruff](https://docs.astral.sh/ruff/):

```bash
pip install -e ".[dev]"
ruff check src tests
```

## License

MIT

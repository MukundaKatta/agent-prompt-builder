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

## License

MIT

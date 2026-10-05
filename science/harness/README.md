# Harness

The tool-use loop the two answer-quality tracks run on. A track supplies one TOML config
(model, provider, budget, prompt text, per-arm tool lists, tool descriptions) and a question
loader; the harness does the rest.

- `config.py`: a run config, everything that differs between the two answer-quality tracks.
- `clients.py`: the two search backends, as plain HTTP calls that return raw results.
- `render.py`: how search results are shown to the model, and the dispatch that runs a tool call.
- `agent.py`: one question through one arm, a tool-use loop with a fixed budget.
- `runner.py`: a set of questions through the configured arms, written as one log.

`uv sync && uv run pytest` runs its tests; no API key is needed.

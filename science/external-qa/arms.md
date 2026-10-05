# Arms, prompt and settings

Settings, tool descriptions and tool sentences are read from the header of
`logs/scifact_20260911_110150.json` and are the same in all five logs. The system prompt
around the tool sentence, the user message and the two budget messages are the logged
harness's prompt text; the logs do not record them. The whole setup, including the prompt
text, is in `configs/scifact.toml`; the loop is `harness/harness/agent.py`.

## Settings

| field | value |
| --- | --- |
| model | `claude-sonnet-5`, served as `eu.anthropic.claude-sonnet-5` on Amazon Bedrock |
| tool budget | 16 tool calls per question per arm, every tool call block counted, both tools share it |
| max_tokens | 8192 |
| thinking | `{"type": "adaptive"}` on every turn, including the forced first call |
| results per search | 10 from Redpine Science, 10 from web search |
| passage length shown | 4000 characters |
| web search | Tavily, `search_depth` advanced, client-side |
| Redpine Science collection | `Redpine Science` |

## Tools

`web_search`: "Search the public web. Returns the most relevant pages, each with its title,
URL and an extract of the page text."

`search_connect`: "Search a full-text collection of peer-reviewed scientific and medical
journal articles. Returns the most relevant passages, each with its title, journal, year and
identifiers."

Both take one argument, `query`.

## The one clause that differs

The system prompt is identical across arms except for this sentence. In each arm with a
tool, the first call is forced to that arm's primary tool; after that the agent decides.

| arm | first call forced to | tool sentence |
| --- | --- | --- |
| no retrieval (`closed_book`) | none | You have no search tools, so answer from your own knowledge. |
| web search (`web_forced`) | `web_search` | You have a web_search tool. You have a budget of 16 tool calls for this question. Use them to find evidence before you answer, and answer from what you find together with your own knowledge. |
| Redpine Science (`redpine_forced`) | `search_connect` | You have a search_connect tool. You have a budget of 16 tool calls for this question. Use them to find evidence before you answer, and answer from what you find together with your own knowledge. |
| Redpine Science then web (`redpine_first`) | `search_connect` | You have two tools, search_connect and web_search. You have a budget of 16 tool calls in total for this question. Search search_connect first; use web_search if what you found is insufficient. Answer from what you find together with your own knowledge. |

## The rest of the system prompt

> You are verifying a scientific claim against the biomedical literature (ScholarQABench
> SciFact). Decide whether the claim is supported (true) or refuted (false) by the evidence.
> {tool sentence} When a retrieved passage supports a statement you make, cite it inline with
> its bracketed number exactly as shown, for example [0] or [2]. Passage numbering starts at
> 0. Cite only numbers that appear in the passages you were given. Start your response with a
> line in exactly this form: Answer: <true or false> -- then explain your reasoning
> afterward. Put only the single word true or false on the Answer line.

Each claim is sent as the user message:

> Claim: {claim}
>
> Start with: Answer: <true or false> -- then explain briefly.

A tool call past the budget receives "Tool budget exhausted: no tool calls remain. Answer
now from what you already have." instead of a result. When the agent is still calling tools
after the budget, it receives "You are out of tool calls. Give your final answer now, based
on what you have found so far." and must answer without tools.

## Scored text

The answer that is scored is the text of every assistant turn in the question joined with
newlines, as in the logged runs, and the model is asked to start with an `Answer: true` or
`Answer: false` line.

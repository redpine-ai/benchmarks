# ScholarQABench SciFact, five runs

Claude Sonnet 5 (`eu.anthropic.claude-sonnet-5` on Amazon Bedrock), 208 claims, four arms,
budget of 16 tool calls, scored by the benchmark's own rule. One row per run; each row is
read from the log of the same name in `logs/`.

| run | log | n | no retrieval | web search | Redpine Science | Redpine Science then web |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `scifact_20260911_110150` | 197 | 88.3 | 92.4 | 94.9 | 93.9 |
| 2 | `scifact_20260915_132619` | 199 | 86.9 | 93.5 | 93.5 | 93.0 |
| 3 | `scifact_20260915_140004` | 199 | 87.4 | 94.0 | 96.0 | 93.5 |
| 4 | `scifact_20260915_145823` | 199 | 87.4 | 93.5 | 94.5 | 94.5 |
| 5 | `scifact_20260915_150156` | 196 | 87.8 | 92.3 | 92.9 | 94.4 |
| mean | | | 87.6 | 93.1 | 94.4 | 93.9 |

Values are the percentage of claims answered correctly. A claim is excluded from every arm of a run when the model refused it with no answer in at least one arm, or when an arm call failed after retries, for example on provider throttling. The `excluded` field of each log gives the counts and each row's `excluded` field gives the reason. Run 1 excluded 11 refused claims, run 2 excluded 9, run 3 excluded 9, run 4 excluded 8 refused claims and 1 claim with a failed call, and run 5 excluded 10 refused claims and 2 claims with a failed call.

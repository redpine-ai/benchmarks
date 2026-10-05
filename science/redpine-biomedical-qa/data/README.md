# Question set

`questions.jsonl` holds the 179 expert-validated questions, one JSON object per line.

| Field | Meaning |
| --- | --- |
| `id` | stable id, `steered/` followed by the source paper's DOI |
| `track` | cardiology (79), neuroimmunology (60), sepsis (20) or immunology (20) |
| `question` | the question as reviewed |
| `gold` | the answer key: the required claims, a certainty sentence, the population and setting, and the over-claim an answer must not make |
| `doi` | the source paper's DOI |
| `access` | `open` (105) or `paywalled` (74), the access class of the source paper |

## Loading it

The file needs nothing but the standard library; the claim parser needs nothing but `re`:

```python
import json
questions = [json.loads(line) for line in open("data/questions.jsonl")]
from src.judge import required_claims, scope_of   # run from the track directory
claims = required_claims(questions[0]["gold"])    # list of str, in gold order
```

## The gold grammar

Every gold is one string in four parts, in this order:

    A strong answer conveys: <claim>; <claim>[; <claim>[; <claim>]]. Certainty: <sentence>. Scope: <population and setting>. An answer is wrong if <over-claim>.

- The required claims are the text between `A strong answer conveys:` and the first
  `. Certainty:`; that period is not part of the last claim. Split on `;`, strip
  whitespace, drop empty segments. 2 golds carry two claims, 79 three, 98 four: 633
  claims in all.
- The certainty sentence runs from `Certainty:` to `Scope:`.
- The scope runs from `Scope:` to the period before `An answer is wrong if`, or to the end.
- The wrong-if clause runs from `An answer is wrong if` to the end. 178 golds carry it;
  `steered/10.1177/15230864251401591` has none.

A judge scores an answer per required claim against this list; the list is parsed, never
chosen by the judge.

## Provenance

Each question was written from one paper in the Redpine Science full-text corpus and
reviewed by a domain expert; of 180 reviewed questions, one was dropped. 125 questions were
kept as written, 40 were reworded in the expert's wording, and 14 had their gold edited.

Before an expert saw a question, three test answers were graded against its gold: one that
read the paper, one that read a different valid source, and one with no retrieval. A gold on
which the paper-informed answer scored below 0.75, or the retrieval-free answer 0.5 or
above, was flagged for the expert. Those scores are not published.

Expert names, the experts' per-question review outcomes and comments, and the quoted
evidence passages behind each claim are not published; the DOI identifies the source paper.

The run log with the model answers is in [`../logs/`](../logs/); the judge verdicts and
the statistics are in [`../results/`](../results/). The experts' claim labels are not
published.

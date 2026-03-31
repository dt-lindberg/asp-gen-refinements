# Bug Report: LLM Response Looping Due to Missing Stop Token

## Background

This project is a pipeline that uses a self-hosted large language model (LLM) to solve logic-grid puzzles by generating Answer Set Programming (ASP) code. The pipeline is split into several steps, each of which calls the LLM with a specific prompt and parses the response:

1. **Constants** — format the puzzle's domain values
2. **Predicates** — define the ASP predicate signatures
3. **Search space** — generate the ASP rules that enumerate candidate solutions
4. **Paraphrasing** — rewrite certain constraint sentences into a canonical form
5. **Constraints** — translate the paraphrased constraints into ASP integrity constraints
6. **Reattempt** — if the solver found 0 or >1 solutions, fix the program and try again

The LLM runs locally via [vLLM](https://github.com/vllm-project/vllm) (a batched inference server) using Qwen3-30B in thinking mode. In thinking mode, the model first generates an internal reasoning trace wrapped in `<think>...</think>` tags, then outputs its final response.

Prompts are formatted using the model's chat template (via `apply_chat_template`) and sent to vLLM as raw strings (`tokenize=False`). The `SamplingParams` object controls generation (temperature, top-p, max tokens, etc.).

---

## The Bug

### Root Cause

`SamplingParams` was constructed without a `stop` parameter:

```python
self.sampling_params = SamplingParams(
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=TOP_P,
    top_k=TOP_K,
    min_p=MIN_P,
    # no stop= parameter
)
```

Qwen3's chat template uses a special end-of-turn token (`<|im_end|>`) to signal that the assistant has finished its response. vLLM is configured to treat this as a stop token, so generation usually halts there. However, this is probabilistic: the model sometimes skips emitting `<|im_end|>` and instead begins generating the prefix for a new conversation turn — the literal text `assistant` followed by a new `<think>` block.

Since nothing in `SamplingParams` tells vLLM to stop on the string `"assistant"`, the model then generates a complete second response (identical or nearly identical to the first), and the cycle repeats. Each loop iteration is one full response, so the final output can be dozens of repetitions of the same content separated by `assistant\n\n`, consuming the entire 32,768-token budget.

### When It Triggers

The probability of looping appears to increase when the expected output is short and simple. Steps like **paraphrasing** (task: copy 5 sentences, maybe expanding one) produce responses of ~200 characters. After generating such a short response the model has thousands of tokens of budget remaining, and the "skip `<|im_end|>` and start a new turn" behaviour is more likely. Complex steps like **reattempt** (task: write a full ASP program with reasoning) rarely loop because their long thinking traces consume most of the budget before the issue can manifest.

Observed loop counts per step (from one 10-puzzle run):

| Step | Worst case (assistant count) | Worst response length |
|------|-----------------------------|-----------------------|
| Paraphrasing | 153 | 64 KB |
| Constraints | 70 | 45 KB |
| Search space | — | 109 KB (token budget hit) |
| Constants | 96 | 18 KB |
| Reattempt | 22 | 30 KB |

---

## Downstream Consequences

The looping response itself is recoverable for the step that generated it — the pipeline extracts the **first** valid code block or sentence list from the response, so the correct answer is still present at the start. The real damage happens when a looped response is stored in an intermediate field and then **used as input to a later step**.

### Case 1: Paraphrasing loop poisons the Constraints step (Puzzle 2)

The paraphrasing step for puzzle 2 (a rocket-launch puzzle with 4 constraints) produced a 985-character looped response. In this instance the loop was not just a repetition — the model confused the **example** provided in the paraphrasing prompt with the actual puzzle input and output the example's text verbatim:

**Original constraints:**
```
1. The Worul, the rocket that will launch in February and the rocket that will launch
   in January are all different rockets.
2. The Dreadco is made by Ubersplore.
3. The rocket developed by Permias will launch 1 month before the Foltron.
4. The rocket that will launch in January is made by Techtrin.
```

**Paraphrased output (what the LLM produced):**
```
1. The squad from Grenada ended with 2 silver medals.          ← from the prompt EXAMPLE
2.1 The team from Oman and the team that won 10 silver medals are different.   ← example
2.2 The team from Oman finished with 2 gold medals or finished with 1 gold medal.
2.3 The team that won 10 silver medals finished with 2 gold medals or finished with 1 gold medal.
3. The Dreadco is made by Ubersplore.     ← actual constraint (renumbered from 2)
4. The rocket developed by Permias will launch 1 month before the Foltron.
5. The rocket that will launch in January is made by Techtrin.
```

The constraints generator received this output as its input. It skipped constraints 1 and 2 (they reference Grenada and Oman — irrelevant to a rocket puzzle) and only encoded constraints 3–5. Puzzle 2's original constraint 1 (the all-different rule) was lost entirely.

In subsequent reattempts the model dutifully tried to implement the silver/gold constraints from the contaminated list, inventing `home_country`, `silver`, and `gold` predicates that have no basis in the actual puzzle:

```asp
silver("Grenada", 2).
{ silver(Country, 10) } = 1 :- home_country(Country), Country != "Grenada", Country != "Oman".
:- silver(Country, 10), gold(Country, G), G != 1, G != 2.
```

The puzzle never reached a valid solution.

### Case 2: Predicates loop poisons the Search Space step (Puzzle 3)

The predicates step for puzzle 3 stored its full looped output (multiple paragraphs of reasoning + `assistant` markers + repeated paragraphs) in the `predicates` field. This field was later pasted verbatim into the search-space prompt as the `Predicates:` section.

The search-space prompt ended with `ASP Rules:`, expecting the LLM to complete it with search-space code for puzzle 3. But the bloated predicates input contained several embedded `ASP Rules:` sections (from within the repeated reasoning trace), confusing the LLM about which section it was completing. It generated the search-space code for the **example puzzle in the prompt** (an employee/price/wood-type puzzle) rather than for puzzle 3 (a witness/date/town puzzle):

```asp
% Generated for the WRONG puzzle — came from the prompt example
employee("Bonita"; "Yvette"; "Tabitha").
price(225; 275; 325).
wood_type("ash"; "poplar"; "sandalwood").
{match(E, P, W): price(P), wood_type(W)}=1 :- employee(E).
```

This wrong search space combined with the correct puzzle-3 constraints produced 729 answer sets (severely underconstrained). The reattempt step regenerated the search space correctly and solved the puzzle, so puzzle 3 ultimately produced the right answer — but only because the reattempt loop saved it.

---

## Why Some Responses Stopped Cleanly

Not all responses looped. The failure is probabilistic:

| Outcome | Frequency | Explanation |
|---------|-----------|-------------|
| Clean stop (0 `assistant` occurrences) | Common | Model generated `<|im_end|>` → vLLM stopped normally |
| Short loop (1–4 occurrences) | Occasional | Model skipped `<|im_end|>` once or twice, then recovered |
| Massive loop (50–150 occurrences) | Rare but damaging | Model kept skipping `<|im_end|>` until `max_tokens` cut it off |
| Very long response, 0 occurrences | Common for complex tasks | Long thinking trace consumed the token budget before any loop could start |

---

## Proposed Fix (Unverified)

The standard recommendation for this class of problem is to add the model's end-of-turn token as an explicit stop sequence in `SamplingParams`:

```python
self.sampling_params = SamplingParams(
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=TOP_P,
    top_k=TOP_K,
    min_p=MIN_P,
    stop=["<|im_end|>"],         # string match
    # stop_token_ids=[151643],   # alternative: token ID match
)
```

**However, there are important open questions before applying this fix:**

1. **Does the model actually generate `<|im_end|>`?** The responses observed in cache show the text `assistant` appearing after a valid response, but it is not known whether `<|im_end|>` (token ID 151643) was emitted as a special token immediately before it and simply not visible in the decoded text. For GGUF-quantized models loaded via vLLM with raw string prompts (`tokenize=False`), special token generation behaviour does not always match the HuggingFace original.

2. **String match vs. token ID match.** `stop=["<|im_end|>"]` matches the *decoded string* `<|im_end|>`. If the model emits this as a special token ID (151643) that decodes to an empty string or is stripped during detokenisation, the string match would never fire. In that case `stop_token_ids=[151643]` would be needed instead.

3. **The fix may not address the root cause.** If clean stops are already caused by `<|im_end|>` being recognised as a stop token (either through GGUF metadata or vLLM's tokeniser config), then the loops — which happen despite this — suggest `<|im_end|>` is *not* reliably being generated before the `assistant` prefix appears. In that scenario, adding it to `stop` would not help.

**To properly diagnose the fix,** the raw token IDs in vLLM's output (before decoding) should be inspected for a few representative responses — both clean-stopping and looping — to determine exactly which token terminates a valid response and what token immediately follows it when a loop starts.

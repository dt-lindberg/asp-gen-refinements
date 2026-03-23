# Implementation Plan: Multi-Attempt One-Shot (V3)

## Overview

Replace the iterative refinement loop (step 8) with a multi-attempt one-shot loop. Instead of asking the model to "fix this broken program", give it the full puzzle context plus all prior attempts with feedback and ask it to write a fresh program.

**Key insight:** 95% of correct solutions come from the first attempt. Refinement hurts accuracy (22% -> 16%). The model performs best when one-shotting. Instead of enforcing refinement, give it multiple independent chances.

---

## Architecture

```
Steps 2-6 (UNCHANGED) -> puzzle context outputs
    |
Step 7 (UNCHANGED) -> first Clingo attempt (= attempt 0)
    | if solved -> done
Step 8 (NEW) -> multi-attempt loop:
    for attempt in 1..MAX_ATTEMPTS:
        prompt = system_prompt
               + puzzle_context (story, paraphrased constraints, constants, predicates)
               + <attempt_1> code + feedback </attempt_1>
               + ...
               + <attempt_N> code + feedback </attempt_N>
               + "write a new program" instruction
        Batch generate across unsolved puzzles
        Run Clingo on each result
        If solved -> mark done
```

---

## Prompt Structure

```
[System prompt — from template]

[Story — replaces["<STORY>"]]
[Paraphrased constraints — puzzle_data["constraints_paraphrased"]]
[Constants — puzzle_data["constants_formatted"]]
[Predicates — puzzle_data["predicates"]]

<attempt_1>
  ```asp
  {search_space + constraints from step 7}
  ```

  <feedback>
  {Clingo feedback — syntax errors / UNSAT / multi answer sets}
  </feedback>
</attempt_1>

<attempt_2>
  ```asp
  {full ASP program from attempt 2}
  ```

  <feedback>
  {Clingo feedback}
  </feedback>
</attempt_2>

...

[Final instruction — from template: "Write a NEW ASP program..."]
```

### Prompt Data Sources

| Prompt section | Source | Available after |
|---|---|---|
| Story | `replaces[i]["<STORY>"]` | Initialization |
| Paraphrased constraints | `puzzle_data[i]["constraints_paraphrased"]` | Step 5 |
| Formatted constants | `puzzle_data[i]["constants_formatted"]` | Step 2 |
| Predicates | `puzzle_data[i]["predicates"]` | Step 3 |
| Attempt 0 code | `puzzle_data[i]["rules_all"]` | Step 7 |
| Attempt 0 feedback | `puzzle_data[i]["clingo_errors_0"]` | Step 7 |
| Attempt N code | `attempt_data[i][N-1][0]` | During loop |
| Attempt N feedback | `attempt_data[i][N-1][3]` | During loop |

---

## Files: What Changes

### 1. `prompts/8_reattempt.txt` — NEW FILE

Template with two parts separated by a `===SEPARATOR===` delimiter:

**Part A — System prompt:**
```
You are an expert Answer Set Programming (ASP) programmer solving logic grid puzzles.
You are given a puzzle description, constants, predicates, and your previous attempts
with feedback from the Clingo solver. Write a complete ASP program that produces
exactly one answer set representing the unique solution.
```

**Part B — Final instruction (appended after all attempt blocks):**
```
Write a NEW complete ASP program that solves this puzzle. Do not patch a previous
attempt — write a fresh solution that avoids the issues identified in the feedback.
Include a SINGLE ASP code-block with the rules.
```

The puzzle context and accumulated `<attempt_N>` blocks are constructed programmatically between parts A and B.

### 2. `refinement_loop.py` — REWRITE LOOP BODY

#### Keep unchanged

| Function | Lines | Purpose |
|---|---|---|
| `_parse_error_lines()` | 18-25 | Extract line numbers from Clingo error messages |
| `_annotate_with_line_numbers()` | 28-30 | Prefix code lines with line numbers |
| `_build_error_context()` | 33-46 | Windowed view around error lines |
| `_build_semantic_feedback_multi()` | 49-74 | Feedback for under-constrained programs |
| `MAX_ATTEMPTS` | 12 | Change value from 6 to 4 |
| `SEVERELY_UNDERCONSTRAINED_THRESHOLD` | 14 | Unchanged |
| `MAX_VARIABLE_ATOMS` | 15 | Unchanged |

#### Rewrite

**`refinement_loop_batch()` (lines 77-183) -> `multi_attempt_batch()`**

New signature:
```python
def multi_attempt_batch(puzzle_data, pipeline, statuses, asets_or_errs_list):
```

Takes `puzzle_data` instead of `replaces` — needs direct access to story, paraphrased constraints, constants, predicates, and rules_all.

Changes:
- **Remove**: error-type grouping (`kind_map`, `by_kind`, lines 110-136). All puzzles use the same prompt template now.
- **Remove**: per-error-type template substitution via `replaces[i]` dict. The `<ASP_CODE>`, `<ERRORS>`, `<ERROR_CONTEXT>`, `<SEMANTIC_FEEDBACK>`, `<NUM_ANSWER_SETS>` placeholders are no longer used.
- **Add**: `_build_attempt_prompt()` helper — constructs the full prompt string per puzzle.
- **Add**: `_build_feedback()` helper — formats Clingo output for one attempt, reusing existing helpers.
- **Change**: single batch call per attempt round (no grouping by error kind).
- **Add**: accumulate `(code, feedback_str)` pairs per puzzle across attempts.

#### New functions

**`_build_feedback(status, asets_or_errs)`**

Formats Clingo output into a feedback string for one attempt. Dispatches to existing helpers:
- Syntax error (`status is RuntimeError`) -> error messages + `_build_error_context()`
- UNSAT (`status is None`, 0 answer sets) -> `"0 answer sets (unsatisfiable)"`
- Multi answer sets (`status is None`, N > 1) -> `_build_semantic_feedback_multi()`

**`_build_attempt_prompt(puzzle_data_i, history, system_prompt, instruction)`**

Constructs the full prompt for one puzzle:
1. Puzzle context: story, paraphrased constraints, constants, predicates
2. For each `(code, feedback)` in history: `<attempt_N>` block
3. Final instruction

### 3. `main.py` — MINIMAL CHANGES

| Line(s) | Current | New |
|---|---|---|
| 10-14 | `from refinement_loop import refinement_loop_batch, MAX_ATTEMPTS, _build_semantic_feedback_multi` | `from refinement_loop import multi_attempt_batch, MAX_ATTEMPTS, _build_semantic_feedback_multi` |
| 26-34 | `path_prompt` includes 3 refinement templates (`refinement_syntax`, `refinement_semantic_unsat`, `refinement_semantic_multi`) | Replace with single `"reattempt"` entry pointing to `prompts/8_reattempt.txt` |
| 168-171 | `refinement_loop_batch(replaces, pipeline, statuses, asets_or_errs_list)` | `multi_attempt_batch(puzzle_data, pipeline, statuses, asets_or_errs_list)` |
| 240-248 | Column names `f"refinement_{i}"` | Rename to `f"attempt_{i}"` |

Everything else stays identical: steps 2-7, result collection (lines 184-258), CLI args (lines 261-275).

### 4. `pipeline.py` — MINOR ADDITION

Add `gen_response_raw_batch(kind, prompts_list)` method (~10 lines). Takes pre-built prompt strings directly, skipping placeholder substitution. The new prompt is constructed in `_build_attempt_prompt()`, so the `replaces` dict substitution in `gen_response_batch()` is unnecessary.

```python
def gen_response_raw_batch(self, kind, prompts):
    """Generate responses for pre-built prompt strings."""
    responses = [None] * len(prompts)
    miss_indices = []
    miss_messages = []

    for i, prompt in enumerate(prompts):
        if prompt in self.cache[kind]:
            responses[i] = self.cache[kind][prompt]
        else:
            miss_indices.append(i)
            miss_messages.append([{"role": "user", "content": prompt}])

    if miss_messages:
        generated = self._get_engine().generate_batch(miss_messages)
        for idx, resp in zip(miss_indices, generated):
            self.cache[kind][prompts[idx]] = resp
            responses[idx] = resp
        self.save_cache()

    return responses
```

### 5. `interface.py` — COSMETIC

Rename "refinement" labels to "attempt" in Streamlit display. No functional changes.

---

## Files: What Stays the Same

| File | Status |
|---|---|
| `vllm_engine.py` | Unchanged |
| `dataset150.py` | Unchanged |
| `utils.py` | Unchanged |
| `logger.py` | Unchanged |
| `prompts/2_constant_formatting.txt` | Unchanged |
| `prompts/3_gen_predicates.txt` | Unchanged |
| `prompts/4_gen_search_space.txt` | Unchanged |
| `prompts/5_paraphrasing.txt` | Unchanged |
| `prompts/6_gen_constraints.txt` | Unchanged |
| `prompts/7_refinement_syntax.txt` | Unused, kept in place |
| `prompts/8_refinement_semantic_unsat.txt` | Unused, kept in place |
| `prompts/9_refinement_semantic_multi.txt` | Unused, kept in place |

---

## Function-Level Change Summary

| Function | File:Line | Action |
|---|---|---|
| `_parse_error_lines()` | `refinement_loop.py:18` | Keep |
| `_annotate_with_line_numbers()` | `refinement_loop.py:28` | Keep |
| `_build_error_context()` | `refinement_loop.py:33` | Keep |
| `_build_semantic_feedback_multi()` | `refinement_loop.py:49` | Keep |
| `refinement_loop_batch()` | `refinement_loop.py:77` | Rewrite -> `multi_attempt_batch()` |
| `_build_attempt_prompt()` | `refinement_loop.py` (new) | New: constructs full prompt per puzzle |
| `_build_feedback()` | `refinement_loop.py` (new) | New: formats Clingo output for one attempt |
| `gen_response_batch()` | `pipeline.py:76` | Keep |
| `gen_response_raw_batch()` | `pipeline.py` (new) | New: batch generation from pre-built prompts |
| `gen_answer_set()` | `pipeline.py:178` | Keep |
| `main()` | `main.py:22` | Small edits (imports, step 8 call, column names) |

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| What to re-generate | Only the final ASP program | Steps 2-6 outputs are structural and stable; the constraint program is where errors occur |
| Expert count | 1 per puzzle | Keep simple; match current single-expert setup |
| Attempt count | 4 (configurable via `MAX_ATTEMPTS`) | Enough diversity without blowing context budget |
| Temperature | 0.7 (unchanged) | Consistent with current pipeline |
| Voting | None — first valid solution wins | No multi-expert voting needed with single expert |
| Feedback format | Reuse existing Clingo feedback helpers | Already battle-tested; syntax errors, UNSAT, multi all handled |
| Prompt style | Single-turn with accumulated context | Model sees full history + puzzle context in one prompt |
| Framing | "Write a NEW program" not "fix this program" | Core philosophical change: don't constrain to refinement |

---

## Risks / Notes

- **Context length**: With `max_model_len=9800`, 4 attempts of full ASP programs + puzzle context may push limits. To be handled separately once implementation is running.
- **Cache invalidation**: New prompt format = new cache keys. Old refinement caches will not be hit. This is correct behavior.
- **Feedback for attempt 0**: Already computed and stored in `puzzle_data[i]["clingo_errors_0"]` at `main.py:149-156`. Reused directly.
- **Old refinement prompts**: Left in `prompts/` directory but no longer referenced. Can be cleaned up later.

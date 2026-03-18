# LLM + ASP with Refinement Loop

This is an extended implementation of [Leveraging Large Language Models to Generate Answer Set Programs](https://proceedings.kr.org/2023/37/kr2023-0037-ishay-et-al.pdf).
[Lab page](https://azreasoners.github.io/ARG-webpage/)

The original pipeline prompts an LLM to translate natural-language logic puzzles into Answer Set Programming (ASP) rules, then evaluates them with Clingo. This fork extends it with support for additional LLM backends, an automatic refinement loop that corrects syntactic and semantic errors, and a Streamlit interface for inspecting results.

## What's new vs. the original

- **Self-hosted LLM backend**: runs `Qwen3-30B-A3B-Instruct` (GGUF, Q4_K_M) locally via vLLM (`vllm_engine.py`). No external API keys required.
- **Batched inference**: LLM steps 2–6 and the refinement loop are batched across all puzzles, sending them to vLLM in a single `generate` call per step. This maximises GPU throughput compared to sequential per-puzzle calls.
- **Refinement loop** (Step 8): if Clingo reports a syntax error or the wrong number of answer sets, the LLM is automatically re-prompted up to `MAX_ATTEMPTS` times to fix the program. The refinement loop is also batched — each attempt groups all still-failing puzzles by error type and issues one batch call per type.
- **Streamlit inspector** (`interface.py`): a visual interface for browsing per-puzzle pipeline outputs and refinement diffs.
- **`mistakes/` directory**: result `.xlsx` files are now written to a `mistakes/` subdirectory (timestamped filenames) instead of the root directory.

## Installation

Run the install job on Snellius (installs vLLM with CUDA support, then Clingo and other dependencies):

```bash
sbatch install_env.job
```

The model is loaded from the local HuggingFace cache (`~/.cache/huggingface/hub/`). Download it beforehand if not already present.

## How to run

Submit the main job:

```bash
sbatch run.job
```

Or run interactively (requires a GPU node):

```bash
python main.py --dataset_name test --num 50 --engine qwen3-30b-local
```

`--dataset_name` is one of `train`, `test`, or `test_HA`. `--num` is the number of puzzles (use `-1` for all 50). Without `--debug`, only failed puzzles are recorded.

Results are saved to `mistakes/mistakes_<timestamp>.xlsx`.

## Pipeline steps

Steps 2–6 are batched: all puzzles are processed together in a single LLM call per step before moving on. Step 7 (Clingo) is CPU-bound and runs sequentially. Step 8 batches puzzles by refinement type within each attempt.

| Step | Description | Mode |
|------|-------------|------|
| 2 | Format constants and categories | batched |
| 3 | Generate predicates of interest | batched |
| 4 | Generate search space (facts + choice rules) | batched |
| 5 | Paraphrase constraints into sentences | batched |
| 6 | Generate ASP constraint rules | batched |
| 7 | Compile and solve with Clingo | sequential (CPU) |
| 8 | Refinement loop | batched per attempt |

Prompts for each step live in `prompts/`. LLM responses are cached per-engine in `caches/` (keyed on the full substituted prompt) so re-runs skip already-generated responses.

## Refinement loop (Step 8)

If Step 7 produces a syntax error or the wrong number of answer sets (0 or >1), the loop re-prompts the LLM with targeted feedback up to `MAX_ATTEMPTS` times. Puzzles that reach exactly 1 answer set are removed from the active set immediately.

Within each attempt, active puzzles are grouped by their error type and sent as a single batch call per group:

- **Syntax error** (`RuntimeError` from Clingo): uses `prompts/7_refinement_syntax.txt`. The ASP code is annotated with line numbers and the error context is included.
- **Unsatisfiable** (0 answer sets): uses `prompts/8_refinement_semantic_unsat.txt`.
- **Under-constrained** (>1 answer sets): uses `prompts/9_refinement_semantic_multi.txt` with a sample of differing atoms across answer sets.

Each attempt is recorded in the output Excel file (`refinement_1`, `#answer_sets_1`, `clingo_time_1`, `clingo_errors_1`, ... up to `MAX_ATTEMPTS`).

## Streamlit inspector

To browse results interactively:

```bash
streamlit run interface.py -- --file mistakes/mistakes_<timestamp>.xlsx
```

The inspector shows each puzzle's pipeline steps (inputs/outputs), the initial Clingo result, each refinement attempt with an inline diff, and the final prediction vs. ground truth.

## How to read the results

Each row in the output `.xlsx` file corresponds to one puzzle. Key columns:

- `story`, `constraints` — raw puzzle input
- `constraints_paraphrased`, `constants_formatted`, `predicates`, `rules_search_space`, `rules_constraints` — intermediate LLM outputs
- `refinement_0`, `#answer_sets_0`, `clingo_time_0`, `clingo_errors_0` — initial Clingo run
- `refinement_N`, `#answer_sets_N`, `clingo_time_N`, `clingo_errors_N` — refinement attempt N (1..MAX\_ATTEMPTS)
- `prediction`, `solution` — final prediction and ground truth
- `reasoning_*` — chain-of-thought traces (not populated for the local vLLM backend)

The `error_analysis/` directory contains manually annotated Excel files tracking error categories for GPT-3 and GPT-4 on the test set. Errors are highlighted in red; fixes in blue. Error categories:

**Paraphrase:**
- Either/or — incorrect conversion of either/or sentences
- All different — incorrect conversion of "a, b, c, and d are different" sentences

**Constraint Generation (semantic):**
- Comparison between times
- Operator — wrong operator (e.g. `<` instead of `>`)
- Incorrect disjunction in the head
- Other

**Constraint Generation (syntax):** wrong variable name or variables out of order
**Constant Formatting:** constant has double quotes when it should not

## Citation original work

```bibtex
@inproceedings{KR2023-37,
    title     = {{Leveraging Large Language Models to Generate Answer Set Programs}},
    author    = {Ishay, Adam and Yang, Zhun and Lee, Joohyung},
    booktitle = {{Proceedings of the 20th International Conference on Principles of Knowledge Representation and Reasoning}},
    pages     = {374--383},
    year      = {2023},
    month     = {8},
    doi       = {10.24963/kr.2023/37},
    url       = {https://doi.org/10.24963/kr.2023/37},
}
```

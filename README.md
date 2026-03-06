# LLM + ASP with Refinement Loop

This is an extended implementation of [Leveraging Large Language Models to Generate Answer Set Programs](https://proceedings.kr.org/2023/37/kr2023-0037-ishay-et-al.pdf).
[Lab page](https://azreasoners.github.io/ARG-webpage/)

The original pipeline prompts an LLM to translate natural-language logic puzzles into Answer Set Programming (ASP) rules, then evaluates them with Clingo. This fork extends it with support for additional LLM backends, an automatic refinement loop that corrects syntactic and semantic errors, and a Streamlit interface for inspecting results.

## What's new vs. the original

- **Additional model backends**: Google `gemini-3-flash-preview` (via Google AI API) and OpenAI `gpt-oss-120b` (via Groq API), in addition to the original OpenAI models (`gpt-4`, `text-davinci-003`, etc.)
- **Refinement loop** (Step 8): if Clingo reports a syntax error or the wrong number of answer sets, the LLM is automatically re-prompted up to `MAX_ATTEMPTS` times to fix the program.
- **Reasoning trace extraction**: for `gpt-oss-120b`, chain-of-thought reasoning steps are extracted and stored alongside results.
- **Streamlit inspector** (`interface.py`): a visual interface for browsing per-puzzle pipeline outputs and refinement diffs.
- **`mistakes/` directory**: result `.xlsx` files are now written to a `mistakes/` subdirectory (timestamped filenames) instead of the root directory.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai google-genai groq clingo pandas xlsxwriter streamlit python-dotenv tqdm
```

## Preparation

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
LOG_LEVEL=info
```

## How to run

Evaluate on puzzles from the dataset. `--engine` selects the LLM backend:

```bash
# GPT-4 (OpenAI)
python main.py --debug --dataset_name test --num 5 --engine gpt-4

# Gemini Flash (Google AI)
python main.py --debug --dataset_name test --num 5 --engine gemini-3-flash-preview

# GPT-OSS 120B (via Groq)
python main.py --debug --dataset_name test --num 5 --engine gpt-oss-120b
```

`--dataset_name` is one of `train`, `test`, or `test_HA`. `--num` is the number of puzzles (use `-1` for all 50). Without `--debug`, only failed puzzles are recorded.

Results are saved to `mistakes/mistakes_<timestamp>.xlsx`.

### Other entry points

```bash
python sudoku.py --engine gpt-4
python jobs_puzzle.py --engine gpt-4
```

## Pipeline steps

The pipeline runs the following steps for each puzzle:

| Step | Description |
|------|-------------|
| 2 | Format constants and categories |
| 3 | Generate predicates of interest |
| 4 | Generate search space (facts + choice rules) |
| 5 | Paraphrase constraints into sentences |
| 6 | Generate ASP constraint rules |
| 7 | Compile and solve with Clingo |
| 8 | Refinement loop (if Clingo fails or gives wrong answer set count) |

Prompts for each step live in `prompts/`. API responses are cached per-engine in `caches/` so re-runs don't repeat API calls.

## Refinement loop (Step 8)

If Step 7 produces a syntax error or the wrong number of answer sets (0 or >1), the loop re-prompts the LLM with targeted feedback up to `MAX_ATTEMPTS` times:

- **Syntax error** (`RuntimeError` from Clingo): uses `prompts/7_refinement_syntax.txt` with the error messages.
- **Unsatisfiable** (0 answer sets): uses `prompts/8_refinement_semantic_unsat.txt`.
- **Under-constrained** (>1 answer sets): uses `prompts/9_refinement_semantic_multi.txt` with a sample of differing atoms across answer sets.

Each attempt is recorded in the output Excel file (`refinement_1`, `#answer_sets_1`, `clingo_time_1`, `clingo_errors_1`, ... up to `MAX_ATTEMPTS`).

## Streamlit inspector

To browse results interactively:

```bash
streamlit run interface.py -- --file mistakes/mistakes_<timestamp>.xlsx
```

The inspector shows each puzzle's pipeline steps (inputs/outputs), the initial Clingo result, each refinement attempt with an inline diff, and the final prediction vs. ground truth. It also surfaces reasoning traces when available (gpt-oss-120b).

## How to read the results

Each row in the output `.xlsx` file corresponds to one puzzle. Key columns:

- `story`, `constraints` — raw puzzle input
- `constraints_paraphrased`, `constants_formatted`, `predicates`, `rules_search_space`, `rules_constraints` — intermediate LLM outputs
- `refinement_0`, `#answer_sets_0`, `clingo_time_0`, `clingo_errors_0` — initial Clingo run
- `refinement_N`, `#answer_sets_N`, `clingo_time_N`, `clingo_errors_N` — refinement attempt N (1..MAX\_ATTEMPTS)
- `prediction`, `solution` — final prediction and ground truth
- `reasoning_*` — chain-of-thought traces (gpt-oss-120b only)

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

# Autonomous Research
You are tasked with autonomously researching:
* Using LLMs to write code in Answer Set Programming (ASP) in order to solve logical grid puzzles.

You reside on a GPU cluster and will execute experiments entirely unsupervised.

## Setup
To set up a new experiment, work with the user to:
1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Use a single sub-agent to explore it.
4. **Verify that the environment exists**: Check that `.venv` exists. If not, run their installation scripts before beginning any experimentation.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation
Each experiment runs on a single Nvidia A100 (40GB) GPU with access to 16 CPU cores. The script runs for a **fixed time budget of 30 minutes** (wall clock time, including startup/compilation). You launch it simply as: `sbatch run.job`.
* Note: sometimes it takes a bit of time for the job to *actually start*, as there might be a queue for the GPUs. This does NOT count towards the time limit.

### What you CAN do
- Modify the following files: `main.py`, `pipeline.py`, `refinement_loop.py`, `vllm_engine.py`, all prompts in `prompts/`. Everything is fair game: prompting, changes to the iteration loop and/or feedback, increasing the number of attempts, change the maximum length of generation sequences, batch size, add thinking, and anything else that is defined in these files.
- Search the web for (recent) information and documentation.

### What you CANNOT do
- Modify the input/benchmark. You cannot change the puzzles the LLM is running on.
- Modify the time limit, it's a hard limit and cannot be changed.
- Change the model, it is what it is.

**The goal is simple**: Get the highest score possible, i.e., solve as many puzzles as possible.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.
* One more puzzle solved that adds 40 lines of hacky code => probably not worth it.
* One more puzzle solved from deleting code and/or dependencies => definitely keep.
* No improvement but much simpler code and/or dependencies => definitely keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Logging results
When an experiment is done, log it to `outputs/<tag>.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:
```
commit	solved	status	description
```
1. git commit hash (short, 7 chars)
2. How many puzzles were solved (e.g. 6) — use 0 for crashes/timeouts
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:
```
commit	solved	status	description
a1b2c3d	4	keep	baseline
c3d4ef1	6	keep	increased sequence length by 20%
d8e7c1a	11	keep	MAX_ATTEMPTS=8
c3d4e5f	0.000	crash	 MAX_ATTEMPTS=16 (timeout)
```

## Interacting with the GPU
The cluster uses a Slurm workload manager. All scripts run through a `.job` file specifying partition, timeout, number of tasks and more. **You are NOT to modify this**; the research goal is to maximize performance with this setup.

### Useful commands
All of these assume you are running from `/src`.
* `sbatch <file>.job` for running jobs, also returns `job_id`.
* `squeue` for checking the queue. Allows you to see when a job goes from being queued to running.
* `jlog <job_id>` (defined in `~/.bashrc`) monitors the structured log for a running job and exits when the job finishes.

### Logging
The script uses a logger (defined in `logger.py`) that writes structured lines — env info, model load time, generation stats — to both stdout and `outputs/<job_id>.log`. The raw LLM response is printed to stdout only and does not appear in the log file. Use `jlog` to monitor a run without the response flooding your context.

## The experiment loop
The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

**LOOP FOREVER**:

1. Look at the git state: the current branch/commit we're on.
2. Tune the files mentioned above with an experimental idea by directly hacking the code.
3. `git commit` your changes.
4. Run the experiment: `sbatch run.job`.
5. Monitor the run: `jlog <job_id>`, do NOT monitor or read the raw (stdout) of runs, do NOT let redundant information flood your context.
6. Check the results in `outputs/<job_id>.json` and record them in the `.tsv` file.
7. If the score improves, you 'advance' the branch by keeping the commit.
8. If the score decreases, revert the changes you made to the code (keep record of results!).

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**GPU allocation**: If it takes a lot of time for the job to get allocated a GPU, there is NOTHING YOU CAN DO. It's the unfortunate truth, your best option is to keep running `jlog <job_id>` until it eventually gets scheduled.

**Timeout**: Each experiment should take <30 minutes total. If a run exceeds the limit IT WILL CRASH.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — research online, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

GOOD LUCK!

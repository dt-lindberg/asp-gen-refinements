Here is the exact empirical process you can run on your SLURM cluster to find the right balance for your A100 vs H100 and your 0-60K token prompts.

### 1. The Setup: `benchmark_serving.py`
vLLM ships with a highly useful script specifically for this purpose: `benchmarks/benchmark_serving.py`. You run your vLLM server in one terminal, and in another, you fire requests at it using this script to simulate heavy load. [github](https://github.com/vllm-project/vllm/issues/21163)

To do this accurately, you need to create a custom JSONL dataset that perfectly mirrors your real workload. Since your prompts range from 0–20K initially and grow to 20K–60K, generate a synthetic JSONL file containing about 200-500 prompts with that exact distribution of input lengths and expected output lengths.

You then run the benchmark against your server:
```bash
python3 benchmarks/benchmark_serving.py \
    --backend vllm \
    --model path/to/Qwen3-30B-A3-Instruct-4bit \
    --dataset-name custom \
    --dataset-path ./my_custom_60k_workload.jsonl \
    --request-rate inf \
    --save-result
```
*Note: Setting `--request-rate inf` blasts the server with all requests at once to test absolute maximum concurrency and queueing behavior.*

### 2. The Key Metrics to Evaluate
When the benchmark finishes, it outputs a summary report. You need to focus on three specific latency metrics to interpret the user experience: [github](https://github.com/vllm-project/vllm/issues/6531)

1. **TTFT (Time To First Token):** How long a user waits before the first word appears. This measures the prefill phase. If this is high, users stare at a blank screen.
2. **TPOT (Time Per Output Token) / ITL (Inter-Token Latency):** How fast the text streams out after it starts. If ITL is above ~50-100ms, the text generation looks stuttery and slow to human eyes.
3. **Throughput (req/s & tok/s):** The total system efficiency. 

### 3. The Experiments to Conduct
You will restart your vLLM server with different parameter combinations and run the exact same `benchmark_serving.py` command against it, recording the results in a spreadsheet.

**Experiment A: Tuning `max_num_batched_tokens` (The Prefill Test)**
*   **Goal:** Find the highest chunk size that lowers TTFT for 60K prompts without causing OOMs or destroying ITL for active generations. [croz](https://croz.net/run-your-own-ai-at-scale-vol-1-tuning-vllm/)
*   **Test Values:** 8192, 16384, 32768, 65536.
*   **What to look for:** As you increase this, TTFT should drop because the GPU processes larger chunks of your massive prompts at once. However, watch the ITL. If ITL spikes heavily at 65536, it means massive chunked prefills are stalling the GPU and interrupting the running sequences. Find the threshold where TTFT is acceptable but ITL stays under ~50ms. [discuss.vllm](https://discuss.vllm.ai/t/question-about-parameter-max-num-batched-tokens/2012)

**Experiment B: Tuning `max_num_seqs` (The Concurrency Test)**
*   **Goal:** Find how many sequences can run simultaneously before KV cache swapping destroys performance.
*   **Test Values:** 5, 10, 16, 32, 64.
*   **What to look for:** Set `--request-rate 10` (10 requests per second) to simulate real traffic. At `max_num_seqs=64`, you might see great initial throughput, but as prompts hit 60K tokens, you'll see massive spikes in TPOT (Time Per Output Token). This indicates vLLM ran out of VRAM and started heavily swapping blocks to the CPU. You want to find the highest `max_num_seqs` where the P99 TPOT (the 99th percentile worst-case latency) remains stable and no swapping warnings appear in your server logs.

### 4. Interpreting the Hardware Differences (A100 vs H100)
Run the exact same matrix of experiments on the A100 (40GB) and H100 (80GB). 
*   **On the A100:** You will likely find that `max_num_seqs` must be kept very low (e.g., 4-8) because 22GB of available KV cache physically cannot hold many 60K token sequences. 
*   **On the H100:** Because it has 80GB VRAM, the KV cache pool is massive. You will likely be able to increase `max_num_seqs` to 20-30 while maintaining a perfectly flat ITL. You can also afford a much larger `max_num_batched_tokens` because the H100 has immensely higher compute bandwidth and larger VRAM buffers for the prefill phase.

By charting TTFT (latency) on the Y-axis and Throughput (tok/s) on the X-axis for each parameter combination, you generate a "Pareto frontier" curve. You simply pick the configuration point on that curve that matches your priority—whether that is maximizing total cluster throughput or guaranteeing snappy responses for end users.

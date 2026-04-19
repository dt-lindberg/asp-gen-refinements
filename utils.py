import re

MAX_EXTRACTED_CHARS = 8000


def extract_code_blocks(response):
    """Extract content from any triple-tick code blocks (``` ```prolog, ```asp, etc.).

    If one or more complete code blocks are found, returns their contents joined
    by two newlines. If only an opening fence is found (unclosed block), extracts
    everything after the opening fence. Otherwise returns the original response unchanged.

    Output is capped at MAX_EXTRACTED_CHARS to prevent a runaway response (e.g.
    a thinking-trace overflow that bypassed code-fence detection) from being
    fed back into the next prompt and exceeding the model's context window.
    """
    match = re.search(r"```[^\n]*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()[:MAX_EXTRACTED_CHARS]
    # Fallback: strip a single unclosed opening fence (e.g. ```asp\n...)
    match = re.search(r"```[^\n]*\n(.*)", response, re.DOTALL)
    if match:
        return match.group(1).strip()[:MAX_EXTRACTED_CHARS]
    match = re.search(r"<asp_program>(.*?)</asp_program>", response, re.DOTALL)
    if match:
        return match.group(1).strip()[:MAX_EXTRACTED_CHARS]
    return response[:MAX_EXTRACTED_CHARS]

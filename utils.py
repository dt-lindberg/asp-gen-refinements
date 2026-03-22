import re


def extract_code_blocks(response):
    """Extract content from any triple-tick code blocks (``` ```prolog, ```asp, etc.).

    If one or more complete code blocks are found, returns their contents joined
    by two newlines. If only an opening fence is found (unclosed block), extracts
    everything after the opening fence. Otherwise returns the original response unchanged.
    """
    blocks = re.findall(r"```[^\n]*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks)
    # Fallback: strip a single unclosed opening fence (e.g. ```asp\n...)
    match = re.search(r"```[^\n]*\n(.*)", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response

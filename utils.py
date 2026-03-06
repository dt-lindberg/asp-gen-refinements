import re


def extract_code_blocks(response):
    """Extract content from any triple-tick code blocks (``` ```prolog, ```asp, etc.).

    If one or more code blocks are found, returns their contents joined
    by two newlines. Otherwise returns the original response unchanged.
    """
    blocks = re.findall(r"```[^\n]*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks)
    return response

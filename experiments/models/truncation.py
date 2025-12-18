# Context truncation utilities for handling oversized prompts
# Created: 2025-12-07
#
# Truncates from FRONT to preserve question/instruction at END.
# Designed for LongMemEval where question is always at the end of the prompt.

import tiktoken
from typing import Tuple

# Use same tokenizer as rest of codebase
TOKENIZER = tiktoken.get_encoding("o200k_base")


def estimate_token_count(text: str) -> int:
    """Count tokens using tiktoken o200k_base encoding."""
    return len(TOKENIZER.encode(text, disallowed_special=()))


def truncate_from_front(
    prompt: str,
    max_tokens: int,
    preserve_chars_from_end: int = 3000,
) -> Tuple[str, dict]:
    """
    Truncate prompt from the front to fit within max_tokens.

    Preserves content at the end (where questions/instructions typically are)
    and removes content from the beginning (older context).

    Args:
        prompt: Full prompt text
        max_tokens: Maximum allowed tokens after truncation
        preserve_chars_from_end: Chars to always preserve from end (safety for question)

    Returns:
        Tuple of (truncated_prompt, metadata_dict)
        metadata_dict: {original_tokens, final_tokens, tokens_removed, truncated}
    """
    original_tokens = estimate_token_count(prompt)

    metadata = {
        'original_tokens': original_tokens,
        'final_tokens': original_tokens,
        'tokens_removed': 0,
        'truncated': False,
    }

    # No truncation needed
    if original_tokens <= max_tokens:
        return prompt, metadata

    tokens_to_remove = original_tokens - max_tokens

    # Split into front (truncatable) and end (preserved)
    if len(prompt) <= preserve_chars_from_end:
        # Prompt too short to truncate safely
        return prompt, metadata

    end_portion = prompt[-preserve_chars_from_end:]
    front_portion = prompt[:-preserve_chars_from_end]

    # Estimate chars to remove (~4 chars per token, with 30% buffer for safety)
    chars_to_remove = int(tokens_to_remove * 4 * 1.3)

    if chars_to_remove >= len(front_portion):
        # Cannot truncate enough from front alone
        return prompt, metadata

    # Remove from front
    truncated_front = front_portion[chars_to_remove:]

    # Find clean sentence boundary (first period/newline after cut point)
    boundary_found = False
    for i, char in enumerate(truncated_front[:500]):  # Search first 500 chars
        if char in '.!?\n' and i + 1 < len(truncated_front):
            next_char = truncated_front[i + 1]
            if next_char in ' \n\t':
                truncated_front = truncated_front[i + 2:]
                boundary_found = True
                break

    # If no sentence boundary, find word boundary
    if not boundary_found:
        first_space = truncated_front.find(' ')
        if 0 < first_space < 100:
            truncated_front = truncated_front[first_space + 1:]

    # Reconstruct
    truncated_prompt = truncated_front + end_portion
    final_tokens = estimate_token_count(truncated_prompt)

    # Iteratively remove more if still over limit
    iterations = 0
    while final_tokens > max_tokens and len(truncated_front) > 500 and iterations < 50:
        # Remove another ~100 tokens worth
        truncated_front = truncated_front[400:]
        # Find next sentence boundary
        for i, char in enumerate(truncated_front[:200]):
            if char in '.!?\n':
                truncated_front = truncated_front[i + 1:].lstrip()
                break
        truncated_prompt = truncated_front + end_portion
        final_tokens = estimate_token_count(truncated_prompt)
        iterations += 1

    metadata.update({
        'final_tokens': final_tokens,
        'tokens_removed': original_tokens - final_tokens,
        'truncated': True,
    })

    return truncated_prompt, metadata

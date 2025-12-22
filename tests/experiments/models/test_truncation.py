# Tests for truncation utilities
# Created: 2025-12-22

import pytest
from experiments.models.truncation import (
    estimate_token_count,
    truncate_from_front,
    TOKENIZER,
)


class TestEstimateTokenCount:
    """Test the estimate_token_count function."""

    def test_empty_string(self):
        """Test token count for empty string."""
        assert estimate_token_count("") == 0

    def test_single_word(self):
        """Test token count for a single word."""
        count = estimate_token_count("hello")
        assert count > 0
        assert count < 5  # Single word should be small

    def test_sentence(self):
        """Test token count for a sentence."""
        sentence = "This is a test sentence with multiple words."
        count = estimate_token_count(sentence)
        # Should be roughly word count, varies by tokenizer
        assert 5 < count < 20

    def test_long_text(self):
        """Test token count scales with text length."""
        short = "Hello world."
        long = short * 100
        short_count = estimate_token_count(short)
        long_count = estimate_token_count(long)
        # Long text should have proportionally more tokens
        assert long_count > short_count * 50

    def test_special_characters(self):
        """Test token count handles special characters."""
        text = "Hello! @#$%^&*() World?"
        count = estimate_token_count(text)
        assert count > 0

    def test_unicode(self):
        """Test token count handles unicode characters."""
        text = "Hello world! Caf\u00e9 \u4e2d\u6587 \u0440\u0443\u0441\u0441\u043a\u0438\u0439"
        count = estimate_token_count(text)
        assert count > 0


class TestTruncateFromFront:
    """Test the truncate_from_front function."""

    def test_no_truncation_needed(self):
        """Test that short text is not truncated."""
        text = "This is a short text."
        max_tokens = 1000

        result, metadata = truncate_from_front(text, max_tokens)

        assert result == text
        assert metadata['truncated'] is False
        assert metadata['tokens_removed'] == 0
        assert metadata['original_tokens'] == metadata['final_tokens']

    def test_truncation_applied(self):
        """Test that long text is truncated."""
        # Create text that exceeds limit with enough front content to truncate
        # Need text long enough that chars_to_remove < len(front_portion)
        text = "A" * 20000 + " " + "End question here?"
        max_tokens = 100

        result, metadata = truncate_from_front(text, max_tokens, preserve_chars_from_end=50)

        assert metadata['truncated'] is True
        assert metadata['tokens_removed'] > 0
        # Question at end should be preserved
        assert "End question here?" in result

    def test_preserves_end_content(self):
        """Test that content at end is preserved."""
        important_end = "IMPORTANT: This is the question you must answer?"
        text = "A" * 10000 + " " + important_end
        max_tokens = 100

        result, metadata = truncate_from_front(text, max_tokens, preserve_chars_from_end=len(important_end) + 10)

        # End should be preserved even after truncation
        assert important_end in result

    def test_metadata_accuracy(self):
        """Test that metadata is accurate."""
        text = "Word " * 200  # About 200-400 tokens
        max_tokens = 50

        result, metadata = truncate_from_front(text, max_tokens)

        # Verify metadata fields
        assert 'original_tokens' in metadata
        assert 'final_tokens' in metadata
        assert 'tokens_removed' in metadata
        assert 'truncated' in metadata

        # Tokens removed should equal difference
        assert metadata['original_tokens'] - metadata['final_tokens'] == metadata['tokens_removed']

    def test_finds_sentence_boundary(self):
        """Test that truncation finds sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. " * 50 + "Final question?"
        max_tokens = 100

        result, metadata = truncate_from_front(text, max_tokens)

        # Should start at a sentence boundary (capital or after period)
        if metadata['truncated']:
            # Result should not start mid-word (heuristic check)
            first_word = result.split()[0] if result.split() else ""
            # First character should be alphanumeric or typical sentence start
            assert first_word[0].isalnum() or first_word[0] in '"\'('

    def test_very_short_text_not_truncated(self):
        """Test that text shorter than preserve_chars is not truncated."""
        text = "Short text."
        max_tokens = 5  # Very low limit
        preserve = 3000  # Default

        result, metadata = truncate_from_front(text, max_tokens, preserve_chars_from_end=preserve)

        # Text is shorter than preserve_chars, so no truncation
        assert result == text
        assert metadata['truncated'] is False

    def test_iterative_truncation(self):
        """Test that iterative truncation works for stubborn cases."""
        # Very long text that needs multiple iterations
        text = "A" * 50000 + " " + "Question at end?"
        max_tokens = 200

        result, metadata = truncate_from_front(text, max_tokens, preserve_chars_from_end=50)

        # Should be truncated
        assert metadata['truncated'] is True
        # Should reduce token count significantly
        assert metadata['tokens_removed'] > 0

    def test_custom_preserve_chars(self):
        """Test custom preserve_chars_from_end parameter."""
        text = "Front content. " * 100 + "Very important ending that must be kept entirely."
        max_tokens = 100
        preserve = 100  # Preserve last 100 chars

        result, metadata = truncate_from_front(text, max_tokens, preserve_chars_from_end=preserve)

        if metadata['truncated']:
            # Last 100 chars should be preserved
            original_end = text[-preserve:]
            assert original_end in result

    def test_tokenizer_used_correctly(self):
        """Test that the o200k_base tokenizer is being used."""
        text = "Test tokenizer encoding"

        # Compare with direct tokenizer use
        expected_tokens = len(TOKENIZER.encode(text, disallowed_special=()))
        actual_tokens = estimate_token_count(text)

        assert actual_tokens == expected_tokens

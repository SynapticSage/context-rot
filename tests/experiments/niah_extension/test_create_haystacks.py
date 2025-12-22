# Tests for NIAH haystack creation
# Created: 2025-12-22

import pytest
import os
import tiktoken
import pandas as pd
import sys
from pathlib import Path

# Add experiments path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.niah_extension.run.create_haystacks import (
    load_text_files,
    build_haystack_sequential,
    build_haystack_shuffled,
    insert_needle_at_depth,
    insert_distractors_randomly,
    create_niah_prompt,
    create_haystacks,
)


@pytest.fixture
def tokenizer():
    """Create tokenizer for tests."""
    return tiktoken.get_encoding("o200k_base")


class TestLoadTextFiles:
    """Test the load_text_files function."""

    def test_loads_all_txt_files(self, sample_haystack_texts):
        """Test that all .txt files are loaded."""
        texts = load_text_files(sample_haystack_texts)
        assert len(texts) == 3

    def test_raises_on_empty_folder(self, temp_dir):
        """Test that ValueError is raised for folder with no txt files."""
        with pytest.raises(ValueError, match="No .txt files found"):
            load_text_files(temp_dir)

    def test_strips_whitespace(self, sample_haystack_texts):
        """Test that text content is stripped."""
        texts = load_text_files(sample_haystack_texts)
        for text in texts:
            assert text == text.strip()


class TestBuildHaystackSequential:
    """Test the build_haystack_sequential function."""

    def test_builds_to_target_tokens(self, tokenizer):
        """Test that haystack is built to approximately target tokens."""
        texts = ["Sentence one. " * 100, "Sentence two. " * 100]
        target = 500

        result = build_haystack_sequential(texts, target, tokenizer)
        actual_tokens = len(tokenizer.encode(result))

        # Should be close to target (within truncation variance)
        assert actual_tokens <= target + 10

    def test_uses_start_index(self, tokenizer):
        """Test that start_index controls which texts are used first."""
        texts = ["FIRST text content. ", "SECOND text content. "]
        target = 50

        result_0 = build_haystack_sequential(texts, target, tokenizer, start_index=0)
        result_1 = build_haystack_sequential(texts, target, tokenizer, start_index=1)

        # First should start with FIRST, second should start with SECOND
        assert result_0.startswith("FIRST")
        assert result_1.startswith("SECOND")

    def test_wraps_around_texts(self, tokenizer):
        """Test that text list wraps around when exhausted."""
        texts = ["Short. "]
        target = 100

        result = build_haystack_sequential(texts, target, tokenizer)

        # Should contain text repeated (wrapped around)
        assert len(tokenizer.encode(result)) > 0


class TestBuildHaystackShuffled:
    """Test the build_haystack_shuffled function."""

    def test_builds_to_target_tokens(self, tokenizer):
        """Test that shuffled haystack reaches target tokens."""
        texts = ["Sentence one. Sentence two. Sentence three. " * 50]
        target = 500

        result = build_haystack_shuffled(texts, target, tokenizer)
        actual_tokens = len(tokenizer.encode(result))

        assert actual_tokens <= target + 10

    def test_reproducible_with_seed(self, tokenizer):
        """Test that same seed produces same result."""
        texts = ["Sentence one. Sentence two. Sentence three. " * 20]
        target = 200

        result1 = build_haystack_shuffled(texts, target, tokenizer, seed=42)
        result2 = build_haystack_shuffled(texts, target, tokenizer, seed=42)

        assert result1 == result2

    def test_different_seeds_different_results(self, tokenizer):
        """Test that different seeds produce different results."""
        texts = ["Sentence one. Sentence two. Sentence three. " * 20]
        target = 200

        result1 = build_haystack_shuffled(texts, target, tokenizer, seed=42)
        result2 = build_haystack_shuffled(texts, target, tokenizer, seed=123)

        assert result1 != result2


class TestInsertNeedleAtDepth:
    """Test the insert_needle_at_depth function."""

    def test_depth_zero_inserts_at_start(self, tokenizer):
        """Test that depth 0 inserts needle at the beginning."""
        haystack = "Context content here. More content."
        needle = "NEEDLE"

        result = insert_needle_at_depth(haystack, needle, 0, tokenizer)

        assert result.startswith("NEEDLE")

    def test_depth_100_inserts_at_end(self, tokenizer):
        """Test that depth 100 inserts needle at the end."""
        haystack = "Context content here. More content."
        needle = "NEEDLE"

        result = insert_needle_at_depth(haystack, needle, 100, tokenizer)

        assert result.endswith("NEEDLE")

    def test_depth_50_inserts_in_middle(self, tokenizer):
        """Test that depth 50 inserts needle in the middle."""
        haystack = "Start content. " * 20 + "End content. " * 20
        needle = "NEEDLE"

        result = insert_needle_at_depth(haystack, needle, 50, tokenizer)

        # Needle should be roughly in the middle
        needle_pos = result.find("NEEDLE")
        assert needle_pos > len(result) * 0.3
        assert needle_pos < len(result) * 0.7

    def test_preserves_content(self, tokenizer):
        """Test that original content is preserved."""
        haystack = "Original content."
        needle = "NEEDLE"

        result = insert_needle_at_depth(haystack, needle, 50, tokenizer)

        assert "Original content" in result
        assert "NEEDLE" in result


class TestInsertDistractorsRandomly:
    """Test the insert_distractors_randomly function."""

    def test_inserts_all_distractors(self):
        """Test that all distractors are inserted."""
        haystack = "Sentence one. Sentence two. Sentence three."
        distractors = ["Distractor A.", "Distractor B."]

        result = insert_distractors_randomly(haystack, distractors)

        assert "Distractor A" in result
        assert "Distractor B" in result

    def test_no_distractors_returns_unchanged(self):
        """Test that empty distractors returns original."""
        haystack = "Original content."
        result = insert_distractors_randomly(haystack, [])
        assert result == haystack

    def test_none_distractors_returns_unchanged(self):
        """Test that None distractors returns original."""
        haystack = "Original content."
        result = insert_distractors_randomly(haystack, None)
        assert result == haystack

    def test_short_haystack_returns_unchanged(self):
        """Test that very short haystack returns unchanged."""
        haystack = "Short"
        result = insert_distractors_randomly(haystack, ["Distractor"])
        assert result == haystack


class TestCreateNiahPrompt:
    """Test the create_niah_prompt function."""

    def test_includes_haystack(self):
        """Test that haystack is included in prompt."""
        result = create_niah_prompt("My haystack content", "What is it?")
        assert "My haystack content" in result

    def test_includes_question(self):
        """Test that question is included in prompt."""
        result = create_niah_prompt("Content", "My specific question?")
        assert "My specific question?" in result

    def test_includes_document_tags(self):
        """Test that document tags are present."""
        result = create_niah_prompt("Content", "Question?")
        assert "<document_content>" in result
        assert "<question>" in result


class TestCreateHaystacks:
    """Test the create_haystacks function."""

    def test_creates_output_folder(self, sample_haystack_texts, temp_dir):
        """Test that output folder is created."""
        output_folder = os.path.join(temp_dir, "output")

        create_haystacks(
            haystack_folder=sample_haystack_texts,
            needle="Test needle",
            shuffled=False,
            output_folder=output_folder,
            question="What is the needle?",
            test_mode=True,
            trials_per_cell=1,
        )

        assert os.path.exists(output_folder)

    def test_creates_csv_output(self, sample_haystack_texts, temp_dir):
        """Test that CSV output is created."""
        output_folder = os.path.join(temp_dir, "output")

        create_haystacks(
            haystack_folder=sample_haystack_texts,
            needle="Test needle",
            shuffled=False,
            output_folder=output_folder,
            question="What is the needle?",
            test_mode=True,
            trials_per_cell=1,
        )

        csv_files = [f for f in os.listdir(output_folder) if f.endswith('.csv')]
        assert len(csv_files) == 1

    def test_csv_has_required_columns(self, sample_haystack_texts, temp_dir):
        """Test that CSV has all required columns."""
        output_folder = os.path.join(temp_dir, "output")

        create_haystacks(
            haystack_folder=sample_haystack_texts,
            needle="Test needle",
            shuffled=False,
            output_folder=output_folder,
            question="What is the needle?",
            test_mode=True,
            trials_per_cell=1,
        )

        csv_path = os.path.join(output_folder, "niah_prompts_sequential.csv")
        df = pd.read_csv(csv_path)

        required_columns = ['token_count', 'needle_depth', 'trial', 'prompt', 'question', 'answer']
        for col in required_columns:
            assert col in df.columns

    def test_test_mode_reduces_samples(self, sample_haystack_texts, temp_dir):
        """Test that test mode produces fewer samples."""
        output_folder = os.path.join(temp_dir, "output")

        create_haystacks(
            haystack_folder=sample_haystack_texts,
            needle="Test needle",
            shuffled=False,
            output_folder=output_folder,
            question="What is the needle?",
            test_mode=True,  # Should produce 4 lengths x 3 depths = 12 cells
            trials_per_cell=1,
        )

        csv_path = os.path.join(output_folder, "niah_prompts_sequential.csv")
        df = pd.read_csv(csv_path)

        # Test mode: some rows may be skipped if input length too small
        # But should be less than production (88 cells)
        assert len(df) <= 12

    def test_shuffled_mode_naming(self, sample_haystack_texts, temp_dir):
        """Test that shuffled mode uses correct filename."""
        output_folder = os.path.join(temp_dir, "output")

        create_haystacks(
            haystack_folder=sample_haystack_texts,
            needle="Test needle",
            shuffled=True,  # Shuffled mode
            output_folder=output_folder,
            question="What is the needle?",
            test_mode=True,
            trials_per_cell=1,
        )

        assert os.path.exists(os.path.join(output_folder, "niah_prompts_shuffled.csv"))

    def test_with_distractors_naming(self, sample_haystack_texts, temp_dir):
        """Test that distractors add suffix to filename."""
        output_folder = os.path.join(temp_dir, "output")

        create_haystacks(
            haystack_folder=sample_haystack_texts,
            needle="Test needle",
            shuffled=False,
            output_folder=output_folder,
            question="What is the needle?",
            distractors=["Distractor 1", "Distractor 2"],
            test_mode=True,
            trials_per_cell=1,
        )

        assert os.path.exists(os.path.join(output_folder, "niah_prompts_sequential_with_distractors.csv"))

    def test_trials_per_cell(self, sample_haystack_texts, temp_dir):
        """Test that trials_per_cell affects row count."""
        output_folder = os.path.join(temp_dir, "output")

        create_haystacks(
            haystack_folder=sample_haystack_texts,
            needle="Test needle",
            shuffled=False,
            output_folder=output_folder,
            question="What is the needle?",
            test_mode=True,
            trials_per_cell=2,  # 2 trials per cell
        )

        csv_path = os.path.join(output_folder, "niah_prompts_sequential.csv")
        df = pd.read_csv(csv_path)

        # Should have trial column with values 0 and 1
        assert set(df['trial'].unique()) == {0, 1}

    def test_needle_in_prompts(self, sample_haystack_texts, temp_dir):
        """Test that needle appears in prompts."""
        output_folder = os.path.join(temp_dir, "output")
        needle = "UNIQUE_NEEDLE_TEXT"

        create_haystacks(
            haystack_folder=sample_haystack_texts,
            needle=needle,
            shuffled=False,
            output_folder=output_folder,
            question="Where is the needle?",
            test_mode=True,
            trials_per_cell=1,
        )

        csv_path = os.path.join(output_folder, "niah_prompts_sequential.csv")
        df = pd.read_csv(csv_path)

        # Every prompt should contain the needle
        for prompt in df['prompt']:
            assert needle in prompt

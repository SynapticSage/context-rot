# Tests for LLM Judge
# Created: 2025-12-22

import pytest
import pandas as pd
import os
from unittest.mock import patch, MagicMock
from experiments.models.llm_judge import LLMJudge


class TestLLMJudgeInit:
    """Test LLMJudge initialization."""

    @patch('experiments.models.llm_judge.OpenAIProvider')
    def test_stores_prompt_template(self, mock_provider):
        """Test that prompt template is stored."""
        judge = LLMJudge(prompt="Test: {output}", model_name="gpt-4")
        assert judge.prompt == "Test: {output}"

    @patch('experiments.models.llm_judge.OpenAIProvider')
    def test_stores_column_names(self, mock_provider):
        """Test that column names are stored."""
        judge = LLMJudge(
            prompt="Test",
            output_column="my_output",
            question_column="my_question",
            correct_answer_column="my_answer",
        )
        assert judge.output_column == "my_output"
        assert judge.question_column == "my_question"
        assert judge.correct_answer_column == "my_answer"

    @patch('experiments.models.llm_judge.OpenAIProvider')
    def test_default_model_name(self, mock_provider):
        """Test default model name."""
        judge = LLMJudge(prompt="Test")
        assert judge.model_name == "gpt-4.1-2025-04-14"

    @patch('experiments.models.llm_judge.OpenAIProvider')
    def test_creates_provider(self, mock_provider):
        """Test that OpenAI provider is created."""
        judge = LLMJudge(prompt="Test")
        mock_provider.assert_called_once()


class TestLoadDistractors:
    """Test the _load_distractors method."""

    @patch('experiments.models.llm_judge.OpenAIProvider')
    def test_loads_distractors_from_json(self, mock_provider, sample_distractors_json):
        """Test loading distractors from JSON file."""
        judge = LLMJudge(prompt="Test", distractors_file=sample_distractors_json)

        assert "Distractor one content" in judge.distractors_text
        assert "Distractor two content" in judge.distractors_text

    @patch('experiments.models.llm_judge.OpenAIProvider')
    def test_formats_distractors_with_numbers(self, mock_provider, sample_distractors_json):
        """Test that distractors are numbered."""
        judge = LLMJudge(prompt="Test", distractors_file=sample_distractors_json)

        # Should have "0. " and "1. " prefixes
        assert "0. " in judge.distractors_text
        assert "1. " in judge.distractors_text

    @patch('experiments.models.llm_judge.OpenAIProvider')
    def test_no_distractors_when_file_not_provided(self, mock_provider):
        """Test that distractors_text is empty when no file provided."""
        judge = LLMJudge(prompt="Test")
        assert judge.distractors_text == ""


class TestFormatPrompt:
    """Test the _format_prompt method."""

    @patch('experiments.models.llm_judge.OpenAIProvider')
    def test_formats_basic_placeholders(self, mock_provider):
        """Test formatting of basic placeholders."""
        judge = LLMJudge(
            prompt="Output: {output}, Question: {question}, Answer: {correct_answer}"
        )

        result = judge._format_prompt("test output", "test question", "test answer")

        assert "Output: test output" in result
        assert "Question: test question" in result
        assert "Answer: test answer" in result

    @patch('experiments.models.llm_judge.OpenAIProvider')
    def test_formats_with_distractors(self, mock_provider, sample_distractors_json):
        """Test formatting with distractors placeholder."""
        judge = LLMJudge(
            prompt="Output: {output}, Distractors: {distractors}",
            distractors_file=sample_distractors_json,
        )

        result = judge._format_prompt("test output", "test question", "test answer")

        assert "Output: test output" in result
        assert "Distractor one content" in result


class TestProcessForEvaluation:
    """Test the _process_for_evaluation method."""

    @patch('experiments.models.llm_judge.OpenAIProvider')
    def test_processes_all_indices(self, mock_provider_class, temp_dir):
        """Test that all indices are processed."""
        mock_provider = MagicMock()
        mock_provider.process_single_prompt.return_value = (0, "true")
        mock_provider_class.return_value = mock_provider

        judge = LLMJudge(prompt="{output}")
        output_path = os.path.join(temp_dir, "output.csv")

        input_df = pd.DataFrame({
            'output': ['Response 1', 'Response 2'],
            'question': ['Q1', 'Q2'],
            'answer': ['A1', 'A2'],
        })
        output_df = input_df.copy()
        output_df['llm_judge_output'] = None

        judge._process_for_evaluation(input_df, output_df, [0, 1], output_path)

        assert mock_provider.process_single_prompt.call_count == 2

    @patch('experiments.models.llm_judge.OpenAIProvider')
    def test_saves_results_to_csv(self, mock_provider_class, temp_dir):
        """Test that results are saved to CSV."""
        mock_provider = MagicMock()
        mock_provider.process_single_prompt.return_value = (0, "true")
        mock_provider_class.return_value = mock_provider

        judge = LLMJudge(prompt="{output}")
        output_path = os.path.join(temp_dir, "output.csv")

        input_df = pd.DataFrame({
            'output': ['Response'],
            'question': ['Q'],
            'answer': ['A'],
        })
        output_df = input_df.copy()
        output_df['llm_judge_output'] = None

        judge._process_for_evaluation(input_df, output_df, [0], output_path)

        assert os.path.exists(output_path)


class TestEvaluate:
    """Test the evaluate method."""

    @patch('experiments.models.llm_judge.OpenAIProvider')
    def test_reads_input_csv(self, mock_provider_class, temp_dir):
        """Test that input CSV is read."""
        mock_provider = MagicMock()
        mock_provider.process_single_prompt.return_value = (0, "true")
        mock_provider.create_batches.return_value = [[0]]
        mock_provider_class.return_value = mock_provider

        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")

        pd.DataFrame({
            'output': ['Test'],
            'question': ['Q'],
            'answer': ['A'],
        }).to_csv(input_path, index=False)

        judge = LLMJudge(prompt="{output}")
        judge.evaluate(input_path, output_path, max_context_length=1000, max_tokens_per_minute=10000)

        assert os.path.exists(output_path)

    @patch('experiments.models.llm_judge.OpenAIProvider')
    def test_resumes_from_existing(self, mock_provider_class, temp_dir):
        """Test resuming from existing output file."""
        mock_provider = MagicMock()
        mock_provider.process_single_prompt.return_value = (1, "true")
        mock_provider.create_batches.return_value = [[1]]
        mock_provider_class.return_value = mock_provider

        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")

        # Create input
        pd.DataFrame({
            'output': ['Test 1', 'Test 2'],
            'question': ['Q1', 'Q2'],
            'answer': ['A1', 'A2'],
        }).to_csv(input_path, index=False)

        # Create existing output with one completed
        pd.DataFrame({
            'output': ['Test 1', 'Test 2'],
            'question': ['Q1', 'Q2'],
            'answer': ['A1', 'A2'],
            'llm_judge_output': ['true', None],
        }).to_csv(output_path, index=False)

        judge = LLMJudge(prompt="{output}")
        judge.evaluate(input_path, output_path, max_context_length=1000, max_tokens_per_minute=10000)

        # Should only process row 1 (index 1)
        # The create_batches call should only include rows needing processing


class TestAnalyzeDistractors:
    """Test the analyze_distractors method."""

    @patch('experiments.models.llm_judge.OpenAIProvider')
    def test_filters_to_false_judge_outputs(self, mock_provider_class, temp_dir):
        """Test that only rows with llm_judge_output=False are analyzed."""
        mock_provider = MagicMock()
        mock_provider.process_single_prompt.return_value = (0, "distractor_1")
        mock_provider.create_batches.return_value = [[0]]
        mock_provider_class.return_value = mock_provider

        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")

        # Create input with mixed judge outputs
        # Note: pandas reads False as boolean False
        pd.DataFrame({
            'output': ['Test 1', 'Test 2'],
            'question': ['Q1', 'Q2'],
            'answer': ['A1', 'A2'],
            'token_count': [100, 100],
            'llm_judge_output': [False, True],  # Only first should be analyzed
        }).to_csv(input_path, index=False)

        judge = LLMJudge(prompt="{output}")
        judge.analyze_distractors(
            input_path, output_path,
            max_context_length=1000, max_tokens_per_minute=10000
        )

        # Check that output file was created with filtered results
        assert os.path.exists(output_path)
        result_df = pd.read_csv(output_path)
        # Should only have one row (the one with llm_judge_output=False)
        assert len(result_df) == 1

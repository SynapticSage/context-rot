# Tests for BaseProvider abstract class
# Created: 2025-12-22

import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import patch, MagicMock
from experiments.models.base_provider import BaseProvider


class ConcreteProvider(BaseProvider):
    """Concrete implementation for testing BaseProvider."""

    def __init__(self, response_func=None):
        self._response_func = response_func or (lambda p, m, t, i: (i, f"Response for {i}"))
        super().__init__()

    def process_single_prompt(self, prompt, model_name, max_output_tokens, index):
        return self._response_func(prompt, model_name, max_output_tokens, index)

    def get_client(self):
        return None


class TestBaseProviderInit:
    """Test BaseProvider initialization."""

    def test_init_sets_client(self):
        """Test that __init__ calls get_client."""
        provider = ConcreteProvider()
        assert provider.client is None  # Our concrete returns None

    def test_init_creates_csv_lock(self):
        """Test that __init__ creates threading lock."""
        provider = ConcreteProvider()
        assert provider._csv_lock is not None

    def test_token_tracker_initially_none(self):
        """Test that token_tracker starts as None."""
        provider = ConcreteProvider()
        assert provider.token_tracker is None


class TestGetContextSafetyMargin:
    """Test the get_context_safety_margin method."""

    def test_default_margin_is_one(self):
        """Test that default safety margin is 1.0 (no margin)."""
        provider = ConcreteProvider()
        assert provider.get_context_safety_margin() == 1.0


class TestGetDefaultTestSamples:
    """Test the _get_default_test_samples method."""

    def test_default_is_twenty(self):
        """Test that default test samples is 20."""
        provider = ConcreteProvider()
        assert provider._get_default_test_samples() == 20


class TestCreateBatches:
    """Test the create_batches method."""

    def test_single_batch_under_limit(self):
        """Test all rows fit in one batch when under limit."""
        provider = ConcreteProvider()
        df = pd.DataFrame({
            'token_count': [100, 200, 300],
        })

        batches = provider.create_batches(df, max_tokens_per_minute=1000)

        assert len(batches) == 1
        assert batches[0] == [0, 1, 2]

    def test_multiple_batches_when_exceeds_limit(self):
        """Test rows split into batches when exceeding limit."""
        provider = ConcreteProvider()
        df = pd.DataFrame({
            'token_count': [500, 500, 500],
        })

        batches = provider.create_batches(df, max_tokens_per_minute=800)

        # First batch: 500 tokens, second batch: 500 tokens, third batch: 500 tokens
        assert len(batches) == 3

    def test_single_large_row_own_batch(self):
        """Test that a row exceeding limit gets its own batch."""
        provider = ConcreteProvider()
        df = pd.DataFrame({
            'token_count': [100, 1500, 200],  # Middle row exceeds 1000 limit
        })

        batches = provider.create_batches(df, max_tokens_per_minute=1000)

        # Should have 3 batches: [0], [1], [2] or [0, 2] and [1]
        assert len(batches) >= 2
        # Large row should be in its own batch
        assert [1] in batches

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        provider = ConcreteProvider()
        df = pd.DataFrame({'token_count': []})

        batches = provider.create_batches(df, max_tokens_per_minute=1000)

        assert batches == []

    def test_preserves_index(self):
        """Test that original indices are preserved."""
        provider = ConcreteProvider()
        df = pd.DataFrame({
            'token_count': [100, 200, 300],
        }, index=[10, 20, 30])  # Non-standard indices

        batches = provider.create_batches(df, max_tokens_per_minute=1000)

        # Should use original indices
        assert batches[0] == [10, 20, 30]


class TestProcessBatch:
    """Test the process_batch method."""

    def test_successful_processing(self, temp_dir):
        """Test successful batch processing."""
        provider = ConcreteProvider()
        output_path = os.path.join(temp_dir, "output.csv")

        input_df = pd.DataFrame({
            'prompt': ['Test 1', 'Test 2'],
            'token_count': [100, 100],
        })
        output_df = input_df.copy()
        output_df['output'] = None
        output_df['_error_count'] = 0

        success_count = provider.process_batch(
            input_df, output_df, [0, 1],
            model_name="test-model",
            output_path=output_path,
            input_column='prompt',
            output_column='output',
        )

        assert success_count == 2
        assert output_df.loc[0, 'output'] == "Response for 0"
        assert output_df.loc[1, 'output'] == "Response for 1"
        assert os.path.exists(output_path)

    def test_error_handling(self, temp_dir):
        """Test that errors are captured and counted."""
        def error_response(p, m, t, i):
            return i, "ERROR: Test error"

        provider = ConcreteProvider(response_func=error_response)
        output_path = os.path.join(temp_dir, "output.csv")

        input_df = pd.DataFrame({
            'prompt': ['Test 1'],
            'token_count': [100],
        })
        output_df = input_df.copy()
        output_df['output'] = None
        output_df['_error_count'] = 0

        success_count = provider.process_batch(
            input_df, output_df, [0],
            model_name="test-model",
            output_path=output_path,
            input_column='prompt',
            output_column='output',
        )

        assert success_count == 0
        assert output_df.loc[0, '_error_count'] == 1

    def test_max_output_tokens_column(self, temp_dir):
        """Test that max_output_tokens is read from DataFrame if present."""
        received_tokens = []

        def capture_response(p, m, t, i):
            received_tokens.append(t)
            return i, "Response"

        provider = ConcreteProvider(response_func=capture_response)
        output_path = os.path.join(temp_dir, "output.csv")

        input_df = pd.DataFrame({
            'prompt': ['Test 1'],
            'token_count': [100],
            'max_output_tokens': [500],
        })
        output_df = input_df.copy()
        output_df['output'] = None
        output_df['_error_count'] = 0

        provider.process_batch(
            input_df, output_df, [0],
            model_name="test-model",
            output_path=output_path,
            input_column='prompt',
            output_column='output',
        )

        assert 500 in received_tokens


class TestMain:
    """Test the main workflow method."""

    @patch('experiments.models.base_provider.LiteLLMTokenTracker')
    def test_creates_output_directory(self, mock_tracker, temp_dir):
        """Test that output directory is created."""
        provider = ConcreteProvider()
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "subdir", "output.csv")

        # Create input file
        pd.DataFrame({
            'prompt': ['Test'],
            'token_count': [100],
        }).to_csv(input_path, index=False)

        provider.main(
            input_path=input_path,
            output_path=output_path,
            input_column='prompt',
            output_column='output',
            model_name='test-model',
            max_context_length=1000,
            max_tokens_per_minute=10000,
        )

        assert os.path.exists(os.path.dirname(output_path))

    @patch('experiments.models.base_provider.LiteLLMTokenTracker')
    def test_test_mode_sampling(self, mock_tracker, temp_dir):
        """Test that test mode reduces samples."""
        provider = ConcreteProvider()
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")

        # Create input with many rows
        pd.DataFrame({
            'prompt': [f'Test {i}' for i in range(100)],
            'token_count': [100] * 100,
        }).to_csv(input_path, index=False)

        provider.main(
            input_path=input_path,
            output_path=output_path,
            input_column='prompt',
            output_column='output',
            model_name='test-model',
            max_context_length=1000,
            max_tokens_per_minute=100000,
            test_mode=True,
        )

        output_df = pd.read_csv(output_path)
        # Default test samples is 20
        assert len(output_df) == 20

    @patch('experiments.models.base_provider.LiteLLMTokenTracker')
    def test_context_length_filtering(self, mock_tracker, temp_dir):
        """Test that rows exceeding context length are filtered."""
        provider = ConcreteProvider()
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")

        pd.DataFrame({
            'prompt': ['Short', 'Long'],
            'token_count': [100, 5000],
        }).to_csv(input_path, index=False)

        provider.main(
            input_path=input_path,
            output_path=output_path,
            input_column='prompt',
            output_column='output',
            model_name='test-model',
            max_context_length=1000,  # Only first row fits
            max_tokens_per_minute=10000,
        )

        output_df = pd.read_csv(output_path)
        assert len(output_df) == 1

    @patch('experiments.models.base_provider.LiteLLMTokenTracker')
    def test_resume_from_existing(self, mock_tracker, temp_dir):
        """Test resuming from existing output file."""
        provider = ConcreteProvider()
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")

        # Create input
        pd.DataFrame({
            'prompt': ['Test 1', 'Test 2', 'Test 3'],
            'token_count': [100, 100, 100],
        }).to_csv(input_path, index=False)

        # Create existing output with some completed
        pd.DataFrame({
            'token_count': [100, 100, 100],
            'output': ['Done', None, 'Done'],
            '_error_count': [0, 0, 0],
        }).to_csv(output_path, index=False)

        provider.main(
            input_path=input_path,
            output_path=output_path,
            input_column='prompt',
            output_column='output',
            model_name='test-model',
            max_context_length=1000,
            max_tokens_per_minute=10000,
        )

        output_df = pd.read_csv(output_path)
        # Only row 1 should have been processed (was None)
        assert output_df.loc[0, 'output'] == 'Done'  # Unchanged
        assert 'Response' in str(output_df.loc[1, 'output'])  # Now processed
        assert output_df.loc[2, 'output'] == 'Done'  # Unchanged

    @patch('experiments.models.base_provider.LiteLLMTokenTracker')
    def test_retry_errors_under_limit(self, mock_tracker, temp_dir):
        """Test that errors are retried up to max_retries."""
        provider = ConcreteProvider()
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")

        pd.DataFrame({
            'prompt': ['Test'],
            'token_count': [100],
        }).to_csv(input_path, index=False)

        # Create existing output with error but under retry limit
        pd.DataFrame({
            'token_count': [100],
            'output': ['ERROR: Previous error'],
            '_error_count': [1],
        }).to_csv(output_path, index=False)

        provider.main(
            input_path=input_path,
            output_path=output_path,
            input_column='prompt',
            output_column='output',
            model_name='test-model',
            max_context_length=1000,
            max_tokens_per_minute=10000,
            max_retries=2,
        )

        output_df = pd.read_csv(output_path)
        # Should have retried and gotten new response
        assert 'Response' in str(output_df.loc[0, 'output'])

    @patch('experiments.models.base_provider.LiteLLMTokenTracker')
    def test_skip_exceeded_retries(self, mock_tracker, temp_dir):
        """Test that rows exceeding retry limit are skipped."""
        provider = ConcreteProvider()
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")

        pd.DataFrame({
            'prompt': ['Test'],
            'token_count': [100],
        }).to_csv(input_path, index=False)

        # Create existing output with error at retry limit
        pd.DataFrame({
            'token_count': [100],
            'output': ['ERROR: Previous error'],
            '_error_count': [2],
        }).to_csv(output_path, index=False)

        provider.main(
            input_path=input_path,
            output_path=output_path,
            input_column='prompt',
            output_column='output',
            model_name='test-model',
            max_context_length=1000,
            max_tokens_per_minute=10000,
            max_retries=2,
        )

        output_df = pd.read_csv(output_path)
        # Should NOT have retried - still has error
        assert 'ERROR' in str(output_df.loc[0, 'output'])

    @patch('experiments.models.base_provider.LiteLLMTokenTracker')
    def test_test_mode_marks_output(self, mock_tracker, temp_dir):
        """Test that test mode adds metadata columns."""
        provider = ConcreteProvider()
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")

        pd.DataFrame({
            'prompt': ['Test'],
            'token_count': [100],
        }).to_csv(input_path, index=False)

        provider.main(
            input_path=input_path,
            output_path=output_path,
            input_column='prompt',
            output_column='output',
            model_name='test-model',
            max_context_length=1000,
            max_tokens_per_minute=10000,
            test_mode=True,
        )

        output_df = pd.read_csv(output_path)
        assert '_test_mode' in output_df.columns
        assert '_test_timestamp' in output_df.columns
        assert '_original_dataset_size' in output_df.columns

    @patch('experiments.models.base_provider.LiteLLMTokenTracker')
    def test_all_filtered_out_early_exit(self, mock_tracker, temp_dir):
        """Test early exit when all rows filtered by context length."""
        provider = ConcreteProvider()
        input_path = os.path.join(temp_dir, "input.csv")
        output_path = os.path.join(temp_dir, "output.csv")

        pd.DataFrame({
            'prompt': ['Test'],
            'token_count': [5000],  # Exceeds limit
        }).to_csv(input_path, index=False)

        provider.main(
            input_path=input_path,
            output_path=output_path,
            input_column='prompt',
            output_column='output',
            model_name='test-model',
            max_context_length=1000,
            max_tokens_per_minute=10000,
        )

        # Should not create output file when all filtered
        assert not os.path.exists(output_path)

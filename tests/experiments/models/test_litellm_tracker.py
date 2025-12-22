# Tests for LiteLLM Token Tracker
# Created: 2025-12-22

import pytest
import json
import os
from unittest.mock import MagicMock, patch
from experiments.models.litellm_tracker import LiteLLMTokenTracker


class TestLiteLLMTokenTrackerInit:
    """Test LiteLLMTokenTracker initialization."""

    def test_creates_output_directory(self, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        output_dir = os.path.join(temp_dir, "new_subdir")
        tracker = LiteLLMTokenTracker(output_dir=output_dir, experiment_name="test")

        assert os.path.exists(output_dir)

    def test_sets_paths_correctly(self, temp_dir):
        """Test that dashboard and summary paths are set correctly."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="myexp")

        assert str(tracker.dashboard_path).endswith("myexp_token_dashboard.txt")
        assert str(tracker.summary_path).endswith("myexp_token_summary.json")

    def test_initializes_counters_to_zero(self, temp_dir):
        """Test that counters start at zero."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")

        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_cost == 0.0
        assert tracker.call_count == 0
        assert tracker.model_stats == {}

    def test_writes_initial_dashboard(self, temp_dir):
        """Test that initial dashboard is written."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")

        assert os.path.exists(tracker.dashboard_path)


class TestLoadCheckpoint:
    """Test checkpoint loading functionality."""

    def test_loads_existing_checkpoint(self, temp_dir):
        """Test loading from existing checkpoint file."""
        # Create a checkpoint file
        checkpoint = {
            'total_input_tokens': 1000,
            'total_output_tokens': 500,
            'total_cost_usd': 0.05,
            'total_calls': 10,
            'model_stats': {'gpt-4': {'input_tokens': 1000, 'output_tokens': 500, 'cost': 0.05, 'calls': 10}},
        }
        summary_path = os.path.join(temp_dir, "test_token_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(checkpoint, f)

        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")

        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        assert tracker.total_cost == 0.05
        assert tracker.call_count == 10

    def test_handles_missing_checkpoint(self, temp_dir):
        """Test that missing checkpoint doesn't cause errors."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")

        # Should initialize to zeros
        assert tracker.call_count == 0

    def test_handles_corrupted_checkpoint(self, temp_dir):
        """Test that corrupted checkpoint is handled gracefully."""
        summary_path = os.path.join(temp_dir, "test_token_summary.json")
        with open(summary_path, 'w') as f:
            f.write("not valid json")

        # Should not raise, just log and continue
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")
        assert tracker.call_count == 0


class TestRegisterUnregister:
    """Test LiteLLM callback registration."""

    @patch('experiments.models.litellm_tracker.litellm')
    def test_register_adds_to_callbacks(self, mock_litellm, temp_dir):
        """Test that register adds tracker to LiteLLM callbacks."""
        mock_litellm.callbacks = []
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")

        tracker.register()

        assert tracker in mock_litellm.callbacks

    @patch('experiments.models.litellm_tracker.litellm')
    def test_register_idempotent(self, mock_litellm, temp_dir):
        """Test that register doesn't add duplicates."""
        mock_litellm.callbacks = []
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")

        tracker.register()
        tracker.register()

        assert mock_litellm.callbacks.count(tracker) == 1

    @patch('experiments.models.litellm_tracker.litellm')
    def test_unregister_removes_from_callbacks(self, mock_litellm, temp_dir):
        """Test that unregister removes tracker from callbacks."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")
        mock_litellm.callbacks = [tracker]

        tracker.unregister()

        assert tracker not in mock_litellm.callbacks


class TestTrackCall:
    """Test the track_call method."""

    def test_extracts_usage_from_response(self, temp_dir, mock_litellm_response):
        """Test that usage is extracted from response."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")

        tracker.track_call(mock_litellm_response, "gpt-4")

        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50
        assert tracker.call_count == 1

    def test_accumulates_across_calls(self, temp_dir, mock_litellm_response):
        """Test that values accumulate across multiple calls."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")

        tracker.track_call(mock_litellm_response, "gpt-4")
        tracker.track_call(mock_litellm_response, "gpt-4")

        assert tracker.total_input_tokens == 200
        assert tracker.total_output_tokens == 100
        assert tracker.call_count == 2

    def test_tracks_per_model_stats(self, temp_dir, mock_litellm_response):
        """Test that per-model stats are tracked."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")

        tracker.track_call(mock_litellm_response, "gpt-4")

        assert "gpt-4" in tracker.model_stats
        assert tracker.model_stats["gpt-4"]["calls"] == 1
        assert tracker.model_stats["gpt-4"]["input_tokens"] == 100

    def test_handles_missing_usage(self, temp_dir):
        """Test handling of response without usage attribute."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")
        response = MagicMock(spec=[])  # No usage attribute

        tracker.track_call(response, "gpt-4")

        # Should not crash, counters unchanged
        assert tracker.call_count == 0

    def test_handles_none_usage(self, temp_dir):
        """Test handling of response with None usage."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")
        response = MagicMock()
        response.usage = None

        tracker.track_call(response, "gpt-4")

        assert tracker.call_count == 0

    def test_extracts_cost_from_hidden_params(self, temp_dir, mock_litellm_response):
        """Test that cost is extracted from _hidden_params."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")

        tracker.track_call(mock_litellm_response, "gpt-4")

        assert tracker.total_cost == 0.001

    def test_saves_checkpoint_every_10_calls(self, temp_dir, mock_litellm_response):
        """Test that checkpoint is saved every 10 calls."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")

        for _ in range(10):
            tracker.track_call(mock_litellm_response, "gpt-4")

        # Check that summary file was written
        assert os.path.exists(tracker.summary_path)
        with open(tracker.summary_path) as f:
            data = json.load(f)
        assert data['total_calls'] == 10


class TestLogSuccessEvent:
    """Test the LiteLLM callback method."""

    def test_callback_extracts_model_name(self, temp_dir, mock_litellm_response):
        """Test that callback extracts model from kwargs."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")
        kwargs = {'model': 'gpt-3.5-turbo'}

        tracker.log_success_event(kwargs, mock_litellm_response, None, None)

        assert "gpt-3.5-turbo" in tracker.model_stats

    def test_callback_handles_exceptions(self, temp_dir):
        """Test that callback doesn't crash on exceptions."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")
        kwargs = {'model': 'test'}

        # Pass invalid response
        tracker.log_success_event(kwargs, None, None, None)

        # Should not raise
        assert tracker.call_count == 0


class TestWriteDashboard:
    """Test dashboard file writing."""

    def test_dashboard_contains_totals(self, temp_dir, mock_litellm_response):
        """Test that dashboard contains total stats."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")
        tracker.track_call(mock_litellm_response, "gpt-4")
        tracker._write_dashboard()

        content = tracker.dashboard_path.read_text()
        assert "TOTALS:" in content
        assert "100" in content  # Input tokens
        assert "50" in content  # Output tokens

    def test_dashboard_contains_model_breakdown(self, temp_dir, mock_litellm_response):
        """Test that dashboard contains per-model breakdown."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")
        tracker.track_call(mock_litellm_response, "gpt-4")
        tracker._write_dashboard()

        content = tracker.dashboard_path.read_text()
        assert "PER-MODEL BREAKDOWN:" in content
        assert "gpt-4" in content


class TestSaveFinal:
    """Test final summary saving."""

    def test_saves_json_summary(self, temp_dir, mock_litellm_response):
        """Test that final summary is saved as JSON."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")
        tracker.track_call(mock_litellm_response, "gpt-4")

        tracker.save_final()

        assert os.path.exists(tracker.summary_path)
        with open(tracker.summary_path) as f:
            data = json.load(f)
        assert data['total_calls'] == 1
        assert data['total_input_tokens'] == 100

    def test_returns_formatted_report(self, temp_dir, mock_litellm_response):
        """Test that save_final returns formatted report string."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")
        tracker.track_call(mock_litellm_response, "gpt-4")

        report = tracker.save_final()

        assert isinstance(report, str)
        assert "FINAL TOKEN USAGE SUMMARY" in report
        assert "test" in report  # experiment name

    def test_summary_contains_all_fields(self, temp_dir, mock_litellm_response):
        """Test that summary JSON contains all expected fields."""
        tracker = LiteLLMTokenTracker(output_dir=temp_dir, experiment_name="test")
        tracker.track_call(mock_litellm_response, "gpt-4")
        tracker.save_final()

        with open(tracker.summary_path) as f:
            data = json.load(f)

        expected_fields = [
            'experiment_name', 'timestamp', 'elapsed_minutes', 'total_calls',
            'total_input_tokens', 'total_output_tokens', 'total_tokens',
            'tokens_per_minute', 'total_cost_usd', 'model_stats'
        ]
        for field in expected_fields:
            assert field in data

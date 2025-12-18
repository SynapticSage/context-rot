"""
LiteLLM Token Tracker for Context Rot Research
Created: 2025-11-24
Modified: 2025-11-29 (integrated CustomLogger callback)

Provides real-time token usage tracking and cost estimation for all model providers.
Uses LiteLLM's unified API and CustomLogger callback to track tokens automatically.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import litellm
from litellm.integrations.custom_logger import CustomLogger


class LiteLLMTokenTracker(CustomLogger):
    """
    Tracks token usage and costs across all API calls during experiment runs.

    Features:
    - Real-time dashboard file updates (can be tailed during execution)
    - Per-provider and per-model breakdowns
    - Automatic cost calculation using LiteLLM's cost tables
    - Rate tracking (tokens/minute)
    - Final summary report
    - LiteLLM CustomLogger integration for automatic tracking

    Usage:
        tracker = LiteLLMTokenTracker(output_dir="results", experiment_name="test")
        tracker.register()  # Register with LiteLLM's callback system
        # ... make litellm.completion() calls ...
        tracker.save_final()
    """

    def __init__(self, output_dir: str = "results", experiment_name: str = "experiment"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.dashboard_path = self.output_dir / f"{experiment_name}_token_dashboard.txt"
        self.summary_path = self.output_dir / f"{experiment_name}_token_summary.json"

        self.start_time = time.time()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0

        # Per-model tracking
        self.model_stats: Dict[str, Dict] = {}

        # Load existing state if available (resume from checkpoint)
        self._load_checkpoint()

        # Initialize dashboard
        self._write_dashboard()

    def _load_checkpoint(self) -> None:
        """Load existing token tracking state from summary file if it exists."""
        if self.summary_path.exists():
            try:
                with open(self.summary_path) as f:
                    state = json.load(f)
                self.total_input_tokens = state.get('total_input_tokens', 0)
                self.total_output_tokens = state.get('total_output_tokens', 0)
                self.total_cost = state.get('total_cost_usd', 0.0)
                self.call_count = state.get('total_calls', 0)
                self.model_stats = state.get('model_stats', {})
                print(f"[TokenTracker] Resumed from checkpoint: {self.call_count} calls, {self.total_input_tokens + self.total_output_tokens:,} tokens")
            except Exception as e:
                print(f"[TokenTracker] Could not load checkpoint: {e}")

    def register(self) -> None:
        """Register this tracker with LiteLLM's callback system."""
        if self not in litellm.callbacks:
            litellm.callbacks.append(self)

    def unregister(self) -> None:
        """Remove this tracker from LiteLLM's callback system."""
        if self in litellm.callbacks:
            litellm.callbacks.remove(self)

    def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """
        LiteLLM callback - called automatically after successful API calls.

        This enables automatic tracking without manually calling track_call().
        """
        try:
            model_name = kwargs.get('model', 'unknown')
            print(f"[TokenTracker] log_success_event triggered for {model_name}")
            self.track_call(response_obj, model_name)
        except Exception as e:
            # Don't let callback errors disrupt main flow
            print(f"[TokenTracker] Warning: callback error: {e}")

    def track_call(self, response, model_name: str) -> None:
        """
        Extract token usage from LiteLLM response and update tracking.

        Args:
            response: LiteLLM response object with usage attribute
            model_name: Model identifier for per-model tracking
        """
        if not hasattr(response, 'usage') or response.usage is None:
            # Debug: log response structure when usage is missing
            print(f"[TokenTracker] DEBUG: No usage in response. Response type: {type(response)}")
            print(f"[TokenTracker] DEBUG: Response attrs: {[a for a in dir(response) if not a.startswith('_')]}")
            if hasattr(response, 'model_extra'):
                print(f"[TokenTracker] DEBUG: model_extra: {response.model_extra}")
            return

        usage = response.usage
        input_tokens = getattr(usage, 'prompt_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0)

        # LiteLLM includes _hidden_params with cost if available
        cost = 0.0
        if hasattr(response, '_hidden_params') and response._hidden_params:
            cost = response._hidden_params.get('response_cost', 0.0)

        # Update totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.call_count += 1

        # Update per-model stats
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'cost': 0.0,
                'calls': 0
            }

        self.model_stats[model_name]['input_tokens'] += input_tokens
        self.model_stats[model_name]['output_tokens'] += output_tokens
        self.model_stats[model_name]['cost'] += cost
        self.model_stats[model_name]['calls'] += 1

        # Update dashboard and save checkpoint every 10 calls
        if self.call_count % 10 == 0:
            self._write_dashboard()
            self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        """Save current state to summary file for resume capability."""
        total_tokens = self.total_input_tokens + self.total_output_tokens
        elapsed = time.time() - self.start_time
        elapsed_min = elapsed / 60.0
        tokens_per_min = total_tokens / elapsed_min if elapsed_min > 0 else 0

        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'elapsed_minutes': round(elapsed_min, 2),
            'total_calls': self.call_count,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': total_tokens,
            'tokens_per_minute': round(tokens_per_min, 0),
            'total_cost_usd': round(self.total_cost, 4),
            'model_stats': self.model_stats
        }
        self.summary_path.write_text(json.dumps(summary, indent=2))

    def _write_dashboard(self) -> None:
        """Write live dashboard to file for real-time monitoring."""
        elapsed = time.time() - self.start_time
        elapsed_min = elapsed / 60.0

        # Calculate rates
        total_tokens = self.total_input_tokens + self.total_output_tokens
        tokens_per_min = total_tokens / elapsed_min if elapsed_min > 0 else 0

        lines = [
            "=" * 80,
            f"Token Usage Dashboard - {self.experiment_name}",
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Elapsed time: {elapsed_min:.1f} minutes",
            "=" * 80,
            "",
            "TOTALS:",
            f"  API calls: {self.call_count:,}",
            f"  Input tokens: {self.total_input_tokens:,}",
            f"  Output tokens: {self.total_output_tokens:,}",
            f"  Total tokens: {total_tokens:,}",
            f"  Rate: {tokens_per_min:,.0f} tokens/min",
            f"  Estimated cost: ${self.total_cost:.4f}",
            ""
        ]

        if self.model_stats:
            lines.append("PER-MODEL BREAKDOWN:")
            for model, stats in sorted(self.model_stats.items()):
                total_model_tokens = stats['input_tokens'] + stats['output_tokens']
                lines.extend([
                    f"  {model}:",
                    f"    Calls: {stats['calls']:,}",
                    f"    Input: {stats['input_tokens']:,} tokens",
                    f"    Output: {stats['output_tokens']:,} tokens",
                    f"    Total: {total_model_tokens:,} tokens",
                    f"    Cost: ${stats['cost']:.4f}",
                    ""
                ])

        lines.extend([
            "=" * 80,
            "To monitor live: tail -f " + str(self.dashboard_path),
            "=" * 80
        ])

        self.dashboard_path.write_text("\n".join(lines))

    def save_final(self) -> str:
        """
        Write final summary and return formatted report.

        Returns:
            Formatted summary string
        """
        elapsed = time.time() - self.start_time
        elapsed_min = elapsed / 60.0
        total_tokens = self.total_input_tokens + self.total_output_tokens
        tokens_per_min = total_tokens / elapsed_min if elapsed_min > 0 else 0

        # Write final dashboard
        self._write_dashboard()

        # Save JSON summary
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'elapsed_minutes': round(elapsed_min, 2),
            'total_calls': self.call_count,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': total_tokens,
            'tokens_per_minute': round(tokens_per_min, 0),
            'total_cost_usd': round(self.total_cost, 4),
            'model_stats': self.model_stats
        }

        self.summary_path.write_text(json.dumps(summary, indent=2))

        # Return formatted report
        report = [
            "",
            "=" * 80,
            "FINAL TOKEN USAGE SUMMARY",
            "=" * 80,
            f"Experiment: {self.experiment_name}",
            f"Duration: {elapsed_min:.1f} minutes",
            f"Total API calls: {self.call_count:,}",
            f"Total tokens: {total_tokens:,} ({self.total_input_tokens:,} in + {self.total_output_tokens:,} out)",
            f"Average rate: {tokens_per_min:,.0f} tokens/min",
            f"Total cost: ${self.total_cost:.2f}",
            "",
            f"Dashboard: {self.dashboard_path}",
            f"Summary JSON: {self.summary_path}",
            "=" * 80,
            ""
        ]

        return "\n".join(report)

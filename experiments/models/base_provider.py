import pandas as pd
import numpy as np
import time
import os
import threading
from typing import Any, Optional
import concurrent.futures
from abc import ABC, abstractmethod
from datetime import datetime
from .litellm_tracker import LiteLLMTokenTracker

class BaseProvider(ABC):
    def __init__(self):
        self.client = self.get_client()
        self.token_tracker: Optional[LiteLLMTokenTracker] = None
        self._csv_lock = threading.Lock()  # Lock for safe concurrent CSV writes

    def get_context_safety_margin(self) -> float:
        """Return safety margin for context length filtering.

        Default is 1.0 (no margin). Providers with tokenizer mismatch
        (e.g., GPT-OSS) should override to return lower values.
        """
        return 1.0

    @abstractmethod
    def process_single_prompt(self, prompt: str, model_name: str, max_output_tokens: int, index: int) -> tuple[int, str]:
        pass

    @abstractmethod
    def get_client(self) -> Any:
        pass

    def _get_default_test_samples(self) -> int:
        """Default test mode sample size"""
        return 20

    def create_batches(self, df: pd.DataFrame, max_tokens_per_minute: int) -> list[list[int]]:
        indices = df.index.tolist()
        token_counts = df['token_count'].tolist()
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx, tokens in zip(indices, token_counts):
            if tokens > max_tokens_per_minute:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                
                batches.append([idx])
                continue
            
            if current_tokens + tokens > max_tokens_per_minute and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(idx)
            current_tokens += tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def process_batch(self, input_df: pd.DataFrame, output_df: pd.DataFrame, indices_to_process: list[int], model_name: str, output_path: str, input_column: str, output_column: str, save_every: int = 10) -> int:
        """Process a batch of prompts. Returns count of successful completions."""
        timeout_per_request = 500
        completed_since_save = 0
        success_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices_to_process)) as executor:
            futures = {
                executor.submit(
                    self.process_single_prompt,
                    prompt=str(input_df.loc[idx, input_column]),
                    model_name=model_name,
                    max_output_tokens=int(input_df.loc[idx, 'max_output_tokens']) if 'max_output_tokens' in input_df.columns else 1000,
                    index=int(idx),
                ): idx
                for idx in indices_to_process
            }

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    idx_result, response = future.result(timeout=timeout_per_request)
                    output_df.loc[idx_result, output_column] = response

                    success = not response.startswith('ERROR')
                    status = "Success" if success else "Error"
                    print(f"{status} - Row {idx_result}: {response[:80]}...")

                    if success:
                        success_count += 1
                    else:
                        # Increment error count on failure for retry tracking
                        output_df.loc[idx_result, '_error_count'] = output_df.loc[idx_result, '_error_count'] + 1

                except concurrent.futures.TimeoutError:
                    print(f"Row {idx}: Request timed out after {timeout_per_request}s - marking as timeout error")
                    output_df.loc[idx, output_column] = f"ERROR_TIMEOUT: Request exceeded {timeout_per_request}s"
                    output_df.loc[idx, '_error_count'] = output_df.loc[idx, '_error_count'] + 1

                except Exception as e:
                    output_df.loc[idx, output_column] = f"FUTURE_ERROR: {str(e)}"
                    output_df.loc[idx, '_error_count'] = output_df.loc[idx, '_error_count'] + 1
                    print(f"Error - Row {idx}: Future error: {e}")

                # Incremental checkpoint save
                completed_since_save += 1
                if completed_since_save >= save_every:
                    with self._csv_lock:
                        output_df.to_csv(output_path, index=False)
                    print(f"  [checkpoint] Saved after {completed_since_save} rows")
                    completed_since_save = 0

        # Final save for remaining rows in batch
        with self._csv_lock:
            output_df.to_csv(output_path, index=False)

        completed_in_batch = len([idx for idx in indices_to_process if not pd.isna(output_df.loc[idx, output_column])])
        total_completed = (~output_df[output_column].isna() &
                          ~output_df[output_column].str.startswith('ERROR', na=False)).sum()

        print(f"Batch complete: {completed_in_batch}/{len(indices_to_process)} rows")
        print(f"Overall progress: {total_completed}/{len(output_df)} ({total_completed/len(output_df)*100:.1f}%)")

        return success_count

    def main(self, input_path: str, output_path: str, input_column: str, output_column: str, model_name: str, max_context_length: int, max_tokens_per_minute: int, test_mode: bool = False, max_samples: int = None, save_every: int = 10, max_retries: int = 2, truncate_to_fit: bool = False) -> None:
        # Initialize token tracker and register with LiteLLM's callback system
        experiment_name = os.path.splitext(os.path.basename(output_path))[0]
        output_dir = os.path.dirname(output_path) or "results"
        self.token_tracker = LiteLLMTokenTracker(output_dir=output_dir, experiment_name=experiment_name)
        self.token_tracker.register()  # Enable automatic tracking via LiteLLM callbacks
        print(f"Token tracking enabled - Dashboard: {self.token_tracker.dashboard_path}")

        input_df = pd.read_csv(input_path)
        original_size = len(input_df)

        # Apply sampling if test mode or max_samples specified
        if test_mode or max_samples:
            n_samples = max_samples if max_samples else self._get_default_test_samples()

            if len(input_df) > n_samples:
                # Stratified sampling - evenly distributed across dataset
                indices = np.linspace(0, len(input_df)-1, n_samples, dtype=int)
                input_df = input_df.iloc[indices].copy()
                input_df = input_df.reset_index(drop=True)
                print(f"{'Test mode' if test_mode else 'Sampling'}: Selected {len(input_df)} from {original_size} rows")

        # Apply safety margin for tokenizer variance (e.g., GPT-OSS uses 0.85)
        safety_margin = self.get_context_safety_margin()
        effective_context_length = int(max_context_length * safety_margin)

        # Optional: truncate oversized prompts from front (preserves question at end)
        if truncate_to_fit:
            from .truncation import truncate_from_front

            exceeds_limit = input_df['token_count'] > effective_context_length
            exceeds_count = exceeds_limit.sum()

            if exceeds_count > 0:
                # Initialize truncation metadata columns
                input_df['_truncated'] = False
                input_df['_original_tokens'] = input_df['token_count']
                input_df['_tokens_removed'] = 0

                truncated_count = 0
                for idx in input_df[exceeds_limit].index:
                    prompt = str(input_df.loc[idx, input_column])
                    truncated_prompt, metadata = truncate_from_front(prompt, effective_context_length)

                    if metadata['truncated']:
                        input_df.loc[idx, input_column] = truncated_prompt
                        input_df.loc[idx, 'token_count'] = metadata['final_tokens']
                        input_df.loc[idx, '_truncated'] = True
                        input_df.loc[idx, '_tokens_removed'] = metadata['tokens_removed']
                        truncated_count += 1

                print(f"Truncation: {truncated_count}/{exceeds_count} oversized prompts truncated to fit {effective_context_length:,} tokens")

        input_df_filtered = input_df[input_df['token_count'] <= effective_context_length].copy()

        margin_note = f" (safety margin: {safety_margin:.0%})" if safety_margin < 1.0 else ""
        print(f"Filtered by context length ({effective_context_length:,} tokens{margin_note}): {len(input_df)} to {len(input_df_filtered)} rows ({len(input_df) - len(input_df_filtered)} filtered out)")

        # Early exit if all rows filtered out
        if len(input_df_filtered) == 0:
            print("WARNING: All rows exceed effective context length - nothing to process")
            if safety_margin < 1.0:
                print(f"  Consider using a model with larger context, or reducing input size")
                print(f"  Min token count in data: {input_df['token_count'].min():,}")
                print(f"  Effective limit: {effective_context_length:,}")
            return

        # Auto-merge: Check for test results when running production mode
        if not test_mode and not os.path.exists(output_path):
            basename = os.path.basename(output_path)
            dirname = os.path.dirname(output_path)
            test_path = os.path.join(dirname, f"test_{basename}") if dirname else f"test_{basename}"

            if os.path.exists(test_path):
                print(f"Found test results at {test_path}")
                test_df = pd.read_csv(test_path)

                # Count completed test rows
                if output_column in test_df.columns:
                    test_completed = (~test_df[output_column].isna() &
                                     ~test_df[output_column].astype(str).str.startswith('ERROR')).sum()
                    print(f"  Test results contain {test_completed} completed rows")

                    if test_completed > 0:
                        # Create output df from filtered input, merge test results
                        output_df = input_df_filtered.drop(columns=[input_column]).copy()
                        output_df[output_column] = None

                        # Match by token_count + needle_depth (NIAH) or question (LongMemEval)
                        merge_keys = []
                        for key in ['token_count', 'needle_depth', 'question']:
                            if key in output_df.columns and key in test_df.columns:
                                merge_keys.append(key)

                        if merge_keys:
                            # Create lookup from test results
                            test_lookup = test_df.set_index(merge_keys)[output_column].to_dict()

                            merged_count = 0
                            for idx, row in output_df.iterrows():
                                key = tuple(row[k] for k in merge_keys)
                                if key in test_lookup:
                                    val = test_lookup[key]
                                    if pd.notna(val) and not str(val).startswith('ERROR'):
                                        output_df.loc[idx, output_column] = val
                                        merged_count += 1

                            if merged_count > 0:
                                print(f"  Merged {merged_count} test results into production run")
                                output_df.to_csv(output_path, index=False)

        if os.path.exists(output_path):
            # Check for mode mismatch
            existing_df = pd.read_csv(output_path)
            is_existing_test = '_test_mode' in existing_df.columns and existing_df['_test_mode'].iloc[0] if len(existing_df) > 0 else False

            if is_existing_test != test_mode:
                print(f"WARNING: Output file mode mismatch!")
                print(f"  Existing file: {'test' if is_existing_test else 'production'} mode")
                print(f"  Current run:   {'test' if test_mode else 'production'} mode")
                backup_path = output_path + '.backup'
                os.rename(output_path, backup_path)
                print(f"  Renamed existing file to: {backup_path}")
                print(f"  Starting fresh with current mode")

                output_df = input_df_filtered.drop(columns=[input_column]).copy()
                output_df[output_column] = None
            else:
                print(f"Loading existing progress from {output_path}")
                output_df = existing_df

                if output_column not in output_df.columns:
                    output_df[output_column] = None
                if '_error_count' not in output_df.columns:
                    output_df['_error_count'] = 0
        else:
            output_df = input_df_filtered.drop(columns=[input_column]).copy()
            output_df[output_column] = None

        # Initialize error count column for retry tracking
        if '_error_count' not in output_df.columns:
            output_df['_error_count'] = 0

        # Mark test mode in output
        if test_mode:
            output_df['_test_mode'] = True
            output_df['_test_timestamp'] = datetime.now().isoformat()
            output_df['_original_dataset_size'] = original_size

        # Rows need processing if: (empty OR has error) AND under max retries
        has_error_or_empty = (
            output_df[output_column].isna() |
            output_df[output_column].str.contains('ERROR', na=False)
        )
        under_retry_limit = output_df['_error_count'] < max_retries
        need_processing = has_error_or_empty & under_retry_limit

        # Log skipped rows that exceeded max retries
        exceeded_retries = has_error_or_empty & ~under_retry_limit
        if exceeded_retries.sum() > 0:
            print(f"Skipping {exceeded_retries.sum()} rows that exceeded {max_retries} retries")

        to_process = output_df[need_processing].index.tolist()
        
        if to_process:
            print(f"{len(to_process)} rows needing processing: {to_process[0]} to {to_process[-1]}")
        else:
            print("All rows already processed successfully")
            return
            
        input_to_process = input_df_filtered.loc[to_process]
        batches = self.create_batches(input_to_process, max_tokens_per_minute)
        print(f"Created {len(batches)} batches based on {max_tokens_per_minute:,} tokens/minute")
        
        for i, batch_indices in enumerate(batches):
            batch_successes = self.process_batch(input_to_process, output_df, batch_indices, model_name, output_path, input_column, output_column, save_every=save_every)
            if i < len(batches) - 1:
                if batch_successes > 0:
                    print("Waiting 60 seconds")
                    time.sleep(60)
                else:
                    print("Skipping wait (batch had no successes)")  # Rate limit only needed on success

        # Save final token tracking summary
        if self.token_tracker:
            summary = self.token_tracker.save_final()
            print(summary)
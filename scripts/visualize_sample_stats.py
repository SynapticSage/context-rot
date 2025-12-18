#!/usr/bin/env python3
"""
Visualize sample statistics across all experiments.
Shows successful vs failed samples per experiment type.
Created: 2025-12-07
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def bootstrap_ci(successes: int, total: int, n_bootstrap: int = 200, ci: float = 0.95) -> tuple[float, float]:
    """
    Calculate bootstrap confidence interval for success rate.

    Args:
        successes: Number of successful samples
        total: Total number of samples
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) tuple as percentages (0-100)
    """
    if total == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(42)  # Reproducible results

    # Create binary array (1=success, 0=failure)
    data = np.array([1] * successes + [0] * (total - successes))

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample) * 100)  # Convert to percentage

    lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)

    return lower, upper


def count_samples(csv_path: str) -> dict:
    """Count total, successful, and error samples in a results CSV."""
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)

    # Find the output column (varies by experiment)
    output_col = None
    for col in ['output', 'llm_judge_output']:
        if col in df.columns:
            output_col = col
            break

    if output_col is None:
        return None

    total = len(df)

    # Count errors (output starts with ERROR_ or is NaN)
    errors = df[output_col].isna() | df[output_col].astype(str).str.startswith('ERROR')
    error_count = errors.sum()
    success_count = total - error_count

    # Categorize error types if present
    error_types = {}
    if error_count > 0:
        error_outputs = df.loc[errors, output_col].astype(str)
        for err in error_outputs:
            if 'context length' in err.lower() or 'maximum context' in err.lower():
                error_types['Context too long'] = error_types.get('Context too long', 0) + 1
            elif 'timeout' in err.lower():
                error_types['Timeout'] = error_types.get('Timeout', 0) + 1
            elif 'connection' in err.lower():
                error_types['Connection error'] = error_types.get('Connection error', 0) + 1
            elif pd.isna(err) or err == 'nan':
                error_types['Not processed'] = error_types.get('Not processed', 0) + 1
            else:
                error_types['Other'] = error_types.get('Other', 0) + 1

    return {
        'total': total,
        'success': success_count,
        'errors': error_count,
        'error_types': error_types
    }


def gather_model_stats(results_dir: str, model_prefix: str) -> dict:
    """Gather statistics for all experiments of a model."""
    stats = {}

    # Define experiment patterns
    experiments = {
        'NIAH': f'{model_prefix}_niah_results.csv',
        'LongMemEval\n(focused)': f'{model_prefix}_longmemeval_focused_results.csv',
        'LongMemEval\n(full)': f'{model_prefix}_longmemeval_full_results.csv',
        'Repeated\nWords': f'{model_prefix}_repeated_words_apple_apples.csv',
    }

    for exp_name, filename in experiments.items():
        filepath = os.path.join(results_dir, filename)
        result = count_samples(filepath)
        if result:
            stats[exp_name] = result

    return stats


def plot_sample_counts(stats: dict, model_name: str, output_path: str):
    """Create bar chart showing success vs error counts per experiment."""
    experiments = list(stats.keys())
    successes = [stats[exp]['success'] for exp in experiments]
    errors = [stats[exp]['errors'] for exp in experiments]
    totals = [stats[exp]['total'] for exp in experiments]

    x = np.arange(len(experiments))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw success bars (green)
    bars_success = ax.bar(x, successes, width, label='Successful', color='#4CAF50',
                          edgecolor='none', linewidth=0)

    # Draw error bars (red) ONLY for experiments with errors to avoid zero-height bar artifacts
    error_indices = [i for i, e in enumerate(errors) if e > 0]
    if error_indices:
        error_x = [x[i] for i in error_indices]
        error_heights = [errors[i] for i in error_indices]
        error_bottoms = [successes[i] for i in error_indices]
        bars_error = ax.bar(error_x, error_heights, width, bottom=error_bottoms,
                           label='Failed', color='#F44336', edgecolor='none', linewidth=0)
    else:
        # Add dummy for legend
        bars_error = ax.bar([], [], width, label='Failed', color='#F44336')

    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Sample Statistics by Experiment - {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()

    # Add count labels on bars
    for i, (s, e, t) in enumerate(zip(successes, errors, totals)):
        # Success count (if visible)
        if s > 0:
            ax.text(i, s/2, str(s), ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        # Error count (if visible)
        if e > 0:
            ax.text(i, s + e/2, str(e), ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        # Total on top
        ax.text(i, t + 2, f'n={t}', ha='center', va='bottom', fontsize=9, style='italic')

    # Add percentage annotations
    for i, (s, t) in enumerate(zip(successes, totals)):
        if t > 0:
            pct = s / t * 100
            ax.text(i, -8, f'{pct:.0f}% success', ha='center', va='top', fontsize=8, color='#666')

    ax.set_ylim(bottom=-15, top=max(totals) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample counts plot saved to: {output_path}")


def plot_error_breakdown(stats: dict, model_name: str, output_path: str):
    """Create pie chart showing error type breakdown across all experiments."""
    # Aggregate error types across all experiments
    all_error_types = {}
    total_errors = 0

    for exp_stats in stats.values():
        for err_type, count in exp_stats.get('error_types', {}).items():
            all_error_types[err_type] = all_error_types.get(err_type, 0) + count
            total_errors += count

    if total_errors == 0:
        print("No errors to visualize in error breakdown")
        return

    # Sort by count
    sorted_types = sorted(all_error_types.items(), key=lambda x: x[1], reverse=True)
    labels = [t[0] for t in sorted_types]
    sizes = [t[1] for t in sorted_types]

    # Colors for error types
    colors = ['#FF6B6B', '#FFE66D', '#4ECDC4', '#95E1D3', '#F38181']

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.0f%%',
        colors=colors[:len(labels)], startangle=90
    )
    ax.set_title(f'Error Type Breakdown - {model_name}\n(Total: {total_errors} errors)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Error breakdown plot saved to: {output_path}")


def plot_completion_summary(stats_20b: dict, stats_120b: dict, output_path: str):
    """Create side-by-side comparison of both models with bootstrap 95% CIs."""
    experiments = list(stats_20b.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, stats, model_name in [
        (axes[0], stats_20b, 'GPT-OSS 20B'),
        (axes[1], stats_120b, 'GPT-OSS 120B')
    ]:
        if not stats:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_name)
            continue

        exp_names = list(stats.keys())
        success_rates = [stats[exp]['success'] / stats[exp]['total'] * 100
                        if stats[exp]['total'] > 0 else 0
                        for exp in exp_names]

        # Calculate bootstrap CIs
        cis = [bootstrap_ci(stats[exp]['success'], stats[exp]['total'])
               for exp in exp_names]

        # Error bar sizes (distance from mean to bounds)
        xerr_lower = [rate - ci[0] for rate, ci in zip(success_rates, cis)]
        xerr_upper = [ci[1] - rate for rate, ci in zip(success_rates, cis)]

        colors = ['#4CAF50' if r >= 80 else '#FFC107' if r >= 50 else '#F44336' for r in success_rates]

        y_pos = np.arange(len(exp_names))

        # Draw bars without error bars first
        bars = ax.barh(y_pos, success_rates, color=colors)

        # Add error bars only where CI width > 0.5%
        for i, (rate, ci) in enumerate(zip(success_rates, cis)):
            ci_width = ci[1] - ci[0]
            if ci_width > 0.5:  # Only show error bars for non-trivial intervals
                ax.errorbar(rate, i, xerr=[[rate - ci[0]], [ci[1] - rate]],
                           fmt='none', color='black', capsize=4, linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(exp_names)
        ax.set_xlim(0, 110)
        ax.set_xlabel('Success Rate (%)')
        ax.set_title(model_name)

        # Add percentage labels with CI
        for i, (rate, ci, exp) in enumerate(zip(success_rates, cis, exp_names)):
            total = stats[exp]['total']
            ax.text(min(ci[1] + 3, 108), i,
                   f'{rate:.0f}% [{ci[0]:.0f}-{ci[1]:.0f}]\nn={total}',
                   va='center', fontsize=8)

    plt.suptitle('Experiment Completion Rates by Model\n(95% CI from 200 bootstrap samples)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Completion summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize sample statistics across experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate stats for both models
  python scripts/visualize_sample_stats.py --results-dir ./results

  # Generate stats for specific model
  python scripts/visualize_sample_stats.py --results-dir ./results --model gpt_oss_20b
        """
    )

    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory containing results CSV files')
    parser.add_argument('--model', type=str, default=None,
                       help='Model prefix (e.g., gpt_oss_20b). If not specified, runs for both.')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as results-dir)')

    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir

    models = [args.model] if args.model else ['gpt_oss_20b', 'gpt_oss_120b']

    all_stats = {}
    for model in models:
        print(f"\nGathering stats for {model}...")
        stats = gather_model_stats(args.results_dir, model)

        if not stats:
            print(f"  No results found for {model}")
            continue

        all_stats[model] = stats

        # Print summary
        print(f"  Found {len(stats)} experiments:")
        for exp, s in stats.items():
            exp_clean = exp.replace('\n', ' ')
            print(f"    {exp_clean}: {s['success']}/{s['total']} successful ({s['success']/s['total']*100:.1f}%)")

        # Generate plots for this model
        model_display = model.replace('_', '-').upper()

        plot_sample_counts(
            stats, model_display,
            os.path.join(output_dir, f'{model}_sample_counts.png')
        )

        plot_error_breakdown(
            stats, model_display,
            os.path.join(output_dir, f'{model}_error_breakdown.png')
        )

    # Generate comparison plot if both models have data
    if 'gpt_oss_20b' in all_stats and 'gpt_oss_120b' in all_stats:
        plot_completion_summary(
            all_stats['gpt_oss_20b'],
            all_stats['gpt_oss_120b'],
            os.path.join(output_dir, 'model_comparison_completion.png')
        )


if __name__ == '__main__':
    main()

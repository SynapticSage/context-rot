import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 200, ci: float = 0.95) -> tuple[float, float]:
    """
    Calculate bootstrap confidence interval for the mean.

    Args:
        data: Array of values (typically 0/1 for accuracy)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    rng = np.random.default_rng(42)  # Reproducible results
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)

    return lower, upper


def visualize_longmemeval_results(focused_filepath: str, full_filepath: str, model_name: str, output_path: str):
    focused_df = pd.read_csv(focused_filepath)
    full_df = pd.read_csv(full_filepath)

    focused_scores = focused_df['llm_judge_output'].values
    full_scores = full_df['llm_judge_output'].values

    focused_mean = np.mean(focused_scores)
    full_mean = np.mean(full_scores)

    # Calculate 95% bootstrap CIs
    focused_ci = bootstrap_ci(focused_scores)
    full_ci = bootstrap_ci(full_scores)

    focused_color = "#EB4026"
    full_color = "#3A76E5"

    plt.figure(figsize=(8, 6))

    # Labels with explanatory parentheticals
    labels = [
        'Focused\n(relevant context only,\n~4k tokens)',
        'Full\n(entire conversation,\n~113k tokens)'
    ]
    x_pos = np.arange(len(labels))

    # Calculate error bar sizes (distance from mean to CI bounds)
    # yerr needs shape (2, n) where row 0 is lower errors, row 1 is upper errors
    yerr = [
        [focused_mean - focused_ci[0], full_mean - full_ci[0]],  # lower errors
        [focused_ci[1] - focused_mean, full_ci[1] - full_mean]   # upper errors
    ]

    bars = plt.bar(x_pos, [focused_mean, full_mean], color=[focused_color, full_color],
                   yerr=yerr, capsize=5, error_kw={'linewidth': 2})

    plt.xticks(x_pos, labels)
    plt.ylim(0, 1)
    plt.ylabel('Average Score')

    # 3-line title: main title + description + scientific question
    plt.suptitle(f'LongMemEval: Retrieval vs Reasoning - {model_name}',
                 fontsize=12, fontweight='bold', y=0.98)
    subtitle = ('Accuracy on multi-turn QA with varying context lengths\n'
                'Does extra context help or hurt retrieval-based reasoning?')
    plt.figtext(0.5, 0.91, subtitle, ha='center', fontsize=9, style='italic')

    # Add value labels with CI
    means = [focused_mean, full_mean]
    cis = [focused_ci, full_ci]
    for i, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
        plt.text(bar.get_x() + bar.get_width()/2, ci[1] + 0.03,
                f"{mean:.2f}\n[{ci[0]:.2f}, {ci[1]:.2f}]",
                ha='center', va='bottom', fontsize=9)

    # Add sample counts as subtitle
    focused_n = len(focused_df)
    full_n = len(full_df)
    plt.figtext(0.5, 0.01,
                f"n={focused_n} focused, n={full_n} full | 95% CI from 200 bootstrap samples",
                ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.82)  # Room for footer + 3-line header
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize LongMemEval results')

    parser.add_argument('--focused-path', type=str, required=True,
                       help='Path to focused results CSV file')
    parser.add_argument('--full-path', type=str, required=True,
                       help='Path to full results CSV file')
    parser.add_argument('--model', '--model-name', type=str, required=True,
                       dest='model', help='Model name for plot titles')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Output path for PNG file')

    args = parser.parse_args()

    try:
        visualize_longmemeval_results(
            focused_filepath=args.focused_path,
            full_filepath=args.full_path,
            model_name=args.model,
            output_path=args.output_path
        )
        print(f"Visualization saved to: {args.output_path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

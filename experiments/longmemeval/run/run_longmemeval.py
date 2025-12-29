# LongMemEval Experiment Runner
# Created: 2025-11-19
# Modified: 2025-12-18 (use provider registry)

import argparse
import sys
import os
import dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.registry import get_provider, all_names

dotenv.load_dotenv()


def main():
    parser = argparse.ArgumentParser(description='Run LongMemEval experiments')

    parser.add_argument('--provider', type=str, required=True,
                       help=f'Provider to use. Available: {", ".join(all_names())}')
    parser.add_argument('--input-path', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to output CSV file')
    parser.add_argument('--input-column', type=str, required=True,
                       help='Column name containing input prompts')
    parser.add_argument('--output-column', type=str, required=True,
                       help='Column name for output results')
    parser.add_argument('--model', '--model-name', type=str, required=True,
                       dest='model', help='Model to run')
    parser.add_argument('--max-context-length', type=int, required=True,
                       help='Maximum context length in tokens')
    parser.add_argument('--max-tokens-per-minute', type=int, required=True,
                       help='Maximum tokens per minute for rate limits')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with reduced samples')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--shuffle-samples', action='store_true',
                       help='Randomly sample instead of evenly-spaced selection')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for shuffle sampling (default: 42)')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N rows (default: 10)')
    parser.add_argument('--truncate-to-fit', action='store_true',
                       help='Truncate oversized prompts from front to fit context limit')

    args = parser.parse_args()

    # Auto-adjust output path for test mode
    if args.test_mode and not os.path.basename(args.output_path).startswith('test_'):
        args.output_path = os.path.join(
            os.path.dirname(args.output_path),
            f"test_{os.path.basename(args.output_path)}"
        )
        print(f"Test mode: Output redirected to {args.output_path}")

    try:
        # Use registry to get provider - supports all registered names and aliases
        provider = get_provider(args.provider)

        provider.main(
            input_path=args.input_path,
            output_path=args.output_path,
            input_column=args.input_column,
            output_column=args.output_column,
            model_name=args.model,
            max_context_length=args.max_context_length,
            max_tokens_per_minute=args.max_tokens_per_minute,
            test_mode=args.test_mode,
            max_samples=args.max_samples,
            save_every=args.save_every,
            truncate_to_fit=args.truncate_to_fit,
            shuffle_samples=args.shuffle_samples,
            seed=args.seed
        )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

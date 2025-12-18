import argparse
import sys
import os
import dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.providers.openai import OpenAIProvider
from models.providers.anthropic import AnthropicProvider
from models.providers.google import GoogleProvider
from models.providers.gptoss import GptOssProvider

dotenv.load_dotenv()

def get_provider(provider_name: str, model_name: str = None):
    if provider_name.lower() == 'openai':
        return OpenAIProvider()
    elif provider_name.lower() == 'anthropic':
        return AnthropicProvider()
    elif provider_name.lower() == 'google':
        return GoogleProvider(model_name)
    elif provider_name.lower() == 'gptoss':
        return GptOssProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Available providers: openai, anthropic, google, gptoss")


def main():
    parser = argparse.ArgumentParser(description='Run providers')
    
    parser.add_argument('--provider', type=str, required=True,
                       choices=['openai', 'anthropic', 'google', 'gptoss'],
                       help='Provider to use')
    parser.add_argument('--input-path', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to output CSV file')
    parser.add_argument('--input-column', type=str, required=True,
                       help='Column name containing input prompts')
    parser.add_argument('--output-column', type=str, required=True,
                       help='Column name for output results')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Model to run')
    parser.add_argument('--max-context-length', type=int, required=True,
                       help='Maximum context length in tokens')
    parser.add_argument('--max-tokens-per-minute', type=int, required=True,
                       help='Maximum tokens per minute for rate limits')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with reduced samples')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process')
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
        provider = get_provider(args.provider, args.model_name)

        provider.main(
            input_path=args.input_path,
            output_path=args.output_path,
            input_column=args.input_column,
            output_column=args.output_column,
            model_name=args.model_name,
            max_context_length=args.max_context_length,
            max_tokens_per_minute=args.max_tokens_per_minute,
            test_mode=args.test_mode,
            max_samples=args.max_samples,
            save_every=args.save_every,
            truncate_to_fit=args.truncate_to_fit
        )
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
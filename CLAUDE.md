# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Context Rot research toolkit for investigating how LLM performance degrades with increasing input token counts. The repository implements three experimental paradigms to measure model performance across varying context lengths.

### Research Goal

**Primary Objective**: Replicate the Context Rot paper results specifically for OpenAI's GPT-OSS models:

| Model | OpenRouter ID | Parameters | Active Params | Context |
|-------|---------------|------------|---------------|---------|
| gpt-oss-20b | `openai/gpt-oss-20b` | 21B MoE | 3.6B | 128k |
| gpt-oss-120b | `openai/gpt-oss-120b` | 117B MoE | 5.1B | 128k |

These are OpenAI's open-weight models released under Apache 2.0. The research investigates whether these models exhibit the same context rot degradation patterns observed in closed models.

**Do NOT substitute other models** (e.g., Qwen, Llama) - the entire purpose of this fork is to generate GPT-OSS specific results.

## Architecture

### Provider Pattern
All model interactions use the **BaseProvider** abstract class (`experiments/models/base_provider.py`) which defines:
- `process_single_prompt()`: Single inference call
- `get_client()`: API client initialization
- `create_batches()`: Token-based batching for rate limiting
- `process_batch()`: Concurrent execution with ThreadPoolExecutor
- `main()`: Full workflow with checkpoint/resume support

Concrete implementations in `experiments/models/providers/`:
- `openai.py`: OpenAI API (temperature=0, max_completion_tokens)
- `anthropic.py`: Anthropic API
- `google.py`: Google Vertex AI

**Key Design**: All run scripts follow identical CLI patterns and delegate to provider.main() for execution.

### Experiment Structure
Each experiment (`experiments/{name}/`) contains:
```
run/           - Data generation and inference scripts
evaluate/      - LLM judge evaluation and analysis
README.md      - Detailed parameter documentation
```

## Environment Setup

### Required API Keys
```bash
export OPENAI_API_KEY="..."           # OpenAI models + LLM judge
export ANTHROPIC_API_KEY="..."        # Anthropic models
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
export GOOGLE_MODEL_PATH="..."        # Google models

# GPT-OSS / Qwen Models (choose one deployment option)
export GPT_OSS_BASE_URL="http://localhost:8000/v1"  # Local (vLLM/ollama/TGI)
export GPT_OSS_API_KEY="dummy"                      # Optional for local

# OR use OpenRouter
export OPENROUTER_API_KEY="sk-or-..."
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

### Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
chmod 600 .env
# Edit .env with your API keys
```

## Test Mode

Validate your setup with reduced samples before running full experiments:

```bash
# Complete test workflow (10-20 minutes, < $2 cost)
./scripts/run_full_research.sh -t

# Individual experiment test
cd experiments/niah_extension
python run/run_niah_extension.py \
  --provider gptoss \
  --model-name qwen2.5:20b \
  --input-path ../../data/niah_prompts/niah_prompts_sequential.csv \
  --output-path ../../results/test_niah_results.csv \
  --input-column prompt \
  --output-column output \
  --max-context-length 131_072 \
  --max-tokens-per-minute 50_000 \
  --test-mode
```

**Test mode features:**
- Reduces samples: NIAH (60 vs 440), LongMemEval (40 vs 612), Repeated Words (15 vs ~300)
- Results prefixed with `test_` to avoid confusion
- Validates full pipeline end-to-end
- Cost: < $2, Time: 10-20 minutes
- Automatically adjusts output paths

**Use test mode for:**
- First-time setup validation
- Checking API connections and credentials
- Testing code changes
- Quick sanity checks before long runs

## Token Tracking

All experiments automatically track token usage in real-time using LiteLLM integration:

**Live Monitoring:**
```bash
# During execution, monitor live token usage
tail -f results/gpt_oss_20b_niah_results_token_dashboard.txt

# Dashboard updates every 10 API calls showing:
# - Total input/output tokens
# - Current rate (tokens/min)
# - Estimated cost
# - Per-model breakdown
```

**Output Files:**
- `results/{experiment}_token_dashboard.txt` - Live dashboard (updates during run)
- `results/{experiment}_token_summary.json` - Final summary with complete metrics

**What's Tracked:**
- API call count
- Input and output token counts
- Token rate (tokens/minute)
- Estimated costs (using LiteLLM's cost tables)
- Per-model breakdowns

**Implementation:**
Token tracking is built into `BaseProvider.main()` and automatically works for all providers (OpenAI, Anthropic, Google, GPT-OSS). Uses LiteLLM's unified API to track tokens across all model types consistently.

## Common Commands

### 1. NIAH Extension (Needle in a Haystack)

**Generate haystacks with semantic needles:**
```bash
cd experiments/niah_extension
python run/create_haystacks.py \
  --haystack-folder ../../data/PaulGrahamEssays \
  --needle "It sometimes surprises people when I tell them I write every week..." \
  --question "What was the best writing advice I got from my college classmate?" \
  --output-folder ../../data/niah_prompts \
  [--shuffled] \
  [--distractors "distractor1" "distractor2"]
```

**Run inference:**
```bash
python run/run_niah_extension.py \
  --provider openai \
  --input-path ../../data/niah_prompts/niah_prompts_sequential.csv \
  --output-path ../../results/model_niah_results.csv \
  --input-column prompt \
  --output-column output \
  --model-name gpt-4.1-2025-04-14 \
  --max-context-length 1_047_576 \
  --max-tokens-per-minute 2_000_000
```

**Evaluate with LLM judge:**
```bash
python evaluate/evaluate_niah_extension.py \
  --input-path ../../results/model_niah_results.csv \
  --output-path ../../results/model_niah_evaluated.csv \
  --model-name gpt-4.1-2025-04-14 \
  --output-column output \
  --question-column question \
  --correct-answer-column answer
```

**Visualize heatmap:**
```bash
python evaluate/visualize.py \
  --csv-path ../../results/model_niah_evaluated.csv \
  --output-path ../../results/model_heatmap.png \
  --title "NIAH Performance - Model Name"
```

### 2. LongMemEval (Conversational QA)

Tests retrieval+reasoning vs reasoning-only by comparing focused (relevant context only) vs full (113k tokens) inputs.

**Run inference (focused):**
```bash
cd experiments/longmemeval
python run/run_longmemeval.py \
  --provider openai \
  --input-path ../../data/cleaned_longmemeval_s_focused.csv \
  --output-path ../../results/model_longmemeval_focused_results.csv \
  --input-column focused_prompt \
  --output-column output \
  --model-name gpt-4.1-2025-04-14 \
  --max-context-length 1_047_576 \
  --max-tokens-per-minute 2_000_000
```

**Run inference (full):**
```bash
python run/run_longmemeval.py \
  --provider openai \
  --input-path ../../data/cleaned_longmemeval_s_full.csv \
  --output-path ../../results/model_longmemeval_full_results.csv \
  --input-column full_prompt \
  --output-column output \
  --model-name gpt-4.1-2025-04-14 \
  --max-context-length 1_047_576 \
  --max-tokens-per-minute 2_000_000
```

**Evaluate both:**
```bash
python evaluate/evaluate_longmemeval.py \
  --input-path ../../results/model_longmemeval_focused_results.csv \
  --output-path ../../results/model_longmemeval_focused_evaluated.csv \
  --model-name gpt-4.1-2025-04-14 \
  --output-column output \
  --question-column question \
  --correct-answer-column answer

python evaluate/evaluate_longmemeval.py \
  --input-path ../../results/model_longmemeval_full_results.csv \
  --output-path ../../results/model_longmemeval_full_evaluated.csv \
  --model-name gpt-4.1-2025-04-14 \
  --output-column output \
  --question-column question \
  --correct-answer-column answer
```

**Visualize comparison:**
```bash
python evaluate/visualize.py \
  --focused-path ../../results/model_longmemeval_focused_evaluated.csv \
  --full-path ../../results/model_longmemeval_full_evaluated.csv \
  --model-name "Model Name" \
  --output-path ../../results/model_longmemeval.png
```

### 3. Repeated Words

Tests exact replication of repeated word sequences with single modified word insertion.

**Run inference (generates data internally):**
```bash
cd experiments/repeated_words
python run/run_repeated_words.py \
  --provider openai \
  --model-name gpt-4.1-2025-04-14 \
  --output-path ../../results/model_repeated_words_apple_apples.csv \
  --common-word apple \
  --modified-word apples \
  --model-max-output-tokens 32_768 \
  --max-context-length 1_047_576 \
  --max-tokens-per-minute 2_000_000
```
**Note**: Takes significant time due to high output token requirements.

**Evaluate and visualize:**
```bash
python evaluate/evaluate_repeated_words.py \
  --input-path ../../results/model_repeated_words_apple_apples.csv \
  --output-dir ../../results/model_repeated_words_apple_apples_evaluated.csv \
  --common-word apple \
  --modified-word apples \
  --model-name "Model Name"
```

Generates: token_count_performance.png, levenshtein_score.png, modified_word_present.png, position_accuracy.png, word_count_delta.png

### 4. GPT-OSS Models (Target Models)

The `gptoss` provider supports OpenAI's GPT-OSS models.

**Target Models (DO NOT SUBSTITUTE):**

| Model | Model ID | Parameters | Active | Context |
|-------|----------|------------|--------|---------|
| gpt-oss-20b | `openai/gpt-oss-20b` | 21B MoE | 3.6B | 128k |
| gpt-oss-120b | `openai/gpt-oss-120b` | 117B MoE | 5.1B | 128k |

**API Priority (auto-detected):**
1. **OpenAI API** (recommended) - `OPENAI_API_KEY`
2. **OpenRouter** (fallback) - `OPENROUTER_API_KEY`
3. **Local** (dev only) - `GPT_OSS_BASE_URL`

**Setup in .env:**
```bash
# Option 1: OpenAI API (recommended - fastest, best rate limits)
OPENAI_API_KEY=sk-your-openai-key-here

# Option 2: OpenRouter (fallback)
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Option 3: Local (for development/testing only)
# GPT_OSS_BASE_URL=http://localhost:11434/v1
```

The provider auto-detects which API to use based on available environment variables.

**Example: Run NIAH with gpt-oss-20b:**
```bash
cd experiments/niah_extension
python run/run_niah_extension.py \
  --provider gptoss \
  --input-path ../../data/niah_prompts/niah_prompts_sequential.csv \
  --output-path ../../results/gpt_oss_20b_niah_results.csv \
  --input-column prompt \
  --output-column output \
  --model-name openai/gpt-oss-20b \
  --max-context-length 131_072 \
  --max-tokens-per-minute 200_000
```

**Example: Run LongMemEval with gpt-oss-120b:**
```bash
cd experiments/longmemeval
python run/run_longmemeval.py \
  --provider gptoss \
  --input-path ../../data/cleaned_longmemeval_s_focused.csv \
  --output-path ../../results/gpt_oss_120b_longmemeval_focused_results.csv \
  --input-column focused_prompt \
  --output-column output \
  --model-name openai/gpt-oss-120b \
  --max-context-length 131_072 \
  --max-tokens-per-minute 200_000
```

**Full Research Run:**
```bash
# Test mode (quick validation, ~10-20 min, <$2)
./scripts/run_full_research.sh -t

# Full run (all experiments, both models)
./scripts/run_full_research.sh
```

## Adding Additional Providers

The GPT-OSS provider has been implemented in `experiments/models/providers/gptoss.py`. To add other providers:

1. Create `experiments/models/providers/newprovider.py`:
```python
from ..base_provider import BaseProvider

class NewProvider(BaseProvider):
    def process_single_prompt(self, prompt: str, model_name: str, max_output_tokens: int, index: int) -> tuple[int, str]:
        # Implement API call
        response = self.client.some_api_call(...)
        return index, response_text

    def get_client(self):
        # Initialize client with API credentials
        return SomeClient(api_key=os.getenv("NEW_PROVIDER_API_KEY"))
```

2. Add to run scripts' `get_provider()` function (in all 3 experiment run scripts):
```python
elif provider_name.lower() == 'newprovider':
    return NewProvider()
```

3. Update `--provider` choices in argparse:
```python
choices=['openai', 'anthropic', 'google', 'gptoss', 'newprovider']
```

## Key Implementation Details

### Rate Limiting & Batching
- `create_batches()` groups requests by token count to respect `--max-tokens-per-minute`
- 60-second sleep between batches
- ThreadPoolExecutor runs requests concurrently within batches

### Checkpointing & Resume
- All runs check for existing `--output-path` CSV
- Skips rows with valid outputs
- Retries rows with ERROR prefix
- Saves after each batch for fault tolerance

### LLM Judge Pattern
- Evaluation scripts use separate LLM (default: gpt-4.1) to judge correctness
- Returns boolean true/false in `llm_judge_output` column
- Used consistently across all experiments

### Token Counting
- Uses tiktoken for OpenAI models
- Input CSVs must have `token_count` column
- Filters rows exceeding `--max-context-length` before processing

## Data Requirements

Download datasets from Google Drive links in experiment READMEs:
- NIAH: PaulGrahamEssays + arXiv dataset
- LongMemEval: cleaned_longmemeval_s_{focused,full}.csv
- Repeated Words: Generated dynamically during run

Distractor examples: `data/pg_distractors.json`

## Results Organization

Standard output structure:
```
results/
├── {model}_niah_results.csv           # Raw inference
├── {model}_niah_evaluated.csv         # LLM judge scores
├── {model}_heatmap.png                # Visualizations
├── {model}_longmemeval_focused_*.csv
├── {model}_longmemeval_full_*.csv
└── {model}_repeated_words_*.csv
```

## Citation
Hong, Kelly, Anton Troynikov, and Jeff Huber. "Context Rot: How Increasing Input Tokens Impacts LLM Performance." Chroma, July 2025. https://research.trychroma.com/context-rot

# Running the Complete Context Rot Research Workflow

## Quick Start

The `scripts/run_full_research.sh` script automates the entire research workflow for both gpt-oss:20b and gpt-oss:120b models.

### 1. Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Configure API keys
nano .env  # Add your GPT_OSS_BASE_URL, OPENAI_API_KEY, etc.

# Download datasets (see docs/DATASETS.md for links - only ~400 MB total)
# - PaulGrahamEssays → data/PaulGrahamEssays/
# - LongMemEval CSVs → data/cleaned_longmemeval_s_*.csv
```

### 2. Run Complete Workflow

```bash
# Run everything (all 3 experiments, both models)
./scripts/run_full_research.sh

# OR run in test mode first (recommended for first-time users)
./scripts/run_full_research.sh -t
```

## Test Mode

**Recommended for first-time users** to validate setup before running the full research workflow.

### Overview

Test mode runs all 3 experiments with strategically reduced sample sizes to complete in 10-20 minutes at < $2 cost while validating the entire pipeline end-to-end.

```bash
./scripts/run_full_research.sh -t
```

### Sample Reductions

| Experiment | Test Samples | Production Samples | Reduction |
|------------|-------------|-------------------|-----------|
| NIAH Extension | 12 (4 lengths × 3 depths) | 88 (8 lengths × 11 depths) | 86% |
| LongMemEval | 40 (20 focused + 20 full) | 612 (306 focused + 306 full) | 93% |
| Repeated Words | ~15 (5 counts × 3 positions) | ~300 (12 counts × ~25 positions) | 95% |

**Total: ~67 test samples vs ~1,000 production samples**

### Sample Selection Strategy

Test mode uses **stratified sampling** to maintain coverage across all conditions:

- **NIAH**: Tests 4 context lengths (1k, 10k, 100k, 500k) at 3 depths (start, middle, end)
- **LongMemEval**: Every 15th row from both focused and full datasets
- **Repeated Words**: 5 word counts (25, 100, 500, 1k, 5k) at 3 positions each

This ensures test mode validates:
- Short and long context handling
- Needle placement effects
- Focused vs full context differences
- Word count scaling behavior

### Cost & Time Estimates

**Time:** 10-20 minutes total
- NIAH: 2-5 minutes
- LongMemEval: 3-8 minutes
- Repeated Words: 5-7 minutes

**Cost:** $0.50 - $2.00 total
- Model inference: $0.20 - $1.00 (deployment dependent)
- LLM judge evaluation: $0.10 - $0.50
- Minimal compared to production run ($75-$215)

### Output Organization

All test results are prefixed with `test_` to avoid confusion:

```
results/
├── test_gpt_oss_20b_niah_results.csv
├── test_gpt_oss_20b_niah_evaluated.csv
├── test_gpt_oss_20b_niah_heatmap.png
├── test_gpt_oss_20b_longmemeval_focused_results.csv
├── test_gpt_oss_20b_longmemeval_full_results.csv
└── test_gpt_oss_20b_repeated_words_apple_apples.csv
```

**CSV Metadata:** Test mode adds columns:
- `_test_mode`: Boolean flag
- `_test_timestamp`: ISO timestamp of test run
- `_original_dataset_size`: Size before sampling

### Use Cases

**When to use test mode:**
1. **First-time setup** - Validate API keys, env vars, dataset paths
2. **Deployment changes** - Test new model deployments or configurations
3. **Code modifications** - Quick validation after code changes
4. **Pre-production checks** - Sanity check before starting long runs
5. **Debugging** - Fast iteration when troubleshooting issues

**What test mode validates:**
- ✅ Environment variables correctly configured
- ✅ API keys working and authenticated
- ✅ Datasets downloaded and accessible
- ✅ Model inference pipeline functional
- ✅ LLM judge evaluation working
- ✅ Visualization generation successful
- ✅ Checkpoint/resume system operational

### Test Mode Completion

After test mode completes successfully:

```bash
[2025-11-23 10:15:00] ==========================================
[2025-11-23 10:15:00] ALL EXPERIMENTS COMPLETE!
[2025-11-23 10:15:00] ==========================================
```

You'll see:
1. All test visualizations in `results/test_*.png`
2. Complete CSV files with test results
3. Validation that full pipeline works correctly

### Transitioning to Production

Once test mode succeeds:

```bash
# Remove test results (optional)
rm results/test_*

# Run full research
./scripts/run_full_research.sh
```

**Note:** Production runs will not be affected by test mode results. The checkpoint system detects mode mismatches and creates backups if needed.

### Troubleshooting Test Mode

**Test mode fails immediately:**
- Check `.env` file exists and has valid API keys
- Verify datasets downloaded (see docs/DATASETS.md)
- Ensure virtual environment activated

**Test mode times out:**
- Reduce `MAX_TOKENS_PER_MINUTE` in script
- Check network connectivity to API endpoints
- Verify local model server running (if using local deployment)

**Test results look incorrect:**
- Review test visualizations for obvious errors
- Check CSV files for ERROR prefixes in output columns
- Compare test sample counts with expectations above

**Mode mismatch warnings:**
- Test mode detected existing production files
- Files automatically backed up with `.backup` suffix
- Safe to proceed - test mode uses separate output paths

## Configuration Options

Edit the top of `scripts/run_full_research.sh` to customize:

```bash
# Model names (adjust for your deployment)
MODEL_20B="qwen2.5:20b"                    # For local ollama
MODEL_120B="qwen/qwen-2.5-72b-instruct"    # For OpenRouter

# Deployment modes: "local" or "cloud"
DEPLOY_20B="local"    # 20b runs well on local hardware
DEPLOY_120B="cloud"   # 120b typically requires cloud

# Rate limits (auto-adjusted based on deployment, or set manually)
MAX_TOKENS_PER_MINUTE=""  # Leave empty for auto-detection

# Enable/disable experiments (1=run, 0=skip)
RUN_NIAH=1
RUN_LONGMEMEVAL=1
RUN_REPEATED_WORDS=1

# Enable/disable models (1=run, 0=skip)
RUN_20B=1
RUN_120B=1

# LLM Judge model (for evaluation)
JUDGE_MODEL="gpt-4o-mini-2024-07-18"  # See script for cost/quality guide
```

### Deployment Mode Configuration

The script supports two deployment modes for each model:

**Local Deployment (`DEPLOY_20B="local"` or `DEPLOY_120B="local"`)**
- Runs on your hardware (ollama, vLLM, text-generation-inference)
- No API costs (just electricity and hardware depreciation)
- Rate limits: Conservative (50k tokens/min default)
- Best for: 20b model on consumer GPUs or high-end CPUs
- Requirements:
  - 20b: 16GB+ RAM, modern CPU or GPU
  - 120b: High-end multi-GPU setup (less common)

**Cloud Deployment (`DEPLOY_20B="cloud"` or `DEPLOY_120B="cloud"`)**
- Uses cloud APIs (OpenRouter, OpenAI API)
- Costs: $20-150 depending on model and token usage
- Rate limits: Higher (200k tokens/min default)
- Best for: 120b model, or when local hardware insufficient
- Requirements: Valid API keys in `.env` file

**Rate Limit Auto-Detection:**
- The script automatically adjusts `MAX_TOKENS_PER_MINUTE` based on deployment mode
- Local: 50k tokens/min (conservative for most hardware)
- Cloud: 200k tokens/min (adjust per provider limits)
- Override by setting `MAX_TOKENS_PER_MINUTE=100000` explicitly

**Environment Variables:**
```bash
# For local deployment
GPT_OSS_BASE_URL=http://localhost:11434/v1  # ollama
GPT_OSS_API_KEY=dummy

# For cloud deployment (OpenRouter)
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Or use OpenAI API directly
OPENAI_API_KEY=sk-...
```

**Validation:**
The script validates deployment configuration at startup and shows:
- Deployment mode for each model
- Detected rate limits
- Missing environment variables (warnings)
- Cost estimates before running

See `.env.example` for detailed deployment setup instructions.

## Token Tracking

All experiments automatically track token usage in real-time using LiteLLM integration.

### Automatic Tracking

Token tracking is built into the workflow and requires no configuration. When you run any experiment:

```bash
./scripts/run_full_research.sh
# OR
./scripts/run_full_research.sh -t  # Test mode also tracked
```

The system automatically:
- Tracks all API calls and token usage
- Calculates real-time rates (tokens/minute)
- Estimates costs using LiteLLM's cost tables
- Saves live dashboards and final summaries

### Live Monitoring

During execution, monitor token usage in real-time:

```bash
# Open a second terminal and watch the live dashboard
tail -f results/gpt_oss_20b_niah_results_token_dashboard.txt
```

The dashboard updates every 10 API calls and shows:
```
================================================================================
Token Usage Dashboard - gpt_oss_20b_niah_results
Last updated: 2025-11-24 14:23:45
Elapsed time: 8.3 minutes
================================================================================

TOTALS:
  API calls: 45
  Input tokens: 234,567
  Output tokens: 12,345
  Total tokens: 246,912
  Rate: 29,748 tokens/min
  Estimated cost: $0.1234

PER-MODEL BREAKDOWN:
  qwen2.5:20b:
    Calls: 45
    Input: 234,567 tokens
    Output: 12,345 tokens
    Total: 246,912 tokens
    Cost: $0.1234

================================================================================
To monitor live: tail -f results/gpt_oss_20b_niah_results_token_dashboard.txt
================================================================================
```

### Output Files

After each experiment completes, you'll find:

1. **Live Dashboard** (`*_token_dashboard.txt`)
   - Real-time view during execution
   - Final snapshot after completion
   - Human-readable format

2. **JSON Summary** (`*_token_summary.json`)
   - Complete metrics in machine-readable format
   - Suitable for programmatic analysis
   - Contains per-model breakdowns

Example file structure:
```
results/
├── gpt_oss_20b_niah_results.csv
├── gpt_oss_20b_niah_results_token_dashboard.txt
├── gpt_oss_20b_niah_results_token_summary.json
├── gpt_oss_20b_longmemeval_focused_results.csv
├── gpt_oss_20b_longmemeval_focused_results_token_dashboard.txt
└── gpt_oss_20b_longmemeval_focused_results_token_summary.json
```

### What's Tracked

Token tracking captures:
- **API call count**: Total number of inference requests
- **Input tokens**: Tokens sent to the model (prompts)
- **Output tokens**: Tokens generated by the model (responses)
- **Rate**: Average tokens/minute throughout execution
- **Cost**: Estimated USD cost based on LiteLLM's pricing tables
- **Per-model stats**: Breakdown by model name for multi-model runs

### Implementation Details

Token tracking uses LiteLLM's unified API:
- **Provider-agnostic**: Works with OpenAI, Anthropic, Google, local models
- **Automatic cost calculation**: LiteLLM maintains up-to-date pricing
- **Zero configuration**: Built into BaseProvider.main()
- **Minimal overhead**: Negligible performance impact

All providers (OpenAI, Anthropic, Google, GptOss) use `litellm.completion()` which automatically captures token usage from the response metadata.

### Cost Accuracy

LiteLLM cost estimates are based on:
- **OpenAI**: Official API pricing (highly accurate)
- **Anthropic**: Official API pricing (highly accurate)
- **Google**: Vertex AI pricing (highly accurate)
- **Local/OpenRouter**: May not have cost data (shows $0.00)

For local deployments (ollama, vLLM), cost tracking shows $0.00 since there are no API charges. Hardware costs (electricity, compute depreciation) are not tracked.

### Troubleshooting

**Dashboard file not found:**
- Wait for first batch to complete (dashboard created after initialization)
- Check that experiment is actually running (not skipped due to checkpointing)

**Cost shows $0.00:**
- Expected for local deployments (ollama, vLLM, TGI)
- LiteLLM may not have pricing for custom/new models
- Token counts are still accurate

**Dashboard not updating:**
- Updates occur every 10 API calls
- For small test runs, may update less frequently
- Check that tail -f is pointing to correct file

## Selective Execution

### Run only NIAH experiment:
```bash
# Edit scripts/run_full_research.sh
RUN_NIAH=1
RUN_LONGMEMEVAL=0
RUN_REPEATED_WORDS=0
```

### Run only 20b model (skip 120b):
```bash
# Edit scripts/run_full_research.sh
RUN_20B=1
RUN_120B=0
```

### Test with Repeated Words only (no dataset download needed):
```bash
# Edit scripts/run_full_research.sh
RUN_NIAH=0
RUN_LONGMEMEVAL=0
RUN_REPEATED_WORDS=1
RUN_20B=1
RUN_120B=0
```

## Checkpoint & Resume

The workflow supports **fully automatic checkpoint/resume** at both the workflow and row levels:

### Workflow-Level Checkpointing
- **State tracking**: JSON file (`.workflow_state.json`) tracks each experiment/model/step completion
- **Smart skipping**: Automatically skips completed steps on restart
- **Progress dashboard**: Shows real-time status of all experiments at startup and on interruption
- **Graceful interruption**: Ctrl+C shows status summary and clean exit

### Row-Level Checkpointing
- Each Python script checks for existing results
- Skips completed rows (validated by checking for NULL/ERROR values)
- Retries rows with errors
- Saves progress after each batch

### Usage

```bash
# Start the workflow
./scripts/run_full_research.sh

# Workflow Status Dashboard shown at start:
# ========================================
# Workflow Status Dashboard
# ========================================
#
# NIAH Extension:
#   ✗ 20b model - not started
#   ✗ 120b model - not started
#
# LongMemEval:
#   ✗ 20b model - focused: 0/0, full: 0/0
#   ...

# If interrupted (Ctrl+C), see status and resume:
# [Ctrl+C pressed]
# ========================================
# Workflow interrupted! Progress saved.
# ========================================
#
# Workflow Status Dashboard
# ========================================
#
# NIAH Extension:
#   ✓ 20b model - complete (88/88)
#   ⧗ 120b model - in progress (45/88)
#
# To resume, re-run: ./scripts/run_full_research.sh

# Resume (automatically continues from last incomplete step)
./scripts/run_full_research.sh
# Skipping niah 20b inference - already marked complete
# Skipping niah 20b evaluation - output complete (88/88)
# Skipping niah 20b visualization - already marked complete
# Running niah 120b inference...
```

### State Management

**State file location**: `results/.workflow_state.json`

**View state**:
```bash
cat results/.workflow_state.json
```

Example state file:
```json
{
  "niah": {
    "20b": {
      "inference": true,
      "evaluation": true,
      "visualization": true
    },
    "120b": {
      "inference": true
    }
  }
}
```

**Clear state** (force full re-run):
```bash
rm results/.workflow_state.json
./scripts/run_full_research.sh
```

**Manual state editing**:
You can manually edit the JSON file to force re-running specific steps:
```bash
# Force re-run of NIAH 20b evaluation
nano results/.workflow_state.json
# Change "evaluation": true to "evaluation": false
# Or delete the "evaluation" key entirely
```

## Execution Time Estimates

**With local ollama (20b only):**
- NIAH Extension: ~2-4 hours
- LongMemEval: ~3-5 hours (high token count)
- Repeated Words: ~4-8 hours (high output tokens)
- **Total: ~9-17 hours**

**With OpenRouter (120b):**
- Similar or faster depending on API rate limits
- Costs apply based on token usage

**Strategy:** Run overnight or over weekend for complete results.

## Expected Outputs

### NIAH Extension
```
results/
├── gpt_oss_20b_niah_results.csv
├── gpt_oss_20b_niah_evaluated.csv
├── gpt_oss_20b_niah_heatmap.png
├── gpt_oss_120b_niah_results.csv
├── gpt_oss_120b_niah_evaluated.csv
└── gpt_oss_120b_niah_heatmap.png
```

### LongMemEval
```
results/
├── gpt_oss_20b_longmemeval_focused_results.csv
├── gpt_oss_20b_longmemeval_focused_evaluated.csv
├── gpt_oss_20b_longmemeval_full_results.csv
├── gpt_oss_20b_longmemeval_full_evaluated.csv
├── gpt_oss_20b_longmemeval.png
└── (same for 120b)
```

### Repeated Words
```
results/
├── gpt_oss_20b_repeated_words_apple_apples_evaluated/
│   ├── token_count_performance.png
│   ├── levenshtein_score.png
│   ├── modified_word_present.png
│   ├── position_accuracy.png
│   └── word_count_delta.png
└── (same for 120b)
```

## Monitoring Progress

The script logs all activity with timestamps:
```
[2025-11-19 23:30:15] Starting NIAH Extension Experiment
[2025-11-19 23:30:20] Step 1/5: Generating NIAH haystacks...
[2025-11-19 23:35:42] Step 2/5: Running NIAH inference with 20b model...
```

You can also monitor individual experiment progress in the CSV files as they're written.

## Troubleshooting

### "Missing required datasets"
Download from links in docs/DATASETS.md. You can download selectively:
- NIAH needs: PaulGrahamEssays
- LongMemEval needs: cleaned_longmemeval_s_*.csv
- Repeated Words: No download needed (generates data)

### "Virtual environment not activated"
```bash
source venv/bin/activate
./scripts/run_full_research.sh
```

### ".env file not found"
```bash
cp .env.example .env
nano .env  # Add your API keys
```

### API rate limits exceeded
Edit `MAX_TOKENS_PER_MINUTE` in the script to lower value:
```bash
MAX_TOKENS_PER_MINUTE=50000  # Reduce if hitting rate limits
```

### Out of memory (local ollama)
Reduce context length or run only smaller experiments:
```bash
RUN_REPEATED_WORDS=1  # Start with smallest experiment
RUN_NIAH=0
RUN_LONGMEMEVAL=0
```

## Advanced Usage

### Dry Run (check configuration without running)
Comment out experiment sections in `scripts/run_full_research.sh`:
```bash
# if [ ${RUN_NIAH} -eq 1 ]; then
#     ...experiments...
# fi
```

### Run experiments separately
```bash
# Just NIAH
cd experiments/niah_extension
python run/run_niah_extension.py --provider gptoss ...

# Just LongMemEval
cd experiments/longmemeval
python run/run_longmemeval.py --provider gptoss ...

# Just Repeated Words
cd experiments/repeated_words
python run/run_repeated_words.py --provider gptoss ...
```

### Custom model names
Edit model configurations at top of script:
```bash
MODEL_20B="your-custom-20b-model"
MODEL_120B="your-custom-120b-model"
```

## Cost Estimation

The script displays a cost estimate at startup based on your deployment configuration.

**LLM Judge (for evaluation):**
- Default: gpt-4o-mini (~$5-15 for all evaluations)
- Higher quality: gpt-4o (~$50-150 for all evaluations)
- Budget: gpt-3.5-turbo (~$2-8 for all evaluations)
- Total evaluations: ~1,400 calls (NIAH + LongMemEval)
- See script comments for detailed judge model trade-offs

**Model Inference:**
- **Local deployment** (`DEPLOY_*="local"`):
  - Free API costs
  - Hardware costs only (electricity, compute depreciation)
  - 20b model: ~$0 per run (common setup)
  - 120b model: Requires high-end multi-GPU setup

- **Cloud deployment** (`DEPLOY_*="cloud"`):
  - 20b model: ~$20-50 per run (all experiments)
  - 120b model: ~$50-150 per run (all experiments)
  - Varies by provider (OpenRouter, OpenAI)
  - LongMemEval has highest token usage (~34.6M input tokens per model)
  - Repeated Words has high output token usage (~32k tokens per test)

**Typical Total Costs:**
- Local 20b + cloud 120b + gpt-4o-mini judge: ~$55-165
- Cloud both models + gpt-4o-mini judge: ~$75-215
- Local 20b only + gpt-4o-mini judge: ~$5-15 (judge only)

The script shows your specific cost estimate based on `DEPLOY_20B`, `DEPLOY_120B`, and `JUDGE_MODEL` settings.

## Next Steps After Completion

1. **Review visualizations** in `results/` directory
2. **Analyze CSV files** for detailed metrics and performance patterns
3. **Compare 20b vs 120b** context rot characteristics
4. **Cite the paper** if publishing results (see README.md)

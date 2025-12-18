# Context Rot: How Increasing Input Tokens Impacts LLM Performance

This repository contains the toolkit for replicating results from our [technical report](https://research.trychroma.com/context-rot).

## Motivation

Large Language Models (LLMs) are typically presumed to process context uniformly—that is, the model should handle the 10,000th token just as reliably as the 100th. However, in practice, this assumption does not hold. We observe that model performance varies significantly as input length changes, even on simple tasks.

<p align="center">
  <img src="images/image.png" alt="repeated words results" width="1000"/><br>
  <span style="font-size: 1em; color: #555;">Latest Models on Repeated Words Task</span>
</p>

## Experiments

Our experiments are organized under the `experiments/` folder:

### 1. **NIAH Extension** (`experiments/niah_extension/`)
Extension of [Needle in a Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) to examine the effects of needles with semantic, rather than direct lexical matches, as well as the effects of introducing variations to the haystack content. 

### 2. **LongMemEval** (`experiments/longmemeval/`)
[LongMemEval](https://arxiv.org/abs/2410.10813) task.

### 3. **Repeated Words** (`experiments/repeated_words/`)
Tests model performance on replicating a sequence of repeated words.

Each experiment contains detailed instructions in their respective `README.md` files.

## Data

Datasets can be downloaded [here](https://drive.google.com/drive/folders/1FuOysriSotnYasJUbZJzn31SWt85_3yf?usp=drive_link).

See [`docs/DATASETS.md`](docs/DATASETS.md) for detailed information about datasets, sizes (~400 MB total), and download links.

## Quick Start

### Option 1: Test Mode (Recommended for First-Time Users)

Validate your setup in 10-20 minutes with reduced samples:

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
nano .env  # Add your API keys

# 3. Run test workflow
./scripts/run_full_research.sh -t
```

**Test mode runs:**
- NIAH: 12 samples (vs 88 production)
- LongMemEval: 40 samples (vs 612 production)
- Repeated Words: 15 samples (vs ~300 production)
- **Cost: < $2, Time: 10-20 minutes**

Results are prefixed with `test_` to avoid confusion with production runs.

### Option 2: Full Research Workflow

After validating with test mode, run the complete research:

```bash
./scripts/run_full_research.sh
```

**Token Tracking**: All experiments automatically track token usage in real-time. Monitor live dashboards during execution:
```bash
tail -f results/gpt_oss_20b_niah_results_token_dashboard.txt
```

See [`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md) for detailed instructions and configuration options.

### Option 3: Manual Execution

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies: `pip install -r requirements.txt`
4. Set up environment variables:
   - **OpenAI**: `OPENAI_API_KEY`
   - **Anthropic**: `ANTHROPIC_API_KEY`
   - **Google**: `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_MODEL_PATH`
   - **GPT-OSS** (Qwen models): See `.env.example` for configuration options

5. Navigate to specific experiment folder and follow README instructions

## Supported Models

- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5
- **Anthropic**: Claude 3 family
- **Google**: Gemini models via Vertex AI
- **GPT-OSS**: Qwen models (gpt-oss:20b, gpt-oss:120b) via local deployment, OpenRouter, or OpenAI API

See [`CLAUDE.md`](CLAUDE.md) for detailed usage examples with all providers.

## Citation
If you find this work useful, please cite our technical report:
```
@techreport{hong2025context,
  title = {Context Rot: How Increasing Input Tokens Impacts LLM Performance},
  author = {Hong, Kelly and Troynikov, Anton and Huber, Jeff},
  year = {2025},
  month = {July},
  institution = {Chroma},
  url = {https://research.trychroma.com/context-rot},
}
```

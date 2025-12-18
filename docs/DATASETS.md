# Dataset Requirements for Context-Rot Research

> **Last Updated:** 2025-11-19

This document catalogs all datasets required to replicate experiments from the [Context Rot technical report](https://research.trychroma.com/context-rot).

## Storage Requirements Summary

**Total Estimated Storage: ~400 MB** (16TB portable drive is NOT required)

All datasets are lightweight and can be stored on standard local storage. The largest single dataset is only 217 MB.

---

## Required Datasets

### 1. PaulGrahamEssays Dataset

**Used by:** NIAH Extension experiment (`experiments/niah_extension/`)

**Download Link:** [Google Drive](https://drive.google.com/drive/folders/14uHYF65yu7cNGANungZX1NRboqwHHuVB?usp=sharing)

**Estimated Size:** ~2-3 MB (collection of text files)

**Format:**
- Individual `.txt` files, one per essay
- Plain text format, UTF-8 encoded
- Typical range: 100-200 essays at ~500KB total words

**Installation Path:** `data/PaulGrahamEssays/`

**Purpose:**
Used as haystack content for Needle in a Haystack experiments. Essays are concatenated to create long contexts at various token lengths (500 to 900,000 tokens).

**Preprocessing:** None required - used directly as source material for haystack generation.

**Additional Files:**
- Needles and distractors are included in the same Google Drive folder
- `data/pg_distractors.json` (946 bytes, already in repository)

**Alternative Sources:**
- HuggingFace: [chromadb/paul_graham_essay](https://huggingface.co/datasets/chromadb/paul_graham_essay) (1.27 MB, 104 essays)
- HuggingFace: [sgoel9/paul_graham_essays](https://huggingface.co/datasets/sgoel9/paul_graham_essays) (2.95 MB)
- Kaggle: [Paul Graham Essays](https://www.kaggle.com/datasets/krsoninikhil/pual-graham-essays)

---

### 2. arXiv AI Papers Dataset (ai-arxiv2)

**Used by:** NIAH Extension experiment (`experiments/niah_extension/`)

**Download Link:** [HuggingFace](https://huggingface.co/datasets/jamescalam/ai-arxiv2)

**Estimated Size:**
- Downloaded: 217 MB
- Parquet format: 109 MB

**Format:**
- Parquet format (auto-converted by HuggingFace)
- 2,670 rows (AI-related academic papers from arXiv)

**Fields:**
- `id` - Unique identifier
- `title` - Paper title (8-162 characters)
- `summary` - Abstract (228-1,920 characters)
- `content` - Full paper content (3.91k-873k characters)
- `authors` - Author names
- `categories` - Subject classifications
- `published` / `updated` - Date fields
- `source` - Source URL
- `journal_ref` - Journal reference (optional)
- `primary_category` - Main subject area
- `references` - Citation references (dict)

**Installation:**
```bash
# Using HuggingFace datasets library
from datasets import load_dataset
dataset = load_dataset("jamescalam/ai-arxiv2")
```

**Purpose:**
Alternative haystack content source for NIAH experiments, particularly for testing with academic/technical content instead of essays.

**Preprocessing:** None required for basic usage. Full content field can be used directly.

---

### 3. LongMemEval Cleaned Datasets

**Used by:** LongMemEval experiment (`experiments/longmemeval/`)

**Download Links:**
- Primary: [Google Drive](https://drive.google.com/drive/folders/1AS1oytdcCH3y6p-DNuaNYI7I48IFgbfe?usp=sharing)
- Alternative: [HuggingFace](https://huggingface.co/datasets/kellyhongg/cleaned-longmemeval-s)

**Required Files:**
1. `cleaned_longmemeval_s_focused.csv`
2. `cleaned_longmemeval_s_full.csv`

**Estimated Size:**
- Combined download: ~158 MB
- Parquet format: ~76 MB
- 306 rows total

**Format:** CSV files

**Fields:**
- `custom_id` - Unique identifier
- `question` - Question to be answered
- `answer` - Correct answer
- `full_input` - Full 113k token conversation history
- `full_input_tokens` - Token count for full input (~113,000 tokens/row)
- `focused_input` - Relevant excerpts only
- `focused_input_tokens` - Token count for focused input

**Installation Path:** `data/`

**Purpose:**
Tests model performance in two conditions:
- **Focused**: Contains only relevant context (simple reasoning task)
- **Full**: Contains full 113k token history with irrelevant context (retrieval + reasoning)

**Preprocessing:**
Already cleaned to remove ambiguous question-answer pairs. No additional preprocessing needed.

**Token Economics:**
- Full dataset: ~113,000 tokens per row × 306 rows = ~34.6 million tokens total
- Focused dataset: Significantly smaller, varies by row
- Plan API budgets accordingly for inference runs

---

### 4. Repeated Words Dataset

**Used by:** Repeated Words experiment (`experiments/repeated_words/`)

**Download:** NOT REQUIRED - Generated programmatically

**Size:** N/A (generated on-the-fly)

**Format:** Generated CSV with prompt/response pairs

**Purpose:**
Tests model's ability to replicate sequences of repeated words with a single unique word inserted at specific positions.

**Generation:**
The experiment script `run_repeated_words.py` generates prompts directly:
- Creates sequences of repeated words (e.g., "apple apple apple...")
- Inserts modified word at various positions (e.g., "apples")
- Tests across different context lengths
- No external dataset required

**Parameters:**
```bash
--common-word apple \
--modified-word apples \
--model-max-output-tokens 32_768
```

---

## Dataset Directory Structure

```
context-rot/
├── data/
│   ├── PaulGrahamEssays/           # ~3 MB
│   │   ├── essay1.txt
│   │   ├── essay2.txt
│   │   └── ...
│   ├── cleaned_longmemeval_s_focused.csv  # ~79 MB
│   ├── cleaned_longmemeval_s_full.csv     # ~79 MB
│   ├── pg_distractors.json                # 946 bytes (included)
│   └── niah_prompts/                      # Generated by experiments
│       ├── niah_prompts_sequential.csv
│       └── niah_prompts_shuffled.csv
└── results/                               # Generated outputs
    └── ...
```

---

## Download Instructions

### Quick Setup (All Datasets)

1. **Paul Graham Essays:**
   ```bash
   # Download from Google Drive link
   # Unzip to data/PaulGrahamEssays/
   ```

2. **arXiv Dataset (optional):**
   ```python
   from datasets import load_dataset
   dataset = load_dataset("jamescalam/ai-arxiv2")
   # Or download manually from HuggingFace
   ```

3. **LongMemEval Cleaned:**
   ```bash
   # Download both CSV files from Google Drive
   # Place in data/ directory
   ```

### Using HuggingFace Datasets Library

```bash
pip install datasets

python -c "
from datasets import load_dataset

# LongMemEval
longmem = load_dataset('kellyhongg/cleaned-longmemeval-s')
longmem.save_to_disk('data/longmemeval')

# arXiv (optional)
arxiv = load_dataset('jamescalam/ai-arxiv2')
arxiv.save_to_disk('data/arxiv')

# Paul Graham Essays
pg = load_dataset('chromadb/paul_graham_essay')
pg.save_to_disk('data/paul_graham')
"
```

---

## Verification Checklist

Before running experiments, verify you have:

- [ ] `data/PaulGrahamEssays/*.txt` files exist (for NIAH Extension)
- [ ] `data/cleaned_longmemeval_s_focused.csv` exists (for LongMemEval)
- [ ] `data/cleaned_longmemeval_s_full.csv` exists (for LongMemEval)
- [ ] `data/pg_distractors.json` exists (already in repo)
- [ ] Total data directory size: ~250-400 MB
- [ ] arXiv dataset downloaded if using alternative haystack content

**Storage Verdict:** Your 16TB portable drive is complete overkill for this project. All datasets fit comfortably in under 500 MB.

---

## Dataset Citations

### LongMemEval
```bibtex
@inproceedings{longmemeval2025,
  title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
  author={Wu, Xiaowu and others},
  booktitle={ICLR},
  year={2025}
}
```

### arXiv Dataset
```
@dataset{jamescalam_ai_arxiv2,
  author = {James Briggs},
  title = {AI arXiv Papers Dataset},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/jamescalam/ai-arxiv2}
}
```

---

## Troubleshooting

### Missing Dataset Files

If experiments fail with "file not found" errors:
1. Check the `data/` directory exists in repository root
2. Verify CSV filenames match exactly (case-sensitive)
3. Ensure `.txt` files in PaulGrahamEssays folder have correct extensions

### Large File Warnings

The LongMemEval full CSV files contain ~113k tokens per row. Some tools may:
- Show warnings about file size
- Take time to load (normal for 158 MB CSV)
- Require adequate RAM (recommend 8GB+ available)

### Token Budget Planning

For API-based inference:
- NIAH Extension: Variable, depends on context lengths tested (500-900k tokens)
- LongMemEval Full: 306 rows × 113k tokens = ~34.6M input tokens
- LongMemEval Focused: Significantly less, varies by row
- Repeated Words: High output token cost due to long sequences

Budget accordingly for OpenAI/Anthropic/Google API costs.

---

## Additional Resources

- [Main Repository](https://github.com/chroma-core/context-rot) (if public)
- [Technical Report](https://research.trychroma.com/context-rot)
- [LongMemEval GitHub](https://github.com/xiaowu0162/LongMemEval)
- [LongMemEval Website](https://xiaowu0162.github.io/long-mem-eval/)
- [Needle in a Haystack Original](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)

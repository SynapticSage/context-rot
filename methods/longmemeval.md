# LongMemEval: Retrieval vs Reasoning

## Overview

This experiment uses the LongMemEval benchmark to systematically test whether **additional context helps or hinders** model performance on conversational question-answering tasks. We compare performance when models receive only relevant context (**focused**) versus the full conversation history (**full**).

---

## Experimental Design

### Research Question

> Does requiring retrieval in addition to reasoning degrade model performance on long-context QA tasks?

### Conditions

| Condition | Context Size | Task Complexity |
|-----------|--------------|-----------------|
| **Focused** | ~4k tokens | Reasoning only |
| **Full** | ~113k tokens | Retrieval + Reasoning |

### Hypothesis

Models performing **two tasks** (retrieval + reasoning) should exhibit degraded performance compared to models performing **one task** (reasoning only), isolating the cost of in-context retrieval.

---

## Dataset

### Source

LongMemEval benchmark (Di et al., 2024): Multi-turn conversational QA requiring retrieval across extended dialogue history.

### Preprocessing

Two parallel datasets are constructed from the same questions:

**Focused Dataset** (`cleaned_longmemeval_s_focused.csv`):
- Contains only the conversation turns relevant to answering each question
- Average context: ~4,000 tokens
- Column: `focused_prompt`

**Full Dataset** (`cleaned_longmemeval_s_full.csv`):
- Contains the complete conversation history
- Average context: ~113,000 tokens
- Column: `full_prompt`

---

## Inference Protocol

### Model Configuration

| Parameter | Value |
|-----------|-------|
| `temperature` | 0 |
| `max_output_tokens` | 256 |
| `tokenizer` | `o200k_base` |

### Rate Limiting

- Batch processing with token-aware rate limiting
- 60-second cooldown between batches
- Configurable `max_tokens_per_minute`

### Context Truncation

For models with context limits below 113k tokens:
- Truncation from **front** of context (preserves question at end)
- Iterative removal with sentence boundary detection
- Metadata logged: `original_tokens`, `final_tokens`, `truncated`

---

## Evaluation Protocol

### LLM Judge

- **Model**: GPT-4.1 (`gpt-4.1-2025-04-14`)
- **Output**: Binary correctness ($\text{correct} \in \{0, 1\}$)

**Judge Prompt:**
```
Given this question and the CORRECT answer, determine whether the response
is correct (meaning it factually aligns with the correct answer).

In some cases, 0 and "I do not have an answer" are considered to be both correct.
If both responses say that there is no answer, this should be judged as true.
If the correct answer contains an answer, but the response abstains from
answering, this should be judged as false.

Question: {question}
CORRECT answer: {correct_answer}
Response to judge: {output}

Instructions: Respond with only "true" or "false".
```

### Abstention Handling

- If ground truth is "no answer" and model abstains → **correct**
- If ground truth contains answer and model abstains → **incorrect**

---

## Metrics

### Primary Metric

**Accuracy** per condition:

$$
\text{Accuracy}_{\text{condition}} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[\text{judge}(y_i) = \text{true}]
$$

### Confidence Intervals

Bootstrap 95% CI with $B = 200$ resamples:

$$
\text{CI}_{95} = \left[ \hat{\mu}^*_{2.5\%}, \hat{\mu}^*_{97.5\%} \right]
$$

where $\hat{\mu}^*$ denotes the bootstrap distribution of sample means.

**Implementation:**
```python
def bootstrap_ci(data, n_bootstrap=200, ci=0.95):
    rng = np.random.default_rng(42)  # Reproducibility
    bootstrap_means = [np.mean(rng.choice(data, len(data), replace=True))
                       for _ in range(n_bootstrap)]
    return np.percentile(bootstrap_means, [(1-ci)/2*100, (1+ci)/2*100])
```

---

## Visualization

### Bar Chart Comparison

- Two bars: Focused vs Full condition
- Error bars: 95% bootstrap CI
- Annotations: Mean accuracy with CI range

### Figure Elements

| Element | Description |
|---------|-------------|
| Title | "LongMemEval: Retrieval vs Reasoning - {Model}" |
| Subtitle | "Accuracy on multi-turn QA with varying context lengths" |
| X-axis | Condition labels with token count annotations |
| Y-axis | Average Score [0, 1] |
| Footer | Sample sizes and CI methodology |

---

## Sample Sizes

| Condition | Samples |
|-----------|---------|
| Focused | 612 |
| Full | 612 |
| **Test Mode** | 40 per condition |

---

## Statistical Interpretation

### Expected Outcomes

- **No context rot**: $\text{Acc}_{\text{full}} \approx \text{Acc}_{\text{focused}}$
- **Context rot present**: $\text{Acc}_{\text{full}} < \text{Acc}_{\text{focused}}$

### Effect Size

The difference $\Delta = \text{Acc}_{\text{focused}} - \text{Acc}_{\text{full}}$ quantifies the **retrieval cost**—the performance penalty incurred by requiring the model to locate relevant information within irrelevant context.

---

## References

- Di, Y., et al. (2024). *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*. arXiv:2410.10813.
- Hong, K., Troynikov, A., & Huber, J. (2025). *Context Rot: How Increasing Input Tokens Impacts LLM Performance*. Chroma Research.

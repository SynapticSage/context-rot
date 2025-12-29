# Needle in a Haystack Extension

## Overview

This experiment extends the standard Needle in a Haystack (NIAH) paradigm to investigate model behavior with **semantic needle-question pairs** (low lexical overlap) and **distractor injection**.

---

## Experimental Design

### Independent Variables

| Variable | Levels | Description |
|----------|--------|-------------|
| Context Length | $L \in \{500, 1\text{k}, 5\text{k}, 10\text{k}, 50\text{k}, 100\text{k}, 500\text{k}, 900\text{k}\}$ | Total tokens in prompt |
| Needle Depth | $d \in \{0, 10, 20, \ldots, 100\}$ % | Position of needle as percentage of context |
| Haystack Mode | Sequential, Shuffled | Sentence ordering within context |
| Distractors | None, $k$ distractors | Semantically similar false answers |

### Dependent Variable

- **Accuracy**: Binary correctness judged by LLM ($\text{acc} \in \{0, 1\}$)

---

## Stimulus Construction

### Haystack Generation

**Sequential Mode:**
- Concatenate source texts (e.g., Paul Graham essays) in document order
- Truncate to target token count $L$ using `tiktoken` (encoding: `o200k_base`)

**Shuffled Mode:**
- Split all source texts into sentences at period boundaries
- Shuffle sentences with seeded RNG for reproducibility
- Concatenate until target token count reached

Formally, for target length $L$:

$$
\text{Haystack} = \text{Truncate}\left(\bigoplus_{i=1}^{n} s_i, L - |\text{needle}| - |\text{overhead}|\right)
$$

where $\bigoplus$ denotes concatenation and overhead includes prompt template tokens.

### Needle Insertion

The needle is inserted at depth $d$ (percentage of context):

$$
\text{insertion\_point} = \lfloor |\text{haystack\_tokens}| \times \frac{d}{100} \rfloor
$$

- **Boundary alignment**: Insertion point is adjusted backward to the nearest sentence boundary (period token)
- **Edge cases**: $d=0$ prepends needle; $d=100$ appends needle

### Distractor Injection

When distractors are enabled:
- Each distractor is inserted at a random sentence boundary
- Distractors are semantically related but factually incorrect alternatives to the needle

---

## Prompt Template

```
You are a helpful AI bot that answers questions for a user. Keep your response short and direct

<document_content>
{haystack_with_needle}
<document_content>

Here is the user question:
<question>
{retrieval_question}
<question>

Don't give information outside the document or repeat your findings.
Assistant: Here is the most relevant information in the documents:
```

---

## Evaluation Protocol

### LLM Judge

- **Model**: GPT-4.1 (`gpt-4.1-2025-04-14`)
- **Task**: Binary classification of response correctness
- **Prompt**: Compares model output against ground truth needle

**Judge Prompt Template:**
```
Given this question and the CORRECT answer, determine whether the response
is correct (meaning it factually aligns with the correct answer).

Question: {question}
CORRECT answer: {correct_answer}
Response to judge: {output}

Instructions: Respond with only "true" or "false".
```

### Distractor Analysis (Optional)

For incorrect responses, a secondary judge identifies which distractor (if any) was selected:

$$
\text{distractor\_label} \in \{-1, 0, 1, \ldots, k-1\}
$$

where $-1$ indicates no specific distractor was chosen.

---

## Metrics

### Primary Metric

**Accuracy** at each $(L, d)$ cell:

$$
\text{Accuracy}(L, d) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[\text{judge}(y_i) = \text{true}]
$$

where $n$ = trials per cell (default: 5).

### Visualization

- **Heatmap**: Rows = context length, Columns = needle depth
- **Color scale**: Accuracy from 0 (red) to 1 (green)

---

## Experimental Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trials_per_cell` | 5 | Replicates per $(L, d)$ combination |
| `tokenizer` | `o200k_base` | OpenAI tiktoken encoding |
| `temperature` | 0 | Deterministic model output |
| `max_output_tokens` | 256 | Maximum response length |

### Sample Sizes

- **Full experiment**: 8 lengths × 11 depths × 5 trials = **440 samples**
- **Test mode**: 4 lengths × 3 depths × 1 trial = **12 samples**

---

## Statistical Considerations

- **Within-cell variance**: Multiple trials per cell with varied haystack content (different start indices or shuffle seeds)
- **Reproducibility**: Shuffle seeds are deterministic: $\text{seed} = 42 + \text{trial} \times 1000$

---

## References

- Kamradt, G. (2023). *Needle in a Haystack Test*. GitHub.
- Hong, K., Troynikov, A., & Huber, J. (2025). *Context Rot: How Increasing Input Tokens Impacts LLM Performance*. Chroma Research.

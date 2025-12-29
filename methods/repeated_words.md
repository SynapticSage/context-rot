# Repeated Words: Exact Replication Task

## Overview

This experiment tests **exact sequence replication** under increasing input length. The model must reproduce a sequence of repeated words containing a single unique ("modified") word at a specific position. This isolates the model's ability to maintain fidelity during long-context copying tasks.

---

## Experimental Design

### Task Definition

Given input sequence $S = (w_1, w_2, \ldots, w_n)$ where:

$$
w_i = \begin{cases}
\text{modified\_word} & \text{if } i = k \\
\text{common\_word} & \text{otherwise}
\end{cases}
$$

The model must output $\hat{S} = S$ exactly.

**Example:**
```
Input:  "apple apple apple apples apple apple"
Output: "apple apple apple apples apple apple"
                        ↑
                   modified word at position k=3
```

### Independent Variables

| Variable | Levels | Description |
|----------|--------|-------------|
| Sequence Length | $n \in \{25, 50, 75, 100, 250, 500, 750, 1\text{k}, 2.5\text{k}, 5\text{k}, 7.5\text{k}, 10\text{k}\}$ | Number of words |
| Modified Position | $k \in \{0, 1, \ldots, n-1\}$ | Position of unique word |

### Position Sampling

For sequences with $n \geq 100$ words:
- Sample 100 evenly-spaced positions: $k \in \{0, \lfloor n/100 \rfloor, 2\lfloor n/100 \rfloor, \ldots, n-1\}$

For sequences with $n < 100$ words:
- Test all positions: $k \in \{0, 1, \ldots, n-1\}$

---

## Stimulus Construction

### Prompt Template

```
Simply replicate the following text, output the exact same text: {sequence}
```

### Token Estimation

Using `tiktoken` with `o200k_base` encoding:

$$
\text{input\_tokens} = |\text{encode}(\text{prompt})|
$$

$$
\text{max\_output\_tokens} = 2 \times \text{input\_tokens}
$$

**Constraint**: $\text{max\_output\_tokens} \leq \text{model\_max\_output\_tokens}$

---

## Evaluation Metrics

### 1. Normalized Levenshtein Score

Measures character-level similarity:

$$
\text{Lev}_{\text{norm}}(S, \hat{S}) = 1 - \frac{\text{Lev}(S, \hat{S})}{\max(|S|, |\hat{S}|)}
$$

where $\text{Lev}(S, \hat{S})$ is the Levenshtein edit distance.

**Interpretation:**
- $1.0$ = perfect match
- $0.0$ = completely different

### 2. Modified Word Present

Binary indicator of whether the modified word appears in output:

$$
\text{present}(k) = \mathbb{1}[\text{modified\_word} \in \hat{S}]
$$

**Detection Logic:**
- For final position ($k = n-1$): Match pattern `" " + modified_word`
- Otherwise: Match pattern `modified_word + " "`

### 3. Position Accuracy

Whether the modified word appears at the correct character offset:

$$
\text{position\_correct}(k) = \mathbb{1}[\text{index}(\text{modified\_word}, S) = \text{index}(\text{modified\_word}, \hat{S})]
$$

**Precondition**: Only evaluated when `modified_word_present = True`

### 4. Word Count Delta

Difference between expected and actual word counts:

$$
\Delta_{\text{words}} = |S_{\text{gold}}|_{\text{words}} - |\hat{S}|_{\text{words}}
$$

**Interpretation:**
- $\Delta > 0$: Model omitted words
- $\Delta < 0$: Model added words
- $\Delta = 0$: Correct length

---

## Refusal Detection

Responses are filtered for refusals using heuristics:

1. **Other word present**: Output contains words other than `common_word` and `modified_word`
2. **Low repetition count**: `common_word` appears fewer than 15 times

Flagged responses are excluded from primary analysis.

---

## Visualization

### Binned Position Plots

For each sequence length $n$, positions are binned into 20 equal intervals:

$$
\text{bin}_j = \left[ \frac{j(n-1)}{20}, \frac{(j+1)(n-1)}{20} \right), \quad j \in \{0, \ldots, 19\}
$$

X-axis is normalized to percentage: $x = \frac{k}{n-1} \times 100\%$

### Generated Plots

| Plot | Y-axis | Interpretation |
|------|--------|----------------|
| `levenshtein_score.png` | Normalized Levenshtein | Overall fidelity by position |
| `modified_word_present.png` | Presence rate | Whether unique word survives |
| `position_accuracy.png` | Position accuracy | Whether unique word is correctly placed |
| `word_count_delta.png` | $\Delta_{\text{words}}$ | Length deviation (hatched bars for negative) |
| `token_count_performance.png` | Levenshtein vs tokens | Performance degradation curve |

### Token Count Performance

Log-scale binning of input token counts:

$$
\text{bins} = \text{logspace}(\log_{10}(\min T), \log_{10}(\max T), 12)
$$

Bin centers computed as geometric mean: $c_j = \sqrt{b_j \cdot b_{j+1}}$

---

## Experimental Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `common_word` | "apple" | Repeated base word |
| `modified_word` | "apples" | Unique variant word |
| `model_max_output_tokens` | 32,768 | Provider output limit |
| `temperature` | 0 | Deterministic output |

### Sample Sizes

- **Full experiment**: 12 lengths × ~100 positions ≈ **~1,200 samples**
- **Test mode**: 5 lengths × 3 positions = **15 samples**

---

## Statistical Considerations

### Position Effects

The **U-shaped curve** hypothesis: Models may exhibit:
- Better performance at sequence boundaries (primacy/recency effects)
- Degraded performance in the middle of long sequences

### Length Effects

Performance degradation expected as:

$$
\text{Accuracy}(n) \propto f(n), \quad \frac{df}{dn} < 0
$$

---

## Implementation Notes

### High Output Token Requirement

This experiment requires significant output token allocation:
- A 10,000-word sequence requires ~20,000 output tokens
- Runtime is dominated by generation time, not prompt processing

### Checkpoint Resume

- Results saved incrementally (`save_every=10`)
- Existing outputs with valid content are skipped on resume
- Rows with `ERROR_*` prefix are retried

---

## References

- Hong, K., Troynikov, A., & Huber, J. (2025). *Context Rot: How Increasing Input Tokens Impacts LLM Performance*. Chroma Research.

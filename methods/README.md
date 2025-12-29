# Methods Documentation

This directory contains detailed methodology specifications for each experiment in the Context Rot research toolkit. Documentation follows academic methods section conventions with formal notation.

---

## Experiments

| Experiment | Description | Primary Metric |
|------------|-------------|----------------|
| [NIAH Extension](niah_extension.md) | Needle retrieval across varying context lengths and depths | Accuracy heatmap |
| [LongMemEval](longmemeval.md) | Retrieval+reasoning vs reasoning-only comparison | Accuracy delta |
| [Repeated Words](repeated_words.md) | Exact sequence replication with position tracking | Levenshtein score |

---

## Common Infrastructure

### LLM Judge

All experiments use GPT-4.1 (`gpt-4.1-2025-04-14`) as an automated judge for binary correctness classification. This provides consistent, scalable evaluation across thousands of samples.

### Tokenization

All token counting uses OpenAI's `tiktoken` library with the `o200k_base` encoding, ensuring consistency with GPT-4 family tokenization.

### Rate Limiting

Experiments implement token-aware batching:

$$
\text{batch}_i = \{r_j : \sum_{r \in \text{batch}_i} T_r \leq \text{max\_tokens\_per\_minute}\}
$$

with 60-second cooldowns between batches.

---

## Notation Conventions

| Symbol | Meaning |
|--------|---------|
| $L$ | Context length (tokens) |
| $d$ | Needle depth (percentage) |
| $n$ | Sequence length (words) |
| $k$ | Position index |
| $\mathbb{1}[\cdot]$ | Indicator function |
| $\text{Lev}(\cdot, \cdot)$ | Levenshtein edit distance |

---

## Citation

```bibtex
@techreport{hong2025contextrot,
  title={Context Rot: How Increasing Input Tokens Impacts LLM Performance},
  author={Hong, Kelly and Troynikov, Anton and Huber, Jeff},
  institution={Chroma},
  year={2025},
  month={July},
  url={https://research.trychroma.com/context-rot}
}
```

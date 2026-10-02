# Benchmark: zero-shot text classification (Dutch + English)

Measured on 2026-10-02 with `bench.py`, on CPU (Apple Silicon Mac), PyTorch, single-label.

## Test set

- 45 Dutch + 15 English customer messages, 5 labels: `cancel`, `refund`, `question`, `complaint`, `praise`
  (Dutch label set: `annuleren`, `terugbetaling`, `vraag`, `klacht`, `compliment`).
- Polite requests phrased as a question ("Kunt u mijn abonnement opzeggen") count as `cancel`, not `question`.
- Every sentence is tested 3 times: without punctuation, with a trailing "." and with a trailing "?". That makes 855
  predictions per model.
- Hand-made labels, small, and not reviewed by anyone else: a rough guide, not proof. Test on your own data.
- `ms/call` is indicative; some runs shared the CPU with other work.

## Results

| model | NL, EN labels | NL, NL labels | EN | no punctuation | with ? | flip by ? | non-questions -> question | ms/call |
|---|---|---|---|---|---|---|---|---|
| **deberta-zeroshot** | 92% | 95% | 96% | 97% | 88% | 12% | 1% -> 14% | 210 |
| **jevk5-lite** | 88% | 94% | 93% | 97% | 81% | 16% | 3% -> 22% | 159 |
| **gliclass-large-v3** | 86% | 92% | 76% | 89% | 82% | 8% | 11% -> 20% | 173 |
| gliner2-decide | 82% | 86% | 98% | 85% | 87% | 6% | 11% -> 13% | 165 |
| gliner2-multi-decide | 80% | 80% | 96% | 89% | 68% | 34% | 0% -> 38% | 53 |
| laya-multilingual | 70% | 83% | 84% | 82% | 68% | 23% | 5% -> 33% | 26 |
| gliclass-x-base | 69% | 78% | 78% | 72% | 79% | 22% | 0% -> 3% | 55 |
| julia-1 | 65% | 39% | 87% | 57% | 58% | 11% | 9% -> 8% | 26 |
| laya-en | 65% | 48% | 98% | 67% | 54% | 35% | 11% -> 47% | 37 |
| bart-mnli | 47% | 48% | 58% | 57% | 36% | 42% | 20% -> 62% | 149 |
| gliner2-decide-1b | 40% | 42% | 84% | 51% | 43% | 56% | 7% -> 57% | 166 |

Bold = the three models kept in `serve.py`. The others were removed from the code.

Columns:
- **NL, EN labels / NL labels**: accuracy on the Dutch sentences with English resp. Dutch labels.
- **EN**: accuracy on the English sentences (English labels).
- **no punctuation / with ?**: accuracy per variant, over all sentences.
- **flip by ?**: share of sentences whose prediction changes when a "?" is added.
- **non-questions -> question**: share of non-question sentences that get labelled `question`, without resp. with a "?".

## Takeaways

- `deberta-zeroshot` is the most accurate on Dutch (92-95%) and the least sensitive to punctuation. It is also the
  slowest, and it runs one pass per label, so it slows down with many labels.
- `jevk5-lite` is close behind (88-94%) and a bit faster.
- `gliclass-large-v3` is the middle ground: 86-92% on Dutch and the smallest "?" flip of the three, but weaker on
  English (76%).
- `gliner2-multi-decide` drops from 89% to 68% when a "?" is added, and tips non-questions over to `question` in 38%
  of cases. That was the reason for this comparison.
- Dutch labels beat English labels on Dutch text for nearly every model.
- Not tested: the hosted JEV from TypeSafe, Kev (English only, needs its own server) and Certo (a research preview
  trained on synthetic data).

## Models

| name | Hugging Face id |
|---|---|
| deberta-zeroshot | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` |
| jevk5-lite | `alibiserikbay/JevK5-Lite` |
| gliclass-large-v3 | `knowledgator/gliclass-large-v3.0` |
| gliner2-decide / -multi-decide / -decide-1b | `fastino/GLiNER2.5-Decide`, `-multi-Decide`, `-Decide-1B` |
| gliclass-x-base | `knowledgator/gliclass-x-base` |
| laya-multilingual / laya-en | `convaiinnovations/laya-multilingual`, `convaiinnovations/laya` |
| julia-1 | `SupersonicLabs/Julia-1` (on transformers 5.x without Julia's own ModernBERT speed path) |
| bart-mnli | `facebook/bart-large-mnli` |

## Re-running

```bash
python bench.py                      # every model in serve.MODELS (currently the 3 kept ones)
python bench.py deberta-zeroshot     # a single model
```

Per-prediction rows are written to `bench_rows.json`. The backend code for the removed models is no longer in
`serve.py`; restore it to measure them again.

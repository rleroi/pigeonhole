<p align="center">
  <img src="assets/logo.png" alt="pigeonhole logo" width="180">
</p>

<h1 align="center">pigeonhole</h1>

**Put any text in the right box.** A tiny self-hosted REST API for zero-shot text classification: intent, routing,
sentiment, triage. No training, no prompts, no per-token bills. Send text and labels, get probabilities back.

```bash
curl -X POST localhost:8000/classify -H 'Content-Type: application/json' -d '{
  "text": "My package arrived damaged and nobody answers my emails",
  "labels": ["cancel", "refund", "question", "complaint", "praise"]
}'
```

```json
{"model": "deberta-zeroshot", "task": "intent", "label": "complaint", "score": 0.98,
 "results": [{"label": "complaint", "score": 0.978}],
 "scores": {"complaint": 0.978, "question": 0.011, "cancel": 0.006, "refund": 0.005, "praise": 0.0}}
```

- **Zero-shot:** change the labels per request, nothing to retrain.
- **Swap models per request:** three open-weight classifiers behind one endpoint (`"model": "jevk5-lite"`).
- **Runs on CPU.** Typically 150-250 ms per request on a laptop. Everything stays on your machine.
- **Works on Dutch too**, and is benchmarked on Dutch and English, including how much a trailing `?` changes the answer. See
  [BENCHMARK.md](BENCHMARK.md).

## Quick start

### Docker

```bash
git clone https://github.com/rleroi/pigeonhole.git
cd pigeonhole
docker compose up -d --build        # http://localhost:8000
```

Models are downloaded on first use and kept in the `models` volume. The image uses CPU-only PyTorch (about 1.5 GB).
With two models loaded (`deberta-zeroshot` and `jevk5-lite`) the container used about 1 GB of RAM (`docker stats`,
measured idle after a few requests) and the model cache was about 2.5 GB on disk.

### Python

Developed and tested on Python 3.14.

```bash
git clone https://github.com/rleroi/pigeonhole.git
cd pigeonhole
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn serve:app --port 8000
```

The first start downloads the default model from Hugging Face. Other models load lazily on their first request and
then stay in memory. Interactive docs are at http://localhost:8000/docs.

Pick a different default model: `DEFAULT_MODEL=gliclass-large-v3 uvicorn serve:app --port 8000`
(with Docker: set `DEFAULT_MODEL` in `compose.yaml`).

## API

### `GET /models`

Lists available models, the default, and which are already loaded.

### `POST /classify`

| field         | type      | default            | description |
|---------------|-----------|--------------------|-------------|
| `text`        | string    | required           | text to classify |
| `labels`      | string[]  | required           | candidate labels. Descriptive words beat codes (`"cancel order"` > `"c1"`). Labels in the text's own language work well |
| `task`        | string    | `"intent"`         | name of the task. Context for the model, not an instruction |
| `model`       | string    | `deberta-zeroshot` | see [Models](#models) |
| `multi_label` | bool      | `false`            | allow several labels at once |
| `threshold`   | float 0-1 | `0.5`              | minimum score when `multi_label` is on |
| `top_k`       | int       | all                | with `multi_label`: cap the number of labels in `results` |

Response fields:

| field     | meaning |
|-----------|---------|
| `results` | the prediction. Without `multi_label`: exactly one label. With `multi_label`: every label with a score >= `threshold` (can be empty), best first |
| `label`, `score` | only without `multi_label`: the single winner (same as `results[0]`). Left out with `multi_label`, where there is no single answer: use `results` |
| `scores`  | the full distribution: every label with its score. Without `multi_label` they sum to 1; with `multi_label` each label is scored independently |

Unknown model names return `400`; a model whose package is not
installed returns `501`.

### Examples

Another model:

```bash
curl -X POST localhost:8000/classify -H 'Content-Type: application/json' \
  -d '{"text": "Where is my refund?", "labels": ["cancel", "refund", "question"], "model": "jevk5-lite"}'
```

Several labels at once:

```bash
curl -X POST localhost:8000/classify -H 'Content-Type: application/json' -d '{
  "text": "My package arrived broken and I want my money back",
  "task": "topic",
  "labels": ["damage", "refund", "delivery", "billing"],
  "multi_label": true,
  "threshold": 0.4
}'
```

Python:

```python
import requests

r = requests.post("http://localhost:8000/classify", json={
    "text": "I want to cancel my order",
    "labels": ["cancel", "refund", "question"],
})
print(r.json()["label"])  # cancel
```

JavaScript:

```js
const res = await fetch("http://localhost:8000/classify", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "I want to cancel my order", labels: ["cancel", "refund", "question"] }),
});
const { label, score, results } = await res.json();
```

PHP:

```php
$ch = curl_init('http://localhost:8000/classify');
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
    CURLOPT_POSTFIELDS => json_encode(['text' => 'I want to cancel my order', 'labels' => ['cancel', 'refund', 'question']]),
    CURLOPT_RETURNTRANSFER => true,
]);
$label = json_decode(curl_exec($ch), true)['label'];
```

## Models

| name                | backend     | Hugging Face id                               | NL accuracy* | ms/call* |
|---------------------|-------------|-----------------------------------------------|--------------|----------|
| `deberta-zeroshot`  | hf-zeroshot | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | 92-95%       | 210      |
| `jevk5-lite`        | jevk5       | `alibiserikbay/JevK5-Lite`                    | 88-94%       | 159      |
| `gliclass-large-v3` | gliclass    | `knowledgator/gliclass-large-v3.0`            | 86-92%       | 173      |

\* Small hand-made test set, see [BENCHMARK.md](BENCHMARK.md). `deberta-zeroshot` is the default: the most accurate
and the least sensitive to punctuation, but also the slowest, and it runs one pass per label.

Each model keeps its own license; check the Hugging Face pages before commercial use.

### Adding a model

Add one line to `MODELS` in `serve.py` using an existing backend, or write a new backend function and hook it into
`run_backend`. Then measure it with `python bench.py <name>`.

## Benchmark

```bash
python bench.py                    # every model in serve.MODELS
python bench.py deberta-zeroshot   # a single model
```

Results and methodology: [BENCHMARK.md](BENCHMARK.md).

## Good to know

- Every model fills `scores` with all labels in the same shape. `results` is the actual prediction (see
  [API](#post-classify)).
- Scores are still not comparable between models: each one is calibrated differently.
- Before routing on a threshold, check that the scores are calibrated on your own data (does 0.8 mean right about
  80% of the time?).
- CPU only. For GPU, change `device` in `serve.py`.
- No authentication. Put a reverse proxy in front of it, or bind to localhost, if it is not internal.

## License

[MIT](LICENSE) for the code in this repo. Model weights come with their own licenses.

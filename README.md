<p align="center">
  <img src="assets/logo.png" alt="pigeonhole logo" width="180">
</p>

<h1 align="center">pigeonhole</h1>

**Put any text in the right box.** A tiny self-hosted REST API for zero-shot text classification: intent, routing,
sentiment, triage. No training, no prompts, no per-token bills. Send text and labels, get probabilities back.

```bash
curl -X POST localhost:8000/classify -H 'Content-Type: application/json' -d '{
  "text": "Ik wil mijn bestelling annuleren",
  "labels": ["annuleren", "terugbetaling", "vraag", "klacht", "compliment"]
}'
```

```json
{"model": "deberta-zeroshot", "task": "intent", "label": "annuleren", "score": 0.99,
 "results": [{"label": "annuleren", "score": 0.99}, {"label": "vraag", "score": 0.01}, ...]}
```

- **Zero-shot:** change the labels per request, nothing to retrain.
- **Swap models per request:** three open-weight classifiers behind one endpoint (`"model": "jevk5-lite"`).
- **Runs on CPU.** Typically 150-250 ms per request on a laptop. Everything stays on your machine.
- **Benchmarked on Dutch and English**, including how much a trailing `?` changes the answer. See
  [BENCHMARK.md](BENCHMARK.md).

## Quick start

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
| `top_k`       | int       | all                | only return the k best labels in `results` |

The response has the winner as top-level `label` and `score` (`null` if nothing passes the threshold), plus `results`
with the labels and scores, sorted by `score`, highest first. Without `multi_label`, most models return every label
and the scores sum to 1; use `label`, or `top_k: 1`, if you only want the winner. Unknown model names return `400`; a model whose package is not
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

- Scores are not comparable between models. `deberta-zeroshot` and `jevk5-lite` return every label, and with
  `multi_label=false` the scores sum to 1. `gliclass-large-v3` returns only the winning label when
  `multi_label=false`.
- Before routing on a threshold, check that the scores are calibrated on your own data (does 0.8 mean right about
  80% of the time?).
- CPU only. For GPU, change `device` in `serve.py`.
- No authentication. Put a reverse proxy in front of it, or bind to localhost, if it is not internal.

## License

[MIT](LICENSE) for the code in this repo. Model weights come with their own licenses.

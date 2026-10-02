import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# name -> backend + HF id. Add a line here to expose a new model.
MODELS = {
    "deberta-zeroshot": {"backend": "hf-zeroshot", "id": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"},
    "jevk5-lite": {"backend": "jevk5", "id": "alibiserikbay/JevK5-Lite"},
    "gliclass-large-v3": {"backend": "gliclass", "id": "knowledgator/gliclass-large-v3.0"},
}
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "deberta-zeroshot")

_loaded: dict = {}
_load_lock = threading.Lock()


def _load(name: str):
    spec = MODELS[name]
    if spec["backend"] == "gliclass":
        from gliclass import GLiClassModel
        from transformers import AutoTokenizer

        return GLiClassModel.from_pretrained(spec["id"]), AutoTokenizer.from_pretrained(spec["id"])
    if spec["backend"] == "jevk5":
        from jevk5 import JevK5Lite

        return JevK5Lite.from_pretrained(spec["id"])
    from transformers import pipeline

    return pipeline("zero-shot-classification", model=spec["id"])


def get_model(name: str):
    if name not in _loaded:
        with _load_lock:
            if name not in _loaded:
                _loaded[name] = _load(name)
    return _loaded[name]


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_model(DEFAULT_MODEL)
    yield


app = FastAPI(title="pigeonhole", description="Zero-shot text classification API", lifespan=lifespan)


class ClassifyRequest(BaseModel):
    text: str
    task: str = "intent"
    labels: list[str]
    model: str = DEFAULT_MODEL
    multi_label: bool = False
    threshold: float = Field(0.5, ge=0, le=1)
    top_k: int | None = Field(None, ge=1, description="multi_label only: cap the number of returned labels")


def _hf_zeroshot(pipe, req: ClassifyRequest):
    out = pipe(req.text, req.labels, multi_label=req.multi_label)
    return [{"label": l, "score": s} for l, s in zip(out["labels"], out["scores"])]


def _gliclass(bundle, req: ClassifyRequest):
    from gliclass import ZeroShotClassificationPipeline

    model, tokenizer = bundle
    pipe = ZeroShotClassificationPipeline(
        model,
        tokenizer,
        classification_type="multi-label" if req.multi_label else "single-label",
        device="cpu",
    )
    # return_hierarchical -> {label: score} for every label (softmax if single-label, sigmoid if multi-label)
    scores = pipe(req.text, req.labels, threshold=0.0, return_hierarchical=True)[0]
    return [{"label": l, "score": s} for l, s in scores.items()]


def _jevk5(model, req: ClassifyRequest):
    head = {"labels": req.labels, "multi_label": True} if req.multi_label else req.labels
    out = model.classify(req.text, {req.task: head})[req.task]
    return [{"label": l, "score": p} for l, p in out["probabilities"].items()]


def run_backend(name: str, model, req: ClassifyRequest):
    """Run one classification; returns [{label, score}] sorted by score desc."""
    backend = MODELS[name]["backend"]
    if backend == "gliclass":
        results = _gliclass(model, req)
    elif backend == "jevk5":
        results = _jevk5(model, req)
    else:
        results = _hf_zeroshot(model, req)
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


@app.get("/models")
def models():
    return {"default": DEFAULT_MODEL, "models": MODELS, "loaded": list(_loaded)}


@app.post("/classify")
def classify(req: ClassifyRequest):
    if req.model not in MODELS:
        raise HTTPException(400, f"unknown model '{req.model}', see GET /models")
    backend = MODELS[req.model]["backend"]

    try:
        model = get_model(req.model)
    except ImportError as e:
        raise HTTPException(501, f"backend '{backend}' not installed: {e}")

    ranked = run_backend(req.model, model, req)  # every label, best first
    if req.multi_label:
        results = [r for r in ranked if r["score"] >= req.threshold]
    else:
        results = ranked[:1]
    if req.top_k:
        results = results[: req.top_k]
    response = {"model": req.model, "task": req.task}
    if not req.multi_label:
        response["label"] = results[0]["label"]
        response["score"] = results[0]["score"]
    response["results"] = results
    response["scores"] = {r["label"]: r["score"] for r in ranked}
    return response

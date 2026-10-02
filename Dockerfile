FROM python:3.14-slim

# git: jevk5 is installed straight from GitHub (see requirements.txt)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home app
WORKDIR /app

# CPU-only torch wheels (the default Linux wheel pulls in CUDA and is several GB)
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY serve.py bench.py ./

# Models are downloaded on first use; mount a volume on /models to keep them
ENV HF_HOME=/models
RUN mkdir /models && chown app /models
VOLUME /models

USER app
EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

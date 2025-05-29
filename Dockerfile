FROM python:3.12-slim AS builder

ARG cpu_only=0

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml .
COPY ./nlpimdbmoviereviews ./nlpimdbmoviereviews

RUN if [ "$cpu_only" = 1 ]; then \
        pip install --no-cache-dir --no-compile . --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir --no-compile .; \
    fi

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY ./output/model.pkl ./output/
COPY ./scripts/score.py .
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

CMD ["python", "score.py"]

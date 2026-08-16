# Serving image for the judge. Deliberately does NOT carry torch/tensorflow — the judge
# only needs pandas, sklearn and the web stack. The heavy ML deps belong in the notebook
# environment, not on the server.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    JUDGE_ENV=production

WORKDIR /app

COPY judge/requirements-judge.txt /app/judge/requirements-judge.txt
RUN pip install --no-cache-dir -r judge/requirements-judge.txt

# adslab is imported by the scorer; the week folders supply study material. .dockerignore
# keeps the 56 MB of PDFs out — only the READMEs and paper indexes are needed.
COPY adslab/ /app/adslab/
COPY judge/ /app/judge/
COPY week01_foundations/ /app/week01_foundations/
COPY docs/ /app/docs/

# The volume mounts over this; it exists so the image runs without one for a smoke test.
RUN mkdir -p /app/judge/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=4)"

# Single worker on purpose: the rate limiter holds state in-process and SQLite prefers
# one writer. Scale by making the machine bigger, or move both before scaling out.
CMD ["uvicorn", "judge.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

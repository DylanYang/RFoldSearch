# Databases are mounted, not baked in: a half-million-chain .pt is several
# GB and changes independently of the code.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ git && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# US-align, only needed when --rerank tm is offered
RUN git clone --depth 1 https://github.com/pylelab/USalign /tmp/USalign \
 && g++ -O3 -o /usr/local/bin/USalign /tmp/USalign/USalign.cpp \
 && rm -rf /tmp/USalign
ENV USALIGN=/usr/local/bin/USalign

COPY app ./app
COPY static ./static

ENV RFOLDSEARCH_DBS="" \
    RFOLDSEARCH_MAX_CONCURRENT=2 \
    RFOLDSEARCH_MAX_UPLOAD_MB=32 \
    RFOLDSEARCH_TOP_N_CAP=2000
EXPOSE 8000

# One worker process: each would hold its own copy of the databases.
# Concurrency comes from RFOLDSEARCH_MAX_CONCURRENT threads inside it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "75"]

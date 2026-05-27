# --- stage 1: build deps ---
# slim = stripped down python, no extra garbage
FROM python:3.11-slim AS builder

WORKDIR /app

# install deps in a separate layer so docker caches them
# only reruns if requirements.txt changes
COPY requirements.txt .

# no cache dir = smaller image
# no-compile = skip .pyc at install time
RUN pip install --no-cache-dir --no-compile --upgrade pip \
    && pip install --no-cache-dir --no-compile -r requirements.txt


# --- stage 2: final image ---
FROM python:3.11-slim AS final

# dont run as root, thats a security thing
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# copy only the installed packages from builder stage
# this keeps the final image lean
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# copy app source
COPY app/ ./app/

# model and logs folders need to exist at runtime
RUN mkdir -p app/model logs && chown -R appuser:appgroup /app

USER appuser

# tell docker this port is used
EXPOSE 8000

# healthcheck so docker knows if the app crashed
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

# start the api
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

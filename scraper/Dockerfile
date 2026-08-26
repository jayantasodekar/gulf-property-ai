# Separate image for the offline scrape job.
#
# Kept out of the runtime image on purpose: the scraper needs curl_cffi's
# browser-impersonation stack, which the serving app has no use for. The job
# writes data/corpus.jsonl.gz, which is then committed and baked into the app
# image at build time.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt requirements-scraper.txt ./
RUN pip install --no-cache-dir -r requirements-scraper.txt

COPY scraper/ ./scraper/
COPY app/ ./app/

RUN useradd --create-home --uid 1000 scraperuser && chown -R scraperuser /app
USER scraperuser

CMD ["python", "-m", "scraper.run", "--source", "both", "--limit", "3000"]

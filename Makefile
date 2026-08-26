.PHONY: install scrape index run test lint eval docker docker-run clean

PY := ./.venv/Scripts/python.exe

install:
	python -m venv .venv
	$(PY) -m pip install -r requirements-scraper.txt -r requirements-dev.txt
	cd web && npm install

scrape:
	$(PY) -m scraper.run --source both --limit 3000

stats:
	$(PY) -m scraper.run --stats-only

index:
	$(PY) -m app.index --build

verify:
	$(PY) -m app.index --verify

web:
	cd web && npm run build

run:
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check app scraper tests

eval:
	$(PY) -m eval.run

docker:
	docker build -t gulf-property-ai:latest .

docker-run:
	docker run --rm -p 7860:7860 --env-file .env gulf-property-ai:latest

clean:
	rm -rf data/corpus.sqlite data/corpus.sqlite-* web/dist .pytest_cache .ruff_cache

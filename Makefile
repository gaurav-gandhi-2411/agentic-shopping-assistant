BRAND ?= snitch
PORT  ?= 8080

.PHONY: install test lint backend frontend demo

## Install all Python and Node dependencies.
install:
	pip install -r requirements.txt
	cd frontend && npm install

## Run the test suite (no Ollama or pre-built index required).
test:
	pytest -m "not requires_ollama"

## Lint with ruff.
lint:
	ruff check . && ruff format --check .

## Start the backend API for a given brand.
##   Usage: make backend BRAND=snitch
backend:
	JWT_VERIFICATION_DISABLED=true BRAND=$(BRAND) uvicorn api.main:app --reload --port $(PORT)

## Start the Next.js frontend (run alongside 'make backend' in a second terminal).
frontend:
	cd frontend && npm run dev

## Download catalogue + build index + start backend for a given brand.
## Run 'make frontend' in a second terminal to open the UI.
##   Usage: make demo BRAND=snitch
demo:
	BRAND=$(BRAND) PORT=$(PORT) bash scripts/quickstart.sh

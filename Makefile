.PHONY: setup test lint validate
setup:
	uv sync --extra dev
validate:
	uv run monster validate-config
test:
	uv run pytest -q
lint:
	uv run ruff check src tests

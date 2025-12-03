.PHONY: lint format install

lint:
	poetry run black --check .
	poetry run isort --check-only .
	poetry run pautoflake --check --recursive .
	poetry run mypy .

format:
	poetry run pautoflake --in-place --remove-all-unused-imports --recursive .
	poetry run isort .
	poetry run black .

install:
	poetry install


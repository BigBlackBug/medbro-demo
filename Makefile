.PHONY: lint format install

FILES ?= .

lint:
	poetry run black --check $(FILES)
	poetry run isort --check-only $(FILES)
	poetry run pautoflake --check --recursive $(FILES)
	poetry run mypy $(FILES)

format:
	poetry run pautoflake --in-place --remove-all-unused-imports --recursive $(FILES)
	poetry run isort $(FILES)
	poetry run black $(FILES)

install:
	poetry install


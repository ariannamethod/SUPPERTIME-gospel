.PHONY: format lint

format:
	isort .
	black .

lint:
	isort --check-only .
	black --check .
	flake8

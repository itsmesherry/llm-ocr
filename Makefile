.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make qa            - Run code quality checks (isort, ruff format, ruff check)"
	@echo "  make sort          - Sort imports with isort"
	@echo "  make format        - Format code with ruff"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make mypy          - Type check with mypy"
	@echo "  make test          - Run tests"
	@echo "  make coverage      - Run tests with coverage report"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make build         - Build package for PyPI"
	@echo "  make check-build   - Check built package"
	@echo "  make publish-test  - Publish to TestPyPI"
	@echo "  make publish       - Publish to PyPI"

.PHONY: sort
sort: 
	isort .

.PHONY: format
format: 
	ruff format

.PHONY: lint
lint:
	ruff check

.PHONY: mypy
mypy: 
	mypy . --check-untyped-defs

.PHONY: test
test:
	pytest tests/ -v

.PHONY: coverage
coverage:
	pytest tests/ --cov=llm_ocr --cov-report=html --cov-report=term

.PHONY: qa
qa: sort format lint

.PHONY: clean
clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .coverage htmlcov/ .ruff_cache .mypy_cache

.PHONY: build
build: clean
	python -m build

.PHONY: check-build
check-build:
	python -m twine check dist/*

.PHONY: publish-test
publish-test: build check-build
	python -m twine upload --repository testpypi dist/*

.PHONY: publish
publish: build check-build
	python -m twine upload dist/*
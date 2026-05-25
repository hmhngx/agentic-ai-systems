.PHONY: install test test-fast test-cov lint typecheck benchmark clean

install:
	@echo "========== install =========="
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

test:
	@echo "========== test =========="
	# Run full test suite with verbose output
	python -m pytest tests/ -v --tb=short

test-fast:
	@echo "========== test-fast =========="
	# Run all tests except the slow semantic_chunk tests (marked slow)
	python -m pytest tests/ -v --tb=short -m "not slow"

test-cov:
	@echo "========== test-cov =========="
	# Run tests with line coverage — must be ≥80% or this target fails
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80

lint:
	@echo "========== lint =========="
	# Run pyflakes on all source files — zero warnings allowed
	python -m pyflakes src/ benchmark.py

typecheck:
	@echo "========== typecheck =========="
	# Run pyright or mypy in strict mode
	# Try pyright first, fall back to mypy
	python -m pyright src/ benchmark.py 2>/dev/null || python -m mypy src/ benchmark.py --strict

benchmark:
	@echo "========== benchmark =========="
	# Run the full benchmark CLI on sample_doc.pdf
	# Requires sample_doc.pdf to exist in 02-chunking/
	@test -f sample_doc.pdf || (echo "ERROR: sample_doc.pdf not found. Place a PDF here first." && exit 1)
	python benchmark.py --pdf_path sample_doc.pdf

clean:
	@echo "========== clean =========="
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	find . -name "test_output.txt" -delete 2>/dev/null || true
	find . -name "benchmark_output.txt" -delete 2>/dev/null || true

# inductor-rs build automation
#
# Usage:
#   make                              - Build release binary (sets up venv if needed)
#   make target/debug/inductor-rs     - Build debug binary
#   make target/portable/release/...  - Build portable binary with copied libraries
#   make clean                        - Remove build artifacts and venv
#
# Options:
#   FEATURES=cuda                     - Enable cargo features (e.g., cuda, rocm)

# Optional: pass cargo features (e.g., FEATURES=cuda)
CARGO_FEATURES := $(if $(FEATURES),--features $(FEATURES),)

# Default target builds the release binary
build: .venv/.installed
	cd core && cargo build --release $(CARGO_FEATURES)

# Shorthand for build
all: build

# Debug build
build-debug: .venv/.installed
	cd core && cargo build $(CARGO_FEATURES)

# CUDA build
build-cuda:
	$(MAKE) build FEATURES=cuda

# ROCm build
build-rocm:
	$(MAKE) build FEATURES=rocm

# Portable build copies libraries instead of symlinking
build-portable: .venv/.installed
	cd core && AOT_PORTABLE=1 cargo build --release --target-dir ../target/portable $(CARGO_FEATURES)

# Create venv and install deps from uv.lock
# Touch .installed marker to track when deps were last synced
.venv/.installed: pyproject.toml
	uv sync
	@touch .venv/.installed

# Initialize with training dependencies
init-training:
	uv sync --group training --group dev

# Initialize training environment with ROCm PyTorch (replaces CUDA torch)
# Note: uv run/sync will revert to CUDA; use .venv/bin/python directly after this
init-training-rocm: init-training
	@echo "Replacing CUDA PyTorch with ROCm PyTorch..."
	uv pip install pip
	.venv/bin/pip uninstall -y torch torchvision || true
	.venv/bin/pip install torch torchvision \
		--index-url https://download.pytorch.org/whl/rocm7.0
	@echo "ROCm PyTorch installed. Verify with: .venv/bin/python -c \"import torch; print(torch.version.hip)\""

# Export a model to AOT format
# Usage: make export MODEL_ARGS="--output model.pt2 --device cpu"
export: .venv/.installed
	TORCH_COMPILE_DISABLE=1 uv run python python/scripts/export_aot.py $(MODEL_ARGS)

# Clean build artifacts
clean:
	cargo clean
	rm -rf .venv target

.PHONY: build all build-debug build-cuda build-rocm build-portable clean
.PHONY: init-training init-training-rocm export

# ============================================================
# Rust Code Quality
# ============================================================

fmt:
	cd core && cargo fmt

fmt-check:
	cd core && cargo fmt --check

clippy:
	cd core && cargo clippy --release $(CARGO_FEATURES) -- -D warnings

.PHONY: fmt fmt-check clippy

# ============================================================
# C++ Code Quality
# ============================================================

fmt-cpp:
	clang-format -i aot_bridge/src/*.cpp aot_bridge/include/*.h

fmt-cpp-check:
	clang-format --dry-run --Werror aot_bridge/src/*.cpp aot_bridge/include/*.h

.PHONY: fmt-cpp fmt-cpp-check

# ============================================================
# Python Code Quality
# ============================================================

quality:
	uv run ruff check python/

style:
	uv run ruff check --fix python/
	uv run ruff format python/

types:
	uv run basedpyright

.PHONY: quality style types

# ============================================================
# Unified Code Quality
# ============================================================

# Run all code quality checks (Rust, Python, C++)
lint: fmt-check clippy quality types fmt-cpp-check
	@echo "All checks passed!"

# Format all code (Rust, Python, C++)
fmt-all: fmt style fmt-cpp
	@echo "All code formatted!"

.PHONY: lint fmt-all

# ============================================================
# Testing
# ============================================================

# Generate test fixture (requires Python environment with PyTorch)
test-fixtures: .venv/.installed
	uv run python python/scripts/generate_test_model.py \
		-o tests/fixtures/dummy_model.pt2 --verify

# Run Rust tests (requires test fixture)
test-rust: test-fixtures
	cd core && cargo test --release

# Run Python tests
test-python:
	uv run pytest

# Run all tests
test: test-python test-rust

.PHONY: test-fixtures test-rust test-python test

# ============================================================
# Documentation
# ============================================================

docs:
	cd core && cargo doc --open

.PHONY: docs

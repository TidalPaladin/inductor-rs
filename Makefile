# inductor-rs build automation
#
# Usage:
#   make                              - Build release binary (sets up venv if needed)
#   make build BACKEND=cuda           - Build with CUDA support
#   make build BACKEND=rocm           - Build with ROCm support
#   make build-debug BACKEND=cuda     - Build debug binary for CUDA
#   make build-portable BACKEND=rocm  - Build portable ROCm binary
#   make init BACKEND=cuda            - Initialize CUDA environment
#   make clean                        - Remove build artifacts and venvs
#
# Options:
#   BACKEND=cpu|cuda|rocm             - Select backend (default: cpu)

BACKEND ?= cpu
VALID_BACKENDS := cpu cuda rocm
ifeq ($(filter $(BACKEND),$(VALID_BACKENDS)),)
$(error BACKEND must be one of $(VALID_BACKENDS))
endif

VENV_DIR := $(if $(filter cpu,$(BACKEND)),.venv,.venv-$(BACKEND))
PYTHON := $(VENV_DIR)/bin/python
UV_ENV := UV_PROJECT_ENVIRONMENT=$(VENV_DIR)
BACKEND_GROUP := backend-$(BACKEND)

CARGO_FEATURES := $(if $(filter cuda,$(BACKEND)),--features cuda,$(if $(filter rocm,$(BACKEND)),--features rocm,))

# Default target builds the release binary
build: $(VENV_DIR)/.installed
	cd core && cargo build --release $(CARGO_FEATURES)

# Shorthand for build
all: build

# Debug build
build-debug: $(VENV_DIR)/.installed
	cd core && cargo build $(CARGO_FEATURES)

# Portable build copies libraries instead of symlinking
build-portable: $(VENV_DIR)/.installed
	cd core && AOT_PORTABLE=1 cargo build --release --target-dir ../target/portable $(CARGO_FEATURES)

# Create venv and install deps from uv.lock
# Touch .installed marker to track when deps were last synced
$(VENV_DIR)/.installed: pyproject.toml uv.lock
	$(UV_ENV) UV_INDEX_STRATEGY=unsafe-best-match uv sync --frozen --no-default-groups --group training --group dev --group export --group $(BACKEND_GROUP)
	@touch $(VENV_DIR)/.installed

# Initialize environment
init:
	$(MAKE) -B $(VENV_DIR)/.installed BACKEND=$(BACKEND)

# Export a model to AOT format
# Usage: make export MODEL_ARGS="--output model.pt2 --device cpu" BACKEND=cpu|cuda|rocm
export: $(VENV_DIR)/.installed
	TORCH_COMPILE_DISABLE=1 $(PYTHON) python/scripts/export_aot.py $(MODEL_ARGS)

# Clean build artifacts
clean:
	cargo clean
	rm -rf .venv .venv-cuda .venv-rocm target

.PHONY: build all build-debug build-portable clean
.PHONY: init export

# ============================================================
# Rust Code Quality
# ============================================================

fmt:
	cd core && cargo fmt

fmt-check:
	cd core && cargo fmt --check

clippy: $(VENV_DIR)/.installed
	cd core && AOT_BRIDGE_SKIP_BUILD=1 cargo clippy --release $(CARGO_FEATURES) -- -D warnings

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

quality: $(VENV_DIR)/.installed
	$(VENV_DIR)/bin/ruff check python/

style: $(VENV_DIR)/.installed
	$(VENV_DIR)/bin/ruff check --fix python/
	$(VENV_DIR)/bin/ruff format python/

types: $(VENV_DIR)/.installed
	$(VENV_DIR)/bin/basedpyright

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
test-fixtures: $(VENV_DIR)/.installed
	$(PYTHON) python/scripts/generate_test_model.py \
		-o tests/fixtures/dummy_model.pt2 --verify

# Run Rust tests (requires test fixture)
test-rust: test-fixtures
	cd core && cargo test --release $(CARGO_FEATURES)

# Run Python tests
test-python: $(VENV_DIR)/.installed
	$(PYTHON) -m pytest

# Run all tests
test: test-python test-rust

# Run backend smoke test (skips if no GPU available)
test-backend: $(VENV_DIR)/.installed
	@set -e; \
	if [ "$(BACKEND)" = "cpu" ]; then \
		echo "ERROR: BACKEND must be cuda or rocm for test-backend"; \
		exit 1; \
	fi; \
	rc=0; \
	device_index=$$(BACKEND=$(BACKEND) $(PYTHON) python/scripts/check_backend.py --print-index) || rc=$$?; \
	if [ $$rc -eq 2 ]; then exit 0; fi; \
	if [ $$rc -ne 0 ]; then exit $$rc; fi; \
	cd core && cargo build --release $(CARGO_FEATURES) && ../target/release/inductor-rs --check --device cuda:$$device_index

# Run CUDA and ROCm smoke tests
test-backends:
	@set -e; \
	for b in cuda rocm; do \
		echo "==> Testing $$b"; \
		$(MAKE) test-backend BACKEND=$$b || exit $$?; \
	done

.PHONY: test-fixtures test-rust test-python test test-backend test-backends

# Verify a portable build runs without host environment library paths.
verify-portable: build-portable
	@if [ ! -x "target/portable/release/inductor-rs" ]; then \
		echo "Error: portable binary missing at target/portable/release/inductor-rs"; \
		exit 1; \
	fi
	@if [ ! -d "target/portable/release/lib" ]; then \
		echo "Error: portable runtime libs missing at target/portable/release/lib"; \
		exit 1; \
	fi
	@tmp_dir=$$(mktemp -d); \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	cp "target/portable/release/inductor-rs" "$$tmp_dir/inductor-rs"; \
	cp -a "target/portable/release/lib" "$$tmp_dir/lib"; \
	env -u LD_LIBRARY_PATH "$$tmp_dir/inductor-rs" --help >/dev/null

.PHONY: verify-portable

# ============================================================
# Documentation
# ============================================================

docs: $(VENV_DIR)/.installed
	cd core && cargo doc --open

.PHONY: docs

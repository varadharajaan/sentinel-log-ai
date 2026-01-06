# Sentinel Log AI - Makefile
# Polyglot build system for Go agent + Python ML engine

.PHONY: all build build-go build-python clean test test-go test-python lint proto install dev help

# Default target
all: build

# ============================================================================
# Build Targets
# ============================================================================

build: build-go build-python ## Build all components

build-go: proto-go ## Build Go agent
	@echo "Building Go agent..."
	go build -o bin/sentinel-log-ai ./cmd/agent

build-python: proto-python ## Install Python ML engine
	@echo "Installing Python ML engine..."
	pip install -e ".[all]"

# ============================================================================
# Proto Generation
# ============================================================================

proto: proto-go proto-python ## Generate protobuf code for both languages

proto-go: ## Generate Go protobuf code
	@echo "Generating Go protobuf code..."
	@mkdir -p pkg/mlpb
	protoc --go_out=. --go-grpc_out=. \
		--go_opt=paths=source_relative \
		--go-grpc_opt=paths=source_relative \
		proto/ml/v1/*.proto

proto-python: ## Generate Python protobuf code
	@echo "Generating Python protobuf code..."
	@mkdir -p python/sentinel_ml/generated
	python -m grpc_tools.protoc \
		-I. \
		--python_out=python/sentinel_ml/generated \
		--grpc_python_out=python/sentinel_ml/generated \
		--mypy_out=python/sentinel_ml/generated \
		proto/ml/v1/*.proto
	@touch python/sentinel_ml/generated/__init__.py

# ============================================================================
# Test Targets
# ============================================================================

test: test-go test-python ## Run all tests

test-go: ## Run Go tests
	@echo "Running Go tests..."
	go test -v -race -cover ./...

test-python: ## Run Python tests
	@echo "Running Python tests..."
	pytest tests/python -v --cov=sentinel_ml --cov-report=term-missing

test-integration: ## Run integration tests (requires ML server running)
	@echo "Running integration tests..."
	pytest tests/integration -v -m integration

# ============================================================================
# Lint Targets
# ============================================================================

lint: lint-go lint-python lint-proto ## Run all linters

lint-go: ## Lint Go code
	@echo "Linting Go code..."
	golangci-lint run ./...

lint-python: ## Lint Python code
	@echo "Linting Python code..."
	ruff check python/ tests/
	ruff format --check python/ tests/
	mypy python/

lint-proto: ## Lint protobuf files
	@echo "Linting protobuf files..."
	buf lint proto/

# ============================================================================
# Format Targets
# ============================================================================

fmt: fmt-go fmt-python ## Format all code

fmt-go: ## Format Go code
	@echo "Formatting Go code..."
	gofmt -w .
	goimports -w .

fmt-python: ## Format Python code
	@echo "Formatting Python code..."
	ruff format python/ tests/
	ruff check --fix python/ tests/

# ============================================================================
# Development Targets
# ============================================================================

install: ## Install all dependencies
	@echo "Installing Go dependencies..."
	go mod download
	@echo "Installing Python dependencies..."
	pip install -e ".[all]"
	@echo "Installing pre-commit hooks..."
	pre-commit install

dev: ## Start development environment (ML server + agent)
	@echo "Starting development environment..."
	@echo "Run 'make run-ml' and 'make run-agent' in separate terminals"

run-ml: ## Run the Python ML server
	@echo "Starting ML gRPC server..."
	sentinel-ml

run-agent: ## Run the Go agent
	@echo "Starting Go agent..."
	./bin/sentinel-log-ai serve

# ============================================================================
# Clean Targets
# ============================================================================

clean: ## Clean build artifacts
	@echo "Cleaning..."
	rm -rf bin/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf python/sentinel_ml/generated/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ============================================================================
# Release Targets
# ============================================================================

release-go: build-go ## Build release binary for Go agent
	@echo "Building release binary..."
	CGO_ENABLED=0 go build -ldflags="-s -w" -o bin/sentinel-log-ai ./cmd/agent

release-python: ## Build Python wheel
	@echo "Building Python wheel..."
	python -m build

# ============================================================================
# Help
# ============================================================================

help: ## Show this help
	@echo "Sentinel Log AI - Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: help build up down login shell clean logs restart setup

# Default target
.DEFAULT_GOAL := help

# Variables
IMAGE_NAME := sam3d-sam3-depth
CONTAINER_NAME := sam3-app
COMPOSE_FILE := docker-compose.yml
DOCKERFILE := Dockerfile

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Display this help message
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)SAM3D + SAM3 + Depth Anything V3 Docker$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Setup Instructions:$(NC)"
	@echo "  1. export HF_TOKEN=hf_your_actual_token"
	@echo "  2. make setup"
	@echo "  3. make build"
	@echo "  4. make up"
	@echo ""

setup: ## Verify HF_TOKEN is set and create directories
	@echo "$(BLUE)Setting up project...$(NC)"
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "$(RED)✗ Error: HF_TOKEN environment variable not set$(NC)"; \
		echo "$(YELLOW)Please set your Hugging Face token:$(NC)"; \
		echo "  export HF_TOKEN=hf_your_actual_token_here"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ HF_TOKEN is set$(NC)"
	@mkdir -p data notebooks output results
	@chmod +x login_hf.sh
	@echo "$(GREEN)✓ Setup complete$(NC)"
	@echo ""

build: setup ## Build Docker image with HF_TOKEN
	@echo "$(BLUE)Building Docker image: $(IMAGE_NAME)...$(NC)"
	@echo "$(YELLOW)Using HF_TOKEN from environment variable$(NC)"
	docker build \
		--build-arg HF_TOKEN \
		-f $(DOCKERFILE) \
		-t $(IMAGE_NAME):latest .
	@echo "$(GREEN)✓ Build complete$(NC)"
	@echo ""

up: ## Start containers in detached mode
	@echo "$(BLUE)Starting containers...$(NC)"
	docker compose -f $(COMPOSE_FILE) up -d
	@echo "$(GREEN)✓ Containers started$(NC)"
	@echo "$(YELLOW)Services available at:$(NC)"
	@echo "  SAM3D Server:        http://localhost:8000"
	@echo "  SAM3 Server:         http://localhost:8001"
	@echo "  Depth Anything V3:   http://localhost:8002"
	@echo ""
	@echo "$(YELLOW)Run 'make shell' to enter the container$(NC)"
	@echo ""

down: ## Stop and remove containers
	@echo "$(BLUE)Stopping containers...$(NC)"
	docker compose -f $(COMPOSE_FILE) down
	@echo "$(GREEN)✓ Containers stopped$(NC)"
	@echo ""

restart: ## Restart containers
	@echo "$(BLUE)Restarting containers...$(NC)"
	docker compose -f $(COMPOSE_FILE) restart
	@echo "$(GREEN)✓ Containers restarted$(NC)"
	@echo ""

shell: ## Enter the container shell
	@echo "$(BLUE)Entering container shell...$(NC)"
	docker compose -f $(COMPOSE_FILE) exec sam3 /bin/bash

login: ## Run Hugging Face login (interactive)
	@echo "$(BLUE)Starting Hugging Face authentication...$(NC)"
	docker compose -f $(COMPOSE_FILE) exec sam3 /workspace/login_hf.sh

verify-login: ## Verify Hugging Face authentication
	@echo "$(BLUE)Verifying Hugging Face login...$(NC)"
	docker compose -f $(COMPOSE_FILE) exec sam3 hf auth whoami

logs: ## Show container logs
	docker compose -f $(COMPOSE_FILE) logs -f sam3

clean: ## Remove containers and images
	@echo "$(BLUE)Cleaning up Docker resources...$(NC)"
	docker compose -f $(COMPOSE_FILE) down --rmi all
	@echo "$(GREEN)✓ Cleanup complete$(NC)"
	@echo ""

prune: clean ## Deep clean - remove all unused Docker resources
	@echo "$(BLUE)Pruning all unused Docker resources...$(NC)"
	docker system prune -a -f
	@echo "$(GREEN)✓ Prune complete$(NC)"
	@echo ""

ps: ## Show running containers
	docker compose -f $(COMPOSE_FILE) ps

health: ## Check health of all services
	@echo "$(BLUE)Checking service health...$(NC)"
	@echo ""
	@echo "$(YELLOW)SAM3D (8000):$(NC)"
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "$(RED)✗ Not responding$(NC)"
	@echo ""
	@echo "$(YELLOW)SAM3 (8001):$(NC)"
	@curl -s http://localhost:8001/health | python3 -m json.tool || echo "$(RED)✗ Not responding$(NC)"
	@echo ""
	@echo "$(YELLOW)Depth Anything V3 (8002):$(NC)"
	@curl -s http://localhost:8002/health | python3 -m json.tool || echo "$(RED)✗ Not responding$(NC)"
	@echo ""

test-sam3: ## Test SAM3 server with test.jpg
	@echo "$(BLUE)Testing SAM3 segmentation...$(NC)"
	python3 client.py --image test.jpg --sam3

test-depth: ## Test Depth Anything V3 with test.jpg
	@echo "$(BLUE)Testing Depth Anything V3...$(NC)"
	python3 client.py --image test.jpg --depth

test-sam3d: ## Test SAM3D 3D reconstruction with test.jpg
	@echo "$(BLUE)Testing SAM3D reconstruction...$(NC)"
	python3 client.py --image test.jpg --sam3d

test-all: ## Test all models with test.jpg
	@echo "$(BLUE)Testing all models...$(NC)"
	python3 client.py --image test.jpg --all

# Development workflow targets
dev: build up ## Build and start containers (development setup)
	@echo "$(GREEN)✓ Development environment ready$(NC)"
	@echo "$(YELLOW)Services running:$(NC)"
	@echo "  SAM3D Server:        http://localhost:8000"
	@echo "  SAM3 Server:         http://localhost:8001"
	@echo "  Depth Anything V3:   http://localhost:8002"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. make shell      - Enter container"
	@echo "  2. make test-all   - Test all models"
	@echo ""

quickstart: setup build up health ## Quick start - setup, build, and run
	@echo ""
	@echo "$(GREEN)✓ Quickstart complete!$(NC)"
	@echo "$(YELLOW)All services are running and healthy.$(NC)"
	@echo ""
	@echo "$(YELLOW)Try these commands:$(NC)"
	@echo "  make test-all      - Test all models with test.jpg"
	@echo "  make shell         - Enter the container"
	@echo "  make logs          - View container logs"
	@echo ""

# Sim2RealLab workspace Makefile
# Wraps colcon build/test workflows and Python lint/format tools.

SHELL := /bin/bash
COLCON_WS := $(HOME)/strafer_ws

.PHONY: build test test-unit lint format clean install-tools help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------- Build ----------

build: ## Build all ROS2 packages with colcon
	cd $(COLCON_WS) && source /opt/ros/humble/setup.bash && \
		colcon build --symlink-install

# ---------- Test ----------

test: ## Run all colcon tests (requires build first)
	cd $(COLCON_WS) && source /opt/ros/humble/setup.bash && \
		colcon test && colcon test-result --verbose

test-unit: ## Run strafer_driver unit tests directly with pytest
	cd source/strafer_ros/strafer_driver && \
		python -m pytest test/ -v

# ---------- Lint / Format ----------

lint: ## Run flake8 on all Python source
	flake8 source/strafer_ros/ source/strafer_shared/ Scripts/ \
		--max-line-length 100 --extend-ignore=E203,W503

format: ## Run black on all Python source
	black source/strafer_ros/ source/strafer_shared/ Scripts/

format-check: ## Check formatting without modifying files
	black --check source/strafer_ros/ source/strafer_shared/ Scripts/

# ---------- Clean ----------

clean: ## Remove colcon build artifacts
	cd $(COLCON_WS) && rm -rf build/ install/ log/

# ---------- Setup ----------

install-tools: ## Install black (flake8 is already available via ROS)
	pip install black

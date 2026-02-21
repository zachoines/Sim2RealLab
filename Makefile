# Sim2RealLab workspace Makefile
# Wraps colcon build/test workflows and Python lint/format tools.

SHELL := /bin/bash
COLCON_WS := $(HOME)/strafer_ws

.PHONY: build test test-unit lint lint-fix format format-check clean install-tools udev help

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
	python3 -m flake8 source/strafer_ros/ source/strafer_shared/ Scripts/ \
		--max-line-length 100 --extend-ignore=E203,W503

lint-fix: ## Auto-fix lint issues with autopep8
	python3 -m autopep8 --in-place --recursive --max-line-length 100 \
		source/strafer_ros/ source/strafer_shared/ Scripts/

format: ## Run black on all Python source
	python3 -m black source/strafer_ros/ source/strafer_shared/ Scripts/

format-check: ## Check formatting without modifying files
	python3 -m black --check source/strafer_ros/ source/strafer_shared/ Scripts/

# ---------- Clean ----------

clean: ## Remove colcon build artifacts
	cd $(COLCON_WS) && rm -rf build/ install/ log/

# ---------- Setup ----------

install-tools: ## Install black, autopep8 (flake8 is already available via ROS)
	pip install black 'autopep8>=2,<3' --no-deps

udev: ## Install udev rules for RoboClaw symlinks (requires sudo)
	sudo cp source/strafer_ros/99-strafer.rules /etc/udev/rules.d/
	sudo udevadm control --reload-rules
	sudo udevadm trigger
	@echo "Verify with: ls -la /dev/roboclaw*"

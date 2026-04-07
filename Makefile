# Sim2RealLab workspace Makefile
# Wraps colcon build/test workflows and Python lint/format tools.

SHELL := /bin/bash
COLCON_WS := $(HOME)/strafer_ws
VENV_VLM := .venv_vlm

.PHONY: build test test-unit test-dgx lint lint-fix format format-check clean kill \
        install-tools udev serve-vlm serve-planner check-nvrtc help

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

# ---------- Kill ----------

kill: ## Kill all running ROS2 / strafer processes
	@pkill -9 -f "ros2|rtabmap|realsense2_camera_node|timestamp_fixer|imu_filter_madgwick|depth_downsampler|roboclaw|depthimage|validate_drive" 2>/dev/null || true
	@sleep 0.5
	@echo "All ROS processes killed."

# ---------- Setup ----------

install-tools: ## Install black, autopep8 (flake8 is already available via ROS)
	pip install black 'autopep8>=2,<3' --no-deps

udev: ## Install udev rules for RoboClaw symlinks (requires sudo)
	sudo cp source/strafer_ros/99-strafer.rules /etc/udev/rules.d/
	sudo udevadm control --reload-rules
	sudo udevadm trigger
	@echo "Verify with: ls -la /dev/roboclaw*"

# ---------- DGX Spark Services ----------

serve-vlm: check-nvrtc ## Start VLM grounding service on port 8100
	$(VENV_VLM)/bin/uvicorn strafer_vlm.service.app:create_app \
		--factory --host 0.0.0.0 --port $${GROUNDING_PORT:-8100}

serve-planner: check-nvrtc ## Start LLM planner service on port 8200
	$(VENV_VLM)/bin/uvicorn strafer_autonomy.planner.app:create_app \
		--factory --host 0.0.0.0 --port $${PLANNER_PORT:-8200}

test-dgx: ## Run autonomy + VLM tests (skips ROS-dependent tests)
	$(VENV_VLM)/bin/python -m pytest \
		source/strafer_autonomy/tests/ source/strafer_vlm/tests/ \
		-m "not requires_ros" -v

check-nvrtc: ## Verify NVRTC symlinks point to system CUDA 13.0
	@NVRTC_DIR="$(VENV_VLM)/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"; \
	if [ ! -L "$$NVRTC_DIR/libnvrtc.so.12" ]; then \
		echo "ERROR: $$NVRTC_DIR/libnvrtc.so.12 is not a symlink."; \
		echo "Run the NVRTC fix from docs/INTEGRATION_DGX_SPARK.md"; \
		exit 1; \
	fi; \
	TARGET=$$(readlink -f "$$NVRTC_DIR/libnvrtc.so.12"); \
	if echo "$$TARGET" | grep -q "cuda-13"; then \
		echo "NVRTC: OK ($$TARGET)"; \
	else \
		echo "ERROR: NVRTC symlink points to $$TARGET (expected cuda-13.x)"; \
		echo "Run the NVRTC fix from docs/INTEGRATION_DGX_SPARK.md"; \
		exit 1; \
	fi

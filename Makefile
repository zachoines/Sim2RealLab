# Sim2RealLab workspace Makefile
# Wraps colcon build/test workflows and Python lint/format tools.

SHELL := /bin/bash
VENV_VLM := .venv_vlm

# Host-specific paths. Override per-host by setting these in .env (see
# .env.example for the documented list); `source env_setup.sh` exports
# them into the shell environment that `make` inherits. The defaults
# below exist only so `make` still runs if .env has not been sourced.
COLCON_WS ?= $(HOME)/strafer_ws
ISAACLAB ?= $(HOME)/Documents/repos/IsaacLab/isaaclab.sh
CONDA_ROOT ?= $(HOME)/miniconda3
CONDA_ENV ?= env_isaaclab3

.PHONY: build test test-unit test-driver test-ros test-autonomy test-vlm test-dgx test-jetson test-lab test-lab-pure lint lint-fix format format-check clean kill \
        launch launch-nav launch-autonomy launch-sim clean-map \
        install-tools udev serve-vlm serve-planner check-nvrtc help \
        sim-bridge sim-bridge-gui sim-harness harness-smoke

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------- Build ----------

build: ## Build all ROS2 packages with colcon
	cd $(COLCON_WS) && source /opt/ros/humble/setup.bash && \
		colcon build --symlink-install

# ---------- Test ----------
# Per-suite building blocks, then per-host umbrellas. Run the umbrella for
# your host — `make test` auto-dispatches (DGX -> test-dgx, Jetson ->
# test-jetson); each suite runs in its own env. The strafer_lab Kit +
# pure-Python suites are `test-lab` / `test-lab-pure`, defined further down.

# Interpreter for the autonomy tests. Host-agnostic default (`python`); the
# DGX umbrella overrides it to .venv_vlm, which carries strafer_autonomy.
AUTONOMY_PY ?= python

test-autonomy: ## Planner/executor unit tests — host-agnostic (needs strafer_autonomy + pytest in the chosen interpreter)
	@# Host-agnostic: runs in $(AUTONOMY_PY) (the active `python` by default;
	@# the DGX umbrella pins it to .venv_vlm). On the DGX, conda `base` has
	@# neither pytest nor strafer_autonomy, so fail with guidance rather than
	@# a cryptic "No module named pytest". PYTHONPATH cleared so the vendored
	@# ROS 2 (3.11) site-packages env_setup.sh adds can't leak launch_testing
	@# into pytest's plugin autoload.
	@if ! PYTHONPATH= $(AUTONOMY_PY) -c "import pytest" >/dev/null 2>&1; then \
		echo "ERROR: '$(AUTONOMY_PY)' has no pytest (likely conda base). Pick an env that carries strafer_autonomy + pytest:"; \
		echo "  DGX:    make test-dgx   (pins .venv_vlm), or  make test-autonomy AUTONOMY_PY=$(VENV_VLM)/bin/python,"; \
		echo "          or activate .venv_vlm / env_isaaclab3 first."; \
		echo "  Jetson: pip install -e source/strafer_autonomy  into the active python first."; \
		exit 1; \
	fi
	PYTHONPATH= $(AUTONOMY_PY) -m pytest source/strafer_autonomy/tests/ \
		-m "not requires_ros" -v

test-vlm: ## VLM service tests — DGX-only (uses .venv_vlm)
	@if [ ! -x "$(VENV_VLM)/bin/python" ]; then \
		echo "ERROR: $(VENV_VLM) not found — bootstrap it first (see source/strafer_vlm/README.md)."; \
		exit 1; \
	fi
	PYTHONPATH= $(VENV_VLM)/bin/python -m pytest source/strafer_vlm/tests/ -v

test-ros: ## ROS 2 package tests via colcon — Jetson (run `make build` first)
	cd $(COLCON_WS) && source /opt/ros/humble/setup.bash && \
		colcon test && colcon test-result --verbose

test-driver: ## strafer_driver unit tests directly with pytest — Jetson
	cd source/strafer_ros/strafer_driver && \
		python -m pytest test/ -v

test-dgx: ## DGX e2e umbrella — autonomy + vlm + lab, each in its env. SKIP_KIT=1 swaps the heavy Kit suite for the fast pure-Python lab half.
	@rc=0; \
	$(MAKE) --no-print-directory test-autonomy AUTONOMY_PY=$(VENV_VLM)/bin/python || rc=1; \
	$(MAKE) --no-print-directory test-vlm || rc=1; \
	if [ "$$SKIP_KIT" = "1" ]; then \
		echo "[test-dgx] SKIP_KIT=1 — running test-lab-pure (no Kit boot) instead of test-lab"; \
		$(MAKE) --no-print-directory test-lab-pure || rc=1; \
	else \
		$(MAKE) --no-print-directory test-lab || rc=1; \
	fi; \
	exit $$rc

test-jetson: ## Jetson e2e umbrella — autonomy + ros + driver
	@rc=0; \
	$(MAKE) --no-print-directory test-autonomy || rc=1; \
	$(MAKE) --no-print-directory test-ros || rc=1; \
	$(MAKE) --no-print-directory test-driver || rc=1; \
	exit $$rc

test: ## Auto-dispatch to the per-host umbrella (DGX -> test-dgx, Jetson -> test-jetson)
	@if [ -x "$(ISAACLAB)" ]; then \
		echo "[test] Isaac Sim found ($(ISAACLAB)) — running test-dgx"; \
		$(MAKE) --no-print-directory test-dgx; \
	elif command -v colcon >/dev/null 2>&1; then \
		echo "[test] colcon found — running test-jetson"; \
		$(MAKE) --no-print-directory test-jetson; \
	else \
		echo "[test] Host not detected. Run 'make test-dgx' or 'make test-jetson' explicitly."; \
		exit 1; \
	fi

test-unit: ## (deprecated) alias for test-driver
	@echo "note: 'make test-unit' was renamed — use 'make test-driver'."
	@$(MAKE) --no-print-directory test-driver

# ---------- Lint / Format ----------

lint: ## Run flake8 on all Python source
	python3 -m flake8 source/strafer_ros/ source/strafer_shared/ source/strafer_lab/scripts/ \
		--max-line-length 100 --extend-ignore=E203,W503

lint-fix: ## Auto-fix lint issues with autopep8
	python3 -m autopep8 --in-place --recursive --max-line-length 100 \
		source/strafer_ros/ source/strafer_shared/ source/strafer_lab/scripts/

format: ## Run black on all Python source
	python3 -m black source/strafer_ros/ source/strafer_shared/ source/strafer_lab/scripts/

format-check: ## Check formatting without modifying files
	python3 -m black --check source/strafer_ros/ source/strafer_shared/ source/strafer_lab/scripts/

# ---------- Launch ----------

launch: launch-nav ## Alias for launch-nav

launch-nav: ## Launch navigation stack (driver + perception + SLAM + Nav2)
	source $(COLCON_WS)/install/setup.bash && \
		ros2 launch strafer_bringup navigation.launch.py

launch-autonomy: ## Launch full autonomy stack (nav + executor → DGX services). Backend via STRAFER_NAV_BACKEND in env_autonomy.env.
	@source source/strafer_ros/strafer_bringup/config/env_autonomy.env && \
	if [ -z "$$VLM_URL" ] || [ -z "$$PLANNER_URL" ]; then \
		echo "Usage: VLM_URL=http://<DGX>:8100 PLANNER_URL=http://<DGX>:8200 make launch-autonomy"; \
		exit 1; \
	fi; \
	if { [ "$$STRAFER_NAV_BACKEND" = "strafer_direct" ] || [ "$$STRAFER_NAV_BACKEND" = "hybrid_nav2_strafer" ]; } && [ -z "$$STRAFER_INFERENCE_MODEL_PATH" ]; then \
		echo "ERROR: STRAFER_NAV_BACKEND=$$STRAFER_NAV_BACKEND but STRAFER_INFERENCE_MODEL_PATH is empty — strafer_inference will NOT advertise navigate_to_pose; every mission silently falls back to nav2. Set STRAFER_INFERENCE_MODEL_PATH in env_autonomy.env."; \
		exit 1; \
	fi
	source $(COLCON_WS)/install/setup.bash && \
		source source/strafer_ros/strafer_bringup/config/env_autonomy.env && \
		ros2 launch strafer_bringup autonomy.launch.py \
			vlm_url:=$$VLM_URL planner_url:=$$PLANNER_URL

launch-sim: ## Launch Jetson sim-in-the-loop bringup (consumes DGX bridge topics; foxglove on :8765). Backend via STRAFER_NAV_BACKEND in env_sim_in_the_loop.env. Env: DONUT_WARMUP=false skips the startup spin; LAUNCH_ARGS="k:=v ..." passes arbitrary extras.
	@source source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env && \
	if { [ "$$STRAFER_NAV_BACKEND" = "strafer_direct" ] || [ "$$STRAFER_NAV_BACKEND" = "hybrid_nav2_strafer" ]; } && [ -z "$$STRAFER_INFERENCE_MODEL_PATH" ]; then \
		echo "ERROR: STRAFER_NAV_BACKEND=$$STRAFER_NAV_BACKEND but STRAFER_INFERENCE_MODEL_PATH is empty — strafer_inference will NOT advertise navigate_to_pose; every mission silently falls back to nav2. Set STRAFER_INFERENCE_MODEL_PATH in env_sim_in_the_loop.env."; \
		exit 1; \
	fi
	source $(COLCON_WS)/install/setup.bash && \
		source source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env && \
		ros2 launch strafer_bringup bringup_sim_in_the_loop.launch.py \
			vlm_url:=$${VLM_URL:-http://192.168.50.196:8100} \
			planner_url:=$${PLANNER_URL:-http://192.168.50.196:8200} \
			donut_warmup:=$${DONUT_WARMUP:-true} \
			$(LAUNCH_ARGS)

# ---------- Clean ----------

clean: ## Remove colcon build artifacts
	cd $(COLCON_WS) && rm -rf build/ install/ log/

clean-map: ## Delete corrupted or stale RTAB-Map database
	rm -f $(HOME)/.ros/rtabmap.db
	@echo "RTAB-Map database removed. SLAM will start fresh on next launch."

# ---------- Kill ----------

kill: ## Kill all running ROS2 / strafer processes
	@pkill -9 -f "ros2|rtabmap|nav2_|realsense2_camera_node|timestamp_fixer|imu_filter_madgwick|depth_downsampler|roboclaw|depthimage|validate_drive|foxglove_bridge|strafer-executor|goal_projection|strafer_inference|strafer_subgoal_generator" 2>/dev/null || true
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

test-lab: ## Run ALL strafer_lab tests in env_isaaclab3 — Kit suites (run_tests.py) + pure-Python (tests/). The canonical strafer_lab gate.
	@# Both halves live in env_isaaclab3: the Kit suites need the bespoke
	@# run_tests.py wrapper (Isaac Sim's os._exit kills pytest's summary),
	@# while tests/ runs as plain pytest with no Kit boot. lerobot coexists
	@# with the env's CUDA torch, so the harness suite folded in here (and
	@# gained pxr) — no separate venv. Both halves always run; the target
	@# exits non-zero if either failed. env_setup.sh supplies LD_PRELOAD and
	@# scrubs the vendored-ROS2 path off PYTHONPATH so pytest autoload stays
	@# clean.
	@source env_setup.sh && rc=0; \
		$(ISAACLAB) -p source/strafer_lab/run_tests.py all || rc=1; \
		$$STRAFER_ISAACLAB_PYTHON -m pytest source/strafer_lab/tests/ || rc=1; \
		exit $$rc

test-lab-pure: ## Fast strafer_lab iteration — the pure-Python tests/ only (no Kit boot, ~seconds) in env_isaaclab3
	@source env_setup.sh && \
		$$STRAFER_ISAACLAB_PYTHON -m pytest source/strafer_lab/tests/ -v

check-nvrtc: ## Verify NVRTC symlinks point to system CUDA 13.0
	@NVRTC_DIR="$(VENV_VLM)/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"; \
	if [ ! -L "$$NVRTC_DIR/libnvrtc.so.12" ]; then \
		echo "ERROR: $$NVRTC_DIR/libnvrtc.so.12 is not a symlink."; \
		echo "Run the NVRTC fix from source/strafer_vlm/README.md (Install)"; \
		exit 1; \
	fi; \
	TARGET=$$(readlink -f "$$NVRTC_DIR/libnvrtc.so.12"); \
	if echo "$$TARGET" | grep -q "cuda-13"; then \
		echo "NVRTC: OK ($$TARGET)"; \
	else \
		echo "ERROR: NVRTC symlink points to $$TARGET (expected cuda-13.x)"; \
		echo "Run the NVRTC fix from source/strafer_vlm/README.md (Install)"; \
		exit 1; \
	fi

# ---------- DGX Spark Sim-in-the-loop ----------

sim-bridge: ## Launch Isaac Sim + ROS 2 bridge (headless)
	@source env_setup.sh && \
		source $(CONDA_ROOT)/etc/profile.d/conda.sh && \
		conda activate $(CONDA_ENV) && \
		echo "[sim-bridge] ROS_DISTRO=$$ROS_DISTRO, LD_LIBRARY_PATH head: $$(echo $$LD_LIBRARY_PATH | cut -d: -f1)" && \
		$(ISAACLAB) -p source/strafer_lab/scripts/run_sim_in_the_loop.py \
			--mode bridge --headless --enable_cameras \
			$${SCENE_NAME:+--scene-name $$SCENE_NAME} \
			$${SCENE_USD:+--scene-usd $$SCENE_USD}

sim-bridge-gui: ## Launch Isaac Sim + ROS 2 bridge with the viewport open
	@source env_setup.sh && \
		source $(CONDA_ROOT)/etc/profile.d/conda.sh && \
		conda activate $(CONDA_ENV) && \
		echo "[sim-bridge-gui] ROS_DISTRO=$$ROS_DISTRO, LD_LIBRARY_PATH head: $$(echo $$LD_LIBRARY_PATH | cut -d: -f1)" && \
		$(ISAACLAB) -p source/strafer_lab/scripts/run_sim_in_the_loop.py \
			--mode bridge --enable_cameras --viz kit \
			$${SCENE_NAME:+--scene-name $$SCENE_NAME} \
			$${SCENE_USD:+--scene-usd $$SCENE_USD}

sim-harness: ## Run sim-in-the-loop autonomous mission sweep (metadata travels in the scene USD)
	@if { [ -z "$$SCENE_NAME" ] && [ -z "$$SCENE_USD" ]; } || [ -z "$$OUTPUT_DIR" ]; then \
		echo "Usage: { SCENE_NAME=<name> | SCENE_USD=<path> } OUTPUT_DIR=<path> [MAX_MISSIONS=N] make sim-harness"; \
		exit 1; \
	fi
	@source env_setup.sh && \
		source $(CONDA_ROOT)/etc/profile.d/conda.sh && \
		conda activate $(CONDA_ENV) && \
		$(ISAACLAB) -p source/strafer_lab/scripts/run_sim_in_the_loop.py \
			--mode harness \
			$${SCENE_NAME:+--scene-name $$SCENE_NAME} \
			$${SCENE_USD:+--scene-usd $$SCENE_USD} \
			--output $$OUTPUT_DIR \
			$${MAX_MISSIONS:+--max-missions $$MAX_MISSIONS} \
			--headless --enable_cameras

harness-smoke: ## Jetson-free Kit smoke of the bridge harness capture path (scripted /cmd_vel)
	@source env_setup.sh && \
		source $(CONDA_ROOT)/etc/profile.d/conda.sh && \
		conda activate $(CONDA_ENV) && \
		$(ISAACLAB) -p source/strafer_lab/scripts/bridge_harness_smoke.py \
			$${SCENE:+--scene $$SCENE} \
			$${OUTPUT_DIR:+--output $$OUTPUT_DIR} \
			$${SMOKE_STEPS:+--steps $$SMOKE_STEPS} \
			$${REQUIRE_DETECTIONS:+--require-detections}

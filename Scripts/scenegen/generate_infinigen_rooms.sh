#!/usr/bin/env bash
# Generate Infinigen indoor rooms for the Phase 6 scene pipeline.
#
# Run this script in WSL/Linux where Infinigen is installed.
# Generated rooms are exported as USDC with --omniverse flag for
# compatibility with Isaac Sim / Omniverse.
#
# Infinigen generates full apartment layouts procedurally — room types
# and counts cannot be restricted. Each seed produces a unique apartment
# with a random mix of rooms.  The singleroom gin config constrains the
# floor plan to smaller layouts (~3 min/room with fast_solve) while
# overhead gin hides ceilings.  WITHOUT singleroom, generation takes
# 2+ hours per seed.
#
# Prerequisites:
#   1. Conda environment "infinigen" with Infinigen + bpy + pyyaml + coacd
#      (see docs/PHASE_6_SYNTHETIC_SCENE_GENERATION.md for setup)
#   2. CUDA toolkit installed: sudo apt install cuda-toolkit-12-6
#   3. Terrain C++ libs compiled (with CUDA): cd ~/infinigen && make terrain
#   4. Git submodules initialized: cd ~/infinigen && git submodule update --init --recursive
#
# Usage (from WSL):
#   # Default (10 rooms, fast quality):
#   bash /mnt/c/Worspace/Scripts/scenegen/generate_infinigen_rooms.sh
#
#   # 3 rooms with low textures:
#   NUM_ROOMS=3 TEXTURE_RES=256 bash /mnt/c/Worspace/Scripts/scenegen/generate_infinigen_rooms.sh
#
#   # High quality (full solver, no fast_solve):
#   QUALITY=high NUM_ROOMS=1 bash /mnt/c/Worspace/Scripts/scenegen/generate_infinigen_rooms.sh
#
# Output:
#   Assets/generated/infinigen_rooms/room_*/export_scene.usdc

set -euo pipefail

# Add CUDA to PATH if available
if [ -d "/usr/local/cuda-12.6/bin" ]; then
    export PATH="/usr/local/cuda-12.6/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"
fi

# Configuration
NUM_ROOMS="${NUM_ROOMS:-10}"
TEXTURE_RES="${TEXTURE_RES:-256}"
INFINIGEN_DIR="${INFINIGEN_DIR:-$HOME/infinigen}"
RAW_DIR="${RAW_DIR:-$HOME/infinigen_output/rooms}"
EXPORT_DIR="${EXPORT_DIR:-$HOME/infinigen_output/export}"
WINDOWS_OUTPUT="/mnt/c/Worspace/Assets/generated/infinigen_rooms"
WINDOWS_COPY="${WINDOWS_COPY:-1}"

# Quality presets:
#   QUALITY=fast    — singleroom + fast_solve + overhead (default, ~3 min/room)
#   QUALITY=high    — singleroom + full solver (~30-60 min/room)
#   QUALITY=multi   — multi-room apartment + fast_solve (~10-15 min/seed)
#   QUALITY=full    — full apartment + full solver + terrain (~2+ hr/room)
QUALITY="${QUALITY:-fast}"

case "${QUALITY}" in
    fast)
        GIN_CONFIGS="fast_solve overhead singleroom"
        GIN_OVERRIDES="compose_indoors.terrain_enabled=False"
        ;;
    high)
        # Full solver (no fast_solve), singleroom layout
        GIN_CONFIGS="overhead singleroom"
        GIN_OVERRIDES="compose_indoors.terrain_enabled=False"
        ;;
    multi)
        # Multi-room apartment with fast solver (no singleroom constraint)
        # Generates layouts with multiple connected rooms (bedroom, bathroom,
        # kitchen, hallway, etc.) for training on diverse multi-floor scenes.
        GIN_CONFIGS="fast_solve overhead"
        GIN_OVERRIDES="compose_indoors.terrain_enabled=False"
        ;;
    full)
        # Full apartment + full solver + terrain (very slow, 2+ hr/room)
        GIN_CONFIGS="overhead"
        GIN_OVERRIDES=""
        ;;
    *)
        echo "ERROR: Unknown QUALITY=${QUALITY}. Use: fast, high, full"
        exit 1
        ;;
esac

echo "============================================"
echo "Infinigen Room Generation for Phase 6"
echo "============================================"
echo "Number of rooms: ${NUM_ROOMS}"
echo "Infinigen dir: ${INFINIGEN_DIR}"
echo "Texture resolution: ${TEXTURE_RES}"
echo "Quality: ${QUALITY}"
echo "Gin configs: ${GIN_CONFIGS}"
echo "Gin overrides: ${GIN_OVERRIDES:-<none>}"
echo ""

# Activate conda environment
if [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate" infinigen
elif [ -f "$HOME/anaconda3/bin/activate" ]; then
    source "$HOME/anaconda3/bin/activate" infinigen
else
    echo "WARNING: Could not find conda. Assuming infinigen env is already active."
fi

# Change to Infinigen directory (needed for gin config resolution)
cd "${INFINIGEN_DIR}"

# Verify Infinigen is importable
if ! python -c "import infinigen_examples.generate_indoors" 2>/dev/null; then
    echo "ERROR: Cannot import infinigen_examples.generate_indoors"
    echo "Make sure Infinigen is installed: pip install -e ."
    echo "And terrain libs are compiled: make terrain"
    exit 1
fi

# Verify terrain libs exist
if [ ! -f "${INFINIGEN_DIR}/infinigen/terrain/lib/cpu/elements/waterbody.so" ]; then
    echo "ERROR: Terrain C++ libs not compiled."
    echo "Run: cd ${INFINIGEN_DIR} && make terrain"
    exit 1
fi

mkdir -p "${RAW_DIR}"
mkdir -p "${EXPORT_DIR}"

TOTAL=0
FAILED=0

echo ""
echo "--- Generating ${NUM_ROOMS} rooms ---"

for i in $(seq 1 "${NUM_ROOMS}"); do
    SEED=$((RANDOM + i))
    ROOM_NAME="room_${i}"
    ROOM_RAW="${RAW_DIR}/${ROOM_NAME}"
    ROOM_EXPORT="${EXPORT_DIR}/${ROOM_NAME}"

    echo "  [${ROOM_NAME}] seed=${SEED} ..."

    # Clean any previous partial output
    rm -rf "${ROOM_RAW}" "${ROOM_EXPORT}"

    # Step 1: Generate room geometry via Blender (bpy)
    # Infinigen generates apartment shells; Replicator adds clutter later.
    if python -m infinigen_examples.generate_indoors \
        --seed "${SEED}" \
        --task coarse \
        --output_folder "${ROOM_RAW}" \
        -g ${GIN_CONFIGS} \
        ${GIN_OVERRIDES:+-p ${GIN_OVERRIDES}} \
        2>&1 | tail -5; then

        # Step 2: Export .blend to USDC with Omniverse-compatible materials
        # The export tool produces: export_scene.blend/export_scene.usdc + textures/
        if python -m infinigen.tools.export \
            --input_folder "${ROOM_RAW}" \
            --output_folder "${ROOM_EXPORT}" \
            -f usdc \
            -r "${TEXTURE_RES}" \
            --omniverse \
            2>&1 | tail -3; then

            # The export creates a nested export_scene.blend/ subdirectory
            USDC_DIR="${ROOM_EXPORT}/export_scene.blend"

            if [ -f "${USDC_DIR}/export_scene.usdc" ]; then
                echo "    OK: ${ROOM_NAME} ($(du -sh "${USDC_DIR}" | cut -f1))"
                TOTAL=$((TOTAL + 1))

                # Copy to Windows workspace if enabled
                if [ "${WINDOWS_COPY}" = "1" ]; then
                    DEST="${WINDOWS_OUTPUT}/${ROOM_NAME}"
                    mkdir -p "${DEST}"
                    cp -r "${USDC_DIR}"/* "${DEST}/"
                    echo "    Copied to: ${DEST}"
                fi
            else
                echo "    FAIL: USDC not found after export for ${ROOM_NAME}"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "    FAIL: export failed for ${ROOM_NAME} (exit code $?)"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "    FAIL: generation failed for ${ROOM_NAME} (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
echo "Generation complete"
echo "  Success: ${TOTAL}"
echo "  Failed:  ${FAILED}"
if [ "${WINDOWS_COPY}" = "1" ]; then
    echo "  Output:  ${WINDOWS_OUTPUT}/"
else
    echo "  Output:  ${EXPORT_DIR}/"
fi
echo "============================================"

if [ "${TOTAL}" -eq 0 ]; then
    echo "WARNING: No rooms generated. Check errors above."
    exit 1
fi

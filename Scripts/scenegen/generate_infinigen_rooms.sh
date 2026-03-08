#!/usr/bin/env bash
# Generate Infinigen indoor rooms for the Phase 6 scene pipeline.
#
# Run this script in WSL/Linux where Infinigen is installed.
# Generated rooms are exported as USDC with --omniverse flag for
# compatibility with Isaac Sim / Omniverse.
#
# Prerequisites:
#   1. Conda environment "infinigen" with Infinigen + bpy + pyyaml + coacd
#      (see docs/PHASE_6_SYNTHETIC_SCENE_GENERATION.md for setup)
#   2. CUDA toolkit installed: sudo apt install cuda-toolkit-12-6
#   3. Terrain C++ libs compiled (with CUDA): cd ~/infinigen && make terrain
#   4. Git submodules initialized: cd ~/infinigen && git submodule update --init --recursive
#
# Usage (from WSL):
#   # Default (fast quality):
#   bash /mnt/c/Worspace/Scripts/scenegen/generate_infinigen_rooms.sh
#
#   # Fully furnished rooms with small clutter:
#   QUALITY=high ROOMS_PER_TYPE=1 bash /mnt/c/Worspace/Scripts/scenegen/generate_infinigen_rooms.sh
#
#   # Full quality + outdoor terrain through windows:
#   QUALITY=full ROOMS_PER_TYPE=1 bash /mnt/c/Worspace/Scripts/scenegen/generate_infinigen_rooms.sh
#
#   # Fast mode but boost small object placement:
#   QUALITY=medium ROOMS_PER_TYPE=1 bash /mnt/c/Worspace/Scripts/scenegen/generate_infinigen_rooms.sh
#
# Output:
#   Assets/generated/infinigen_rooms/dining_room_*/export_scene.usdc
#   Assets/generated/infinigen_rooms/living_room_*/export_scene.usdc
#   Assets/generated/infinigen_rooms/bedroom_*/export_scene.usdc
#   Assets/generated/infinigen_rooms/kitchen_*/export_scene.usdc

set -euo pipefail

# Add CUDA to PATH if available
if [ -d "/usr/local/cuda-12.6/bin" ]; then
    export PATH="/usr/local/cuda-12.6/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"
fi

# Configuration
ROOMS_PER_TYPE="${ROOMS_PER_TYPE:-10}"
TEXTURE_RES="${TEXTURE_RES:-2048}"
INFINIGEN_DIR="${INFINIGEN_DIR:-$HOME/infinigen}"
RAW_DIR="${RAW_DIR:-$HOME/infinigen_output/rooms}"
EXPORT_DIR="${EXPORT_DIR:-$HOME/infinigen_output/export}"
WINDOWS_OUTPUT="/mnt/c/Worspace/Assets/generated/infinigen_rooms"
WINDOWS_COPY="${WINDOWS_COPY:-1}"

# Gin configs for single-room indoor generation
# "singleroom" = 1 room, no open doors
# "fast_solve" = terrain disabled, reduced solver iterations (~3 min/room)
#
# Quality presets:
#   QUALITY=fast    — fast_solve gin, no terrain (default, ~5 min/room)
#   QUALITY=medium  — fast_solve but boost small solver steps (~10 min/room)
#   QUALITY=high    — full solver, medium+small objects, no terrain (~30 min/room)
#   QUALITY=full    — full solver + terrain outside windows (~45 min/room)
#
# Individual overrides (applied after preset):
#   TERRAIN=1       — force terrain_enabled=True (outdoor scenery through windows)
#   SOLVE_SMALL=N   — override solve_steps_small (default varies by preset)
#   SOLVE_MEDIUM=N  — override solve_steps_medium
QUALITY="${QUALITY:-fast}"
TERRAIN="${TERRAIN:-}"
SOLVE_SMALL="${SOLVE_SMALL:-}"
SOLVE_MEDIUM="${SOLVE_MEDIUM:-}"

case "${QUALITY}" in
    fast)
        GIN_CONFIGS="singleroom fast_solve"
        GIN_OVERRIDES=""
        ;;
    medium)
        # Use fast_solve base but increase small-object solver steps
        GIN_CONFIGS="singleroom fast_solve"
        GIN_OVERRIDES="compose_indoors.solve_steps_small=40"
        ;;
    high)
        # Full solver (no fast_solve), disable terrain for speed
        GIN_CONFIGS="singleroom"
        GIN_OVERRIDES="compose_indoors.terrain_enabled=False"
        ;;
    full)
        # Full solver + terrain
        GIN_CONFIGS="singleroom"
        GIN_OVERRIDES=""
        ;;
    *)
        echo "ERROR: Unknown QUALITY=${QUALITY}. Use: fast, medium, high, full"
        exit 1
        ;;
esac

# Apply individual overrides
if [ "${TERRAIN}" = "1" ]; then
    GIN_OVERRIDES="${GIN_OVERRIDES} compose_indoors.terrain_enabled=True"
fi
if [ -n "${SOLVE_SMALL}" ]; then
    GIN_OVERRIDES="${GIN_OVERRIDES} compose_indoors.solve_steps_small=${SOLVE_SMALL}"
fi
if [ -n "${SOLVE_MEDIUM}" ]; then
    GIN_OVERRIDES="${GIN_OVERRIDES} compose_indoors.solve_steps_medium=${SOLVE_MEDIUM}"
fi

# Room types to generate (override with ROOM_TYPES env var)
# e.g. ROOM_TYPES="DiningRoom" for a single type
if [ -n "${ROOM_TYPES:-}" ]; then
    IFS=' ' read -ra ROOM_TYPE_ARR <<< "${ROOM_TYPES}"
else
    ROOM_TYPE_ARR=(
        "DiningRoom"
        "LivingRoom"
        "Bedroom"
        "Kitchen"
    )
fi

# Map room type to output prefix
declare -A ROOM_PREFIX
ROOM_PREFIX[DiningRoom]="dining_room"
ROOM_PREFIX[LivingRoom]="living_room"
ROOM_PREFIX[Bedroom]="bedroom"
ROOM_PREFIX[Kitchen]="kitchen"

echo "============================================"
echo "Infinigen Room Generation for Phase 6"
echo "============================================"
echo "Rooms per type: ${ROOMS_PER_TYPE}"
echo "Room types: ${ROOM_TYPE_ARR[*]}"
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

for ROOM_TYPE in "${ROOM_TYPE_ARR[@]}"; do
    PREFIX="${ROOM_PREFIX[$ROOM_TYPE]}"
    echo ""
    echo "--- Generating ${ROOMS_PER_TYPE} ${ROOM_TYPE} rooms ---"

    for i in $(seq 1 "${ROOMS_PER_TYPE}"); do
        SEED=$((RANDOM + i))
        ROOM_RAW="${RAW_DIR}/${PREFIX}_${i}"
        ROOM_EXPORT="${EXPORT_DIR}/${PREFIX}_${i}"
        ROOM_NAME="${PREFIX}_${i}"

        echo "  [${ROOM_NAME}] seed=${SEED} ..."

        # Clean any previous partial output
        rm -rf "${ROOM_RAW}" "${ROOM_EXPORT}"

        # Step 1: Generate room geometry via Blender (bpy)
        # Uses singleroom gin config + optional fast_solve
        # Infinigen generates room shells; Replicator adds clutter later.
        GIN_OVERRIDES_FULL="compose_indoors.room_type=\"${ROOM_TYPE}\" compose_indoors.restrict_single_supported_roomtype=True"
        if [ -n "${GIN_OVERRIDES}" ]; then
            GIN_OVERRIDES_FULL="${GIN_OVERRIDES} ${GIN_OVERRIDES_FULL}"
        fi
        if python -m infinigen_examples.generate_indoors \
            --seed "${SEED}" \
            --task coarse \
            --output_folder "${ROOM_RAW}" \
            -g ${GIN_CONFIGS} \
            -p ${GIN_OVERRIDES_FULL} \
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

#!/usr/bin/env bash
# Run a robot-stack test suite, natively when the host carries ROS + colcon and
# otherwise inside strafer-cpu:humble against the live working tree.
#
#   tools/run_ros_tests.sh ros        # every strafer_ros package
#   tools/run_ros_tests.sh driver     # strafer_driver only
#   tools/run_ros_tests.sh autonomy   # strafer_autonomy, no ROS needed
#   tools/run_ros_tests.sh ros -k Starvation      # extra args reach pytest
#
# pytest, not `colcon test`: colcon's ament_python task invokes
# `python3 -m unittest -v` with no discovery arguments, which collects nothing
# and reports OK, so a colcon-driven gate passes without running a test.
#
# Env: STRAFER_CPU_IMAGE, ROS_TEST_PKGS, FORCE_CONTAINER=1.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${STRAFER_CPU_IMAGE:-strafer-cpu:humble}"
SUITE="${1:-ros}"; shift || true
# printf %q over an empty argv yields a literal '' that pytest reads as a path.
PYTEST_ARGS=""
[ $# -gt 0 ] && PYTEST_ARGS=$(printf '%q ' "$@")

ROS_TEST_PKGS="${ROS_TEST_PKGS:-strafer_inference strafer_navigation strafer_slam \
strafer_bringup strafer_driver strafer_perception strafer_description}"
# launch/ and config/ install as data_files, which colcon COPIES even under
# --symlink-install, so these need a rebuild before their config tests see
# working-tree edits.
ROS_BUILD_PKGS="strafer_inference strafer_navigation strafer_slam strafer_bringup"

case "$SUITE" in
  ros)      PKGS="$ROS_TEST_PKGS" ;;
  driver)   PKGS="strafer_driver" ;;
  autonomy) PKGS="" ;;
  *) echo "usage: $(basename "$0") {ros|driver|autonomy} [pytest args]" >&2; exit 2 ;;
esac

native_ros() { command -v colcon >/dev/null 2>&1 && [ -d /opt/ros ]; }
have_pytest() { python3 -m pytest --version >/dev/null 2>&1; }

# The per-package loop. One package at a time: each ships a test/__init__.py, so
# a single pytest run over all of them collides on the module name "test".
# pytest exit 5 (collected nothing) is normal per-package under a -k filter, so
# it is neutral there; a grand total of zero across every package is the real
# failure and is caught after the loop, so a typo'd filter cannot read green.
read -r -d '' RUN_PKGS <<'LOOP'
  rc=0; total=0
  for p in $PKGS; do
    [ -d "$ROOT/$p/test" ] || continue
    out=$(cd "$ROOT/$p" && python3 -m pytest test/ -q $ARGS 2>&1); prc=$?
    line=$(echo "$out" | grep -E "passed|failed|error|no tests ran" | tail -1)
    n=$(echo "$line" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+" | head -1)
    total=$((total + ${n:-0}))
    case $prc in
      0) printf "  ok    %-22s %s\n" "$p" "$line" ;;
      5) printf "  none  %-22s no tests collected\n" "$p" ;;
      *) printf "  FAIL  %-22s %s\n" "$p" "$line"; echo "$out" | tail -40; rc=1 ;;
    esac
  done
  if [ "$total" -eq 0 ]; then
    echo "  ERROR: no tests ran in ANY package — broken layout, or a filter that"
    echo "         matched nothing. Not reporting this as a pass."
    rc=1
  fi
  echo "  ----- $total passed -----"
  exit $rc
LOOP

run_autonomy() {
  # Marked "not requires_ros" and deliberately run WITHOUT sourcing ROS: a
  # vendored ROS 2 site-packages on PYTHONPATH leaks launch_testing into
  # pytest's plugin autoload.
  local root="$1"
  PYTHONPATH= python3 -m pytest "$root/tests/" -m "not requires_ros" -q $PYTEST_ARGS
}

if [ "${FORCE_CONTAINER:-0}" != 1 ] && \
   { { [ "$SUITE" = autonomy ] && have_pytest; } || \
     { [ "$SUITE" != autonomy ] && native_ros; }; }; then
  echo "[$SUITE] host toolchain present — running natively"
  if [ "$SUITE" = autonomy ]; then
    run_autonomy "$REPO/source/strafer_autonomy"; exit $?
  fi
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  ( cd "${COLCON_WS:-$HOME/strafer_ws}" \
      && colcon build --symlink-install --packages-select $ROS_BUILD_PKGS >/dev/null 2>&1 \
      && source install/setup.bash ) || true
  ROOT="$REPO/source/strafer_ros" PKGS="$PKGS" ARGS="$PYTEST_ARGS" bash -c "$RUN_PKGS"
  exit $?
fi

if [ "$SUITE" = autonomy ] && docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[$SUITE] no host toolchain — running in $IMAGE"
  # The whole repo, not just the package: tests/conftest.py resolves a
  # strafer_lab stub relative to the repo root, and a package-only mount leaves
  # it unable to find source/strafer_lab, turning skips into collection errors.
  exec docker run --rm --network host -v "$REPO:/repo" \
    -e ARGS="$PYTEST_ARGS" "$IMAGE" bash -lc '
      cd /repo/source/strafer_autonomy
      eval "set -- $ARGS"
      PYTHONPATH= exec python3 -m pytest tests/ -m "not requires_ros" -q "$@"
    '
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: no native ROS toolchain and image '$IMAGE' is missing." >&2
  echo "       Build it first:  make images" >&2
  exit 1
fi
echo "[$SUITE] no host toolchain — running in $IMAGE (live working tree)"

# --network host: the L4T kernel lacks iptable_raw, so Docker's bridge
# networking fails to initialise on the robot host.
docker run --rm --network host \
  -v "$REPO/source/strafer_ros:/ws/src/strafer_ros" \
  -v "$REPO/source/strafer_shared:/opt/strafer/strafer_shared" \
  -e SUITE="$SUITE" -e PKGS="$PKGS" -e ARGS="$PYTEST_ARGS" \
  -e BUILD_PKGS="$ROS_BUILD_PKGS" -e RUN_PKGS="$RUN_PKGS" \
  "$IMAGE" bash -lc '
    set -o pipefail             # not -u: ROS setup.bash reads unbound vars
    source /opt/ros/humble/setup.bash
    cd /ws && colcon build --symlink-install --packages-select $BUILD_PKGS \
      --cmake-args -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
    source /ws/install/setup.bash
    ROOT=/ws/src/strafer_ros bash -c "$RUN_PKGS"
  '

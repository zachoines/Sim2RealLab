#!/usr/bin/env bash
# Set the inference container's (policy artifact x subgoal anchoring) and prove
# the change took effect from inside the container.
#
#   ./configure_inference.sh <model> <mission|rolling>
#
#   model      a filename under /models (e.g. policy.onnx), or any absolute
#              container path. Run with no arguments to list what is mounted.
#   anchoring  mission | rolling -- the two values subgoal_generator_node
#              accepts (_ANCHOR_MISSION / _ANCHOR_ROLLING).
#
# The model argument is deliberately not an alias table: which artifacts exist
# is a property of the host's models directory and changes per experiment, so
# hardcoding names here would date the tool on its first A/B against a new pair.
#
# WHY A TOOL AND NOT TWO COMMANDS
#
#   1. `docker compose restart` REUSES the old container environment. A policy
#      rollback applied that way keeps running the previous artifact while the
#      config on disk says otherwise, with nothing in the logs to say so. Only
#      `up -d --force-recreate` applies it.
#   2. A `docker compose` invocation whose -f/--profile chain differs from the
#      one the stack was created with makes compose reconcile a DIFFERENT
#      desired state, which can alter or drop services it was never asked to
#      touch. This script reads the chain back off the running container's
#      `com.docker.compose.project.config_files` label and mirrors it.
#   3. Hand-maintained copies of subgoal_generator.yaml drift from the image's
#      installed config, and a drifted copy changes more than the one line it
#      was meant to. This script generates the anchoring config from the IMAGE --
#      not from the running container, whose copy of that path is this tool's
#      own previous bind mount -- rewrites exactly the anchoring key, and diffs
#      the two to prove nothing else moved.
#   4. Every cadence counter in the node is cumulative since process start with
#      no reset path, so a surviving node sums the new run onto the previous one.
#      Recreation is what zeroes them.
#
# The verification at the end is the point: it exits non-zero when the container
# is not demonstrably running what was asked for, rather than falling through on
# a timeout.
set -euo pipefail

CTR=strafer_inference

usage() {
  echo "usage: $(basename "$0") <model> <mission|rolling>" >&2
  echo "  model: a filename under /models, or an absolute container path" >&2
  if docker inspect "$CTR" >/dev/null 2>&1; then
    echo "" >&2
    echo "mounted under /models:" >&2
    docker exec "$CTR" sh -c 'ls -1 /models 2>/dev/null' 2>/dev/null \
      | sed 's/^/  /' >&2 || echo "  (could not list)" >&2
  fi
  exit 2
}
[ $# -eq 2 ] || usage

MODEL_ARG="$1"
ANCHOR="$2"
SVC=inference
DEPLOY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN_DIR="$DEPLOY/.anchor_configs"          # gitignored; regenerated per switch
INSTALLED=/ws/install/strafer_inference/share/strafer_inference/config/subgoal_generator.yaml

case "$ANCHOR" in
  mission|rolling) ;;
  *) echo "anchoring must be 'mission' or 'rolling'" >&2; usage ;;
esac

# A bare name resolves under /models; anything absolute is taken as given, so a
# model mounted elsewhere still works without special-casing.
case "$MODEL_ARG" in
  /*) MODEL="$MODEL_ARG" ;;
  *)  MODEL="/models/$MODEL_ARG" ;;
esac

docker inspect "$CTR" >/dev/null 2>&1 || {
  echo "ERROR: $CTR is not running. This tool reconfigures a live stack;" >&2
  echo "       it does not create one." >&2
  exit 1
}

# Fail here rather than after a recreate: an unreadable model path otherwise
# surfaces only as a node that comes up without a policy, several minutes later.
docker exec "$CTR" test -f "$MODEL" 2>/dev/null || {
  echo "ERROR: $MODEL does not exist inside $CTR." >&2
  echo "mounted under /models:" >&2
  docker exec "$CTR" sh -c 'ls -1 /models 2>/dev/null' 2>/dev/null | sed 's/^/  /' >&2
  exit 1
}

# --- mirror the chain the stack was actually created with --------------------
CHAIN_CSV="$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project.config_files"}}' "$CTR")"
[ -n "$CHAIN_CSV" ] || { echo "ERROR: $CTR carries no compose config_files label" >&2; exit 1; }
FLAGS=()
IFS=',' read -ra FILES <<< "$CHAIN_CSV"
for f in "${FILES[@]}"; do
  # The anchor overlay is appended fresh below; never inherit a stale copy.
  [ "$(basename "$f")" = "docker-compose.override.anchor.yml" ] && continue
  FLAGS+=(-f "$f")
done
FLAGS+=(-f "$DEPLOY/docker-compose.override.anchor.yml")

# --- generate the anchoring config FROM THE IMAGE ------------------------------
# Read the baseline from the image, NOT with `docker exec` on the running
# container: once this tool has run, that path inside the container IS the anchor
# bind mount, so exec would read back this tool's own previous output and the
# drift guard below would compare a file against itself -- blind to exactly the
# image-side drift it exists to catch.
mkdir -p "$GEN_DIR"
BASE="$GEN_DIR/subgoal_generator.image.yaml"
OUT="$GEN_DIR/subgoal_generator.$ANCHOR.yaml"
IMG="$(docker inspect -f '{{.Config.Image}}' "$CTR")"
docker run --rm --entrypoint cat "$IMG" "$INSTALLED" > "$BASE"
[ -s "$BASE" ] || { echo "ERROR: could not read $INSTALLED from image $IMG" >&2; exit 1; }

# Rewrite exactly the anchoring value, preserving indentation and quoting style.
sed -E "s/^([[:space:]]*subgoal_anchoring:[[:space:]]*).*$/\1\"$ANCHOR\"/" "$BASE" > "$OUT"
grep -qE "^[[:space:]]*subgoal_anchoring:[[:space:]]*\"$ANCHOR\"" "$OUT" || {
  echo "ERROR: no subgoal_anchoring key found in the image's config" >&2; exit 1; }

DIFF_LINES="$(diff "$BASE" "$OUT" | grep -c '^[<>]' || true)"
if [ "$DIFF_LINES" -gt 2 ]; then
  echo "ERROR: the generated config differs from the image's in more than the" >&2
  echo "       anchoring line ($DIFF_LINES changed lines). Refusing to mount." >&2
  diff "$BASE" "$OUT" >&2 || true
  exit 1
fi

echo "=== configuring inference: $(basename "$MODEL") x ${ANCHOR} ==="
echo "    image  $IMG"
echo "    model  $MODEL"
echo "    config $OUT  ($((DIFF_LINES / 2)) line changed vs the image)"

STRAFER_INFERENCE_MODEL_PATH="$MODEL" \
STRAFER_ANCHOR_CONFIG="$OUT" \
docker compose "${FLAGS[@]}" up -d --force-recreate "$SVC"

# --- wait for the node, then verify ------------------------------------------
# `docker logs ... | grep -q` is avoided throughout: grep exits at its first
# match and SIGPIPEs docker logs, which Go re-raises as 141, and under
# `set -o pipefail` that fails the pipeline even though the match succeeded.
# Snapshot the log once per poll and match against the variable instead.
#
# 300 s covers a cold TensorRT engine build (~90 s for DEPTH_SUBGOAL here, and
# longer whenever the engine cache is cold).
echo "=== waiting for policy load ==="
DEADLINE=$((SECONDS + 300))
LOGS=""
while :; do
  LOGS="$(docker logs "$CTR" 2>&1 || true)"
  case "$LOGS" in *"strafer_inference node up:"*) break ;; esac
  if [ "$SECONDS" -ge "$DEADLINE" ]; then
    echo "ERROR: node did not report 'node up:' within 300 s. Last log lines:" >&2
    printf '%s\n' "$(docker logs --tail 20 "$CTR" 2>&1)" >&2
    exit 1
  fi
  sleep 2
done

echo "=== verified from inside the container ==="
FAIL=0
GOT_MODEL="$(docker exec "$CTR" printenv STRAFER_INFERENCE_MODEL_PATH || true)"
if [ "$GOT_MODEL" = "$MODEL" ]; then
  echo "  model path     $GOT_MODEL"
else
  echo "  MODEL MISMATCH: want $MODEL, got ${GOT_MODEL:-<unset>}" >&2; FAIL=1
fi

GOT_ANCHOR="$(docker exec "$CTR" grep -oE '^[[:space:]]*subgoal_anchoring:[[:space:]]*"[a-z]+"' "$INSTALLED" \
              | grep -oE '"[a-z]+"' | tr -d '"' || true)"
if [ "$GOT_ANCHOR" = "$ANCHOR" ]; then
  echo "  anchoring cfg  $GOT_ANCHOR"
else
  echo "  ANCHOR MISMATCH: want $ANCHOR, got ${GOT_ANCHOR:-<none>}" >&2; FAIL=1
fi

# The `anchoring=` line comes from the subgoal generator, which only launches
# under hybrid_nav2_strafer. Under strafer_direct there is no generator and no
# such line, so requiring it would fail every correct swap on that backend.
BACKEND="$(docker exec "$CTR" printenv STRAFER_NAV_BACKEND 2>/dev/null || echo "")"
case "$BACKEND" in
  hybrid_nav2_strafer)
    # The generator is a second node in the same launch and can log after the
    # inference node reports up, so poll a bounded window rather than race it.
    GEN_DEADLINE=$((SECONDS + 60))
    while :; do
      LOGS="$(docker logs "$CTR" 2>&1 || true)"
      case "$LOGS" in
        *"anchoring=$ANCHOR "*) echo "  node log       anchoring=$ANCHOR"; break ;;
      esac
      if [ "$SECONDS" -ge "$GEN_DEADLINE" ]; then
        echo "  SUBGOAL GENERATOR did not log anchoring=$ANCHOR within 60 s" >&2
        FAIL=1; break
      fi
      sleep 2
    done
    ;;
  *)
    echo "  node log       skipped (STRAFER_NAV_BACKEND=${BACKEND:-<unset>};"
    echo "                 the subgoal generator runs only under hybrid_nav2_strafer)"
    ;;
esac

case "$LOGS" in
  *"policy_loaded=True"*) echo "  policy         loaded" ;;
  *) echo "  POLICY NOT LOADED" >&2; FAIL=1 ;;
esac

# A `-dirty` or `unknown` revision means the running code cannot be named --
# surfaced here because this is the last checkpoint before a measurement.
REV="$(printf '%s\n' "$LOGS" | grep -m1 -oE 'revision=[^ ]+' || true)"
echo "  image          ${REV:-revision=<unstamped>}"
case "$REV" in
  *dirty*|revision=unknown|"") echo "  WARNING: image is not a clean stamped build" >&2 ;;
esac

[ "$FAIL" -eq 0 ] || { echo "SWITCH FAILED VERIFICATION" >&2; exit 1; }
echo "=== $(basename "$MODEL") x ${ANCHOR} active, counters zeroed by recreate ==="

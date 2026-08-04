#!/usr/bin/env bash
# Swap the inference container's (policy artifact x subgoal anchoring) and PROVE
# the swap took effect from inside the container.
#
#   ./switch_arm.sh <model> <mission|rolling>
#
#   model:  v1 | v2 | an absolute /models/... path
#
# Promoted from the 2026-08-02 validation session's session-local script, which
# is the only thing that reliably caught the silent-swap foot-gun below.
#
# WHY A TOOL AND NOT TWO COMMANDS
#
#   1. `docker compose restart` REUSES the old container environment. A policy
#      rollback applied that way silently keeps running the previous artifact
#      while the config on disk says otherwise -- observed, not theoretical.
#      Only `up -d --force-recreate` applies it.
#   2. A `docker compose` invocation whose -f/--profile chain differs from the
#      one the stack was created with makes compose reconcile a DIFFERENT
#      desired state, which can alter or drop services you did not name. This
#      script reads the chain back off the running container's
#      `com.docker.compose.project.config_files` label and mirrors it, so the
#      chain cannot drift by hand.
#   3. Hand-maintained copies of subgoal_generator.yaml DRIFT from the image's
#      installed config, and a drifted copy changes more than the one line you
#      meant to change. This script generates the arm config FROM the image's
#      own installed file and rewrites exactly the anchoring key, then diffs the
#      two to prove nothing else moved.
#   4. Every cadence counter in the node is cumulative since process start with
#      no reset path, so a surviving node sums the new arm onto the old one.
#      Recreation is what zeroes them.
#
# The verification at the end is the point. It exits non-zero if the container
# is not demonstrably running what was asked for -- never silently falls through
# on timeout, which is how a mis-swapped arm reaches a report.
set -euo pipefail

usage() { echo "usage: $(basename "$0") <v1|v2|/models/path.onnx> <mission|rolling>" >&2; exit 2; }
[ $# -eq 2 ] || usage

MODEL_KEY="$1"
ANCHOR="$2"
SVC=inference
CTR=strafer_inference
DEPLOY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN_DIR="$DEPLOY/.anchor_configs"          # gitignored; regenerated per switch
INSTALLED=/ws/install/strafer_inference/share/strafer_inference/config/subgoal_generator.yaml

case "$ANCHOR" in
  mission|rolling) ;;
  *) echo "anchoring must be 'mission' or 'rolling'" >&2; usage ;;
esac

# v1 is /models/policy.onnx. There is deliberately no /models/*_v1.onnx symlink:
# the artifact identity is the file, and an alias invites a stale one.
case "$MODEL_KEY" in
  v1) MODEL=/models/policy.onnx ;;
  v2) MODEL=/models/strafer_depth_subgoal_v2_998.onnx ;;
  /*) MODEL="$MODEL_KEY" ;;
  *)  echo "model must be v1, v2, or an absolute /models/... path" >&2; usage ;;
esac

docker inspect "$CTR" >/dev/null 2>&1 || {
  echo "ERROR: $CTR is not running. Bring the stack up first -- this tool swaps" >&2
  echo "       an arm on a live stack; it does not create one." >&2
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

# --- generate the arm config FROM the image's installed config ---------------
mkdir -p "$GEN_DIR"
BASE="$GEN_DIR/subgoal_generator.installed.yaml"
OUT="$GEN_DIR/subgoal_generator.$ANCHOR.yaml"
docker exec "$CTR" cat "$INSTALLED" > "$BASE"
[ -s "$BASE" ] || { echo "ERROR: could not read $INSTALLED from $CTR" >&2; exit 1; }

# Rewrite exactly the anchoring value, preserving indentation and quoting style.
sed -E "s/^([[:space:]]*subgoal_anchoring:[[:space:]]*).*$/\1\"$ANCHOR\"/" "$BASE" > "$OUT"
grep -qE "^[[:space:]]*subgoal_anchoring:[[:space:]]*\"$ANCHOR\"" "$OUT" || {
  echo "ERROR: no subgoal_anchoring key found in the installed config" >&2; exit 1; }

# Prove the generated file differs from the image's config in ONE line only.
DIFF_LINES="$(diff "$BASE" "$OUT" | grep -c '^[<>]' || true)"
if [ "$DIFF_LINES" -gt 2 ]; then
  echo "ERROR: generated config differs from the installed one in more than the" >&2
  echo "       anchoring line ($DIFF_LINES changed lines). Refusing to mount." >&2
  diff "$BASE" "$OUT" >&2 || true
  exit 1
fi

echo "=== switching to ${MODEL_KEY} x ${ANCHOR} ==="
echo "    model  $MODEL"
echo "    config $OUT  (generated from the image; $((DIFF_LINES/2)) line changed)"

STRAFER_INFERENCE_MODEL_PATH="$MODEL" \
STRAFER_ANCHOR_CONFIG="$OUT" \
docker compose "${FLAGS[@]}" up -d --force-recreate "$SVC"

# --- wait for the node, then VERIFY. Never fall through silently. ------------
echo "=== waiting for policy load ==="
DEADLINE=$((SECONDS + 300))
until docker logs "$CTR" 2>&1 | grep -q "strafer_inference node up:"; do
  if [ $SECONDS -ge $DEADLINE ]; then
    echo "ERROR: node did not report 'node up:' within 300 s. Last log lines:" >&2
    docker logs --tail 20 "$CTR" 2>&1 >&2
    exit 1
  fi
  sleep 2
done

echo "=== VERIFIED FROM INSIDE THE CONTAINER ==="
FAIL=0
GOT_MODEL="$(docker exec "$CTR" printenv STRAFER_INFERENCE_MODEL_PATH || true)"
[ "$GOT_MODEL" = "$MODEL" ] \
  && echo "  model path     $GOT_MODEL" \
  || { echo "  MODEL MISMATCH: want $MODEL, got ${GOT_MODEL:-<unset>}" >&2; FAIL=1; }

GOT_ANCHOR="$(docker exec "$CTR" grep -oE '^[[:space:]]*subgoal_anchoring:[[:space:]]*"[a-z]+"' "$INSTALLED" | grep -oE '"[a-z]+"' | tr -d '"' || true)"
[ "$GOT_ANCHOR" = "$ANCHOR" ] \
  && echo "  anchoring cfg  $GOT_ANCHOR" \
  || { echo "  ANCHOR MISMATCH: want $ANCHOR, got ${GOT_ANCHOR:-<none>}" >&2; FAIL=1; }

docker logs "$CTR" 2>&1 | grep -q "anchoring=$ANCHOR" \
  && echo "  node log       anchoring=$ANCHOR" \
  || { echo "  NODE LOG does not report anchoring=$ANCHOR" >&2; FAIL=1; }

docker logs "$CTR" 2>&1 | grep -q "policy_loaded=True" \
  && echo "  policy         loaded" \
  || { echo "  POLICY NOT LOADED" >&2; FAIL=1; }

# A `-dirty` or `unknown` revision means you cannot say what code produced the
# run -- surfaced here because this is the last checkpoint before a measurement.
REV="$(docker logs "$CTR" 2>&1 | grep -m1 -oE 'revision=[^ ]+' || true)"
echo "  image          ${REV:-revision=<unstamped>}"
case "$REV" in
  *dirty*|revision=unknown|"") echo "  WARNING: image is not a clean stamped build" >&2 ;;
esac

[ "$FAIL" -eq 0 ] || { echo "SWITCH FAILED VERIFICATION" >&2; exit 1; }
echo "=== ${MODEL_KEY} x ${ANCHOR} active, counters zeroed by recreate ==="

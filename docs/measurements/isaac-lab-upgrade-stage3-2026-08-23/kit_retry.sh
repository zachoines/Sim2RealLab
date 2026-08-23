#!/usr/bin/env bash
# Run one Kit command, retrying if it hits the new pair's intermittent early-init hang.
#
# On the new pair a Kit-booting process intermittently deadlocks during carb
# initialisation — RSS stays around 48 MB, two threads sit in `futex_wait_queue`,
# no CUDA context is ever opened, and the process never proceeds. It does not
# recover on its own. The old pair does not do this (see the same-session control
# run: 487/487, no timeouts). Frequency is roughly one launch in three to ten.
#
# Every gate in this session that boots Kit therefore runs through here, so that
# a hang is retried rather than silently recorded as a failed measurement — and
# so that the retry count is itself part of the record instead of being lost.
#
# Usage:  kit_retry.sh <timeout_s> <attempts> <logfile> -- <command...>
#
# Writes each attempt to <logfile>.attemptN and copies the successful attempt to
# <logfile>. Prints one accounting line per attempt. Exits non-zero if every
# attempt hangs or fails.

set -u
TIMEOUT=$1; ATTEMPTS=$2; LOG=$3; shift 3
[ "${1:-}" = "--" ] && shift

mkdir -p "$(dirname "$LOG")"
for i in $(seq 1 "$ATTEMPTS"); do
    S=$(date +%s)
    timeout "$TIMEOUT" "$@" > "${LOG}.attempt${i}" 2>&1
    RC=$?
    E=$(date +%s)
    if [ "$RC" -eq 124 ]; then
        echo "[kit_retry] attempt $i: HANG (no completion in ${TIMEOUT}s) — retrying"
        # A hung Kit is killed by timeout(1) and leaves its carb shared-memory
        # segment and named semaphore behind; clear the dead ones so the leak
        # does not accumulate across retries.
        for f in /dev/shm/carb-RStringInternals-*; do
            b=$(basename "$f" 2>/dev/null) || continue
            pid="${b##*-}"
            case "$pid" in ''|*[!0-9]*) continue ;; esac
            kill -0 "$pid" 2>/dev/null || rm -f "/dev/shm/$b" "/dev/shm/sem.$b"
        done
        continue
    fi
    cp "${LOG}.attempt${i}" "$LOG"
    echo "[kit_retry] attempt $i: exit=$RC wall=$((E-S))s -> $LOG"
    exit "$RC"
done
echo "[kit_retry] all $ATTEMPTS attempts hung"
exit 124

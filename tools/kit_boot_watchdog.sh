#!/usr/bin/env bash
# Run one Kit-booting command, relaunching it if carb initialisation deadlocks.
#
#   tools/kit_boot_watchdog.sh -- ./isaaclab.sh -p script.py --headless
#   tools/kit_boot_watchdog.sh --log run.log -- python -m pytest test_sim/env
#   tools/kit_boot_watchdog.sh --attempts 1 -- ./isaaclab.sh -p one_shot.py
#
# A Kit process on Isaac Sim 6.0.1.0 / aarch64 intermittently deadlocks inside
# carb initialisation: two threads park in futex_wait_queue, resident size stays
# near 47 MB, no CUDA context is ever opened, and stdout stops after the
# launcher's own first line. It never recovers. Isaac Sim 6.0.0.0 does not do
# this, so on the earlier pin this wrapper never fires and costs one poll a
# second.
#
# Detection is by progress, not by a runtime bound: a deadlocked tree
# accumulates no CPU time and writes no output, while every healthy phase of a
# run — plugin load, import, collection, stepping — does both continuously. A
# tree that moves neither counter for --stall-window seconds has deadlocked.
# That rule holds inside a pytest subprocess too, where the boot is preceded by
# imports heavy enough to defeat any resident-size threshold.
#
# Watching stops after --boot-window, so only a launch that never got started
# can be relaunched and a long run that is doing real work is never touched.
# What that leaves catchable is a stall whose last progress fell at or before
# (--boot-window minus --stall-window); the kill lands --stall-window later.
# With the defaults: a launch that never moves is killed at about 60 s, and
# anything still moving at 60 s is left alone for the rest of its life.
#
# Each attempt prints one accounting line carrying the resident size, so a
# deadlock is identifiable against the recorded signature rather than merely
# timed out.
#
# With --log FILE, every attempt is kept at FILE.attemptN and FILE holds the
# last one — the winner if there was one. Without it nothing is left behind.
set -uo pipefail

ATTEMPTS=3
STALL_WINDOW=60
BOOT_WINDOW=120
POLL=1
LOG=""
LABEL=""
next=""

usage() {
    echo "usage: $(basename "$0") [--attempts N] [--stall-window S] [--boot-window S]" >&2
    echo "                          [--poll S] [--log FILE] [--label NAME] -- <command> [args...]" >&2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --attempts)     ATTEMPTS="$2"; shift 2 ;;
        --stall-window) STALL_WINDOW="$2"; shift 2 ;;
        --boot-window)  BOOT_WINDOW="$2"; shift 2 ;;
        --poll)         POLL="$2"; shift 2 ;;
        --log)          LOG="$2"; shift 2 ;;
        --label)        LABEL="$2"; shift 2 ;;
        --help|-h)      usage; exit 0 ;;
        --)             shift; break ;;
        *)              usage; exit 2 ;;
    esac
done
[ $# -gt 0 ] || { usage; exit 2; }

for _n in "$ATTEMPTS" "$STALL_WINDOW" "$BOOT_WINDOW"; do
    case "$_n" in
        ''|*[!0-9]*) echo "$(basename "$0"): --attempts/--stall-window/--boot-window take integers" >&2
                     exit 2 ;;
    esac
done
[ "$ATTEMPTS" -ge 1 ] || { echo "$(basename "$0"): --attempts must be at least 1" >&2; exit 2; }
unset _n

for _n in "$ATTEMPTS" "$STALL_WINDOW" "$BOOT_WINDOW"; do
    case "$_n" in
        ''|*[!0-9]*) echo "$(basename "$0"): --attempts/--stall-window/--boot-window take integers" >&2
                     exit 2 ;;
    esac
done
[ "$ATTEMPTS" -ge 1 ] || { echo "$(basename "$0"): --attempts must be at least 1" >&2; exit 2; }
unset _n

[ -n "$LABEL" ] || LABEL="$(basename "$1")"
TAG="[kit-boot-watchdog $LABEL]"

if [ -n "$LOG" ]; then
    mkdir -p "$(dirname "$LOG")"
    KEEP_LOG=1
else
    LOG="$(mktemp -t kit-boot-watchdog.XXXXXX)"
    KEEP_LOG=0
fi

# A caller that asked for no log gets none, on every exit path including the
# signal one — otherwise an interrupted run leaves its capture in $TMPDIR.
cleanup_logs() {
    [ "$KEEP_LOG" -eq 1 ] || rm -f "$LOG" "${LOG}".attempt*
}
trap cleanup_logs EXIT

# Sum utime+stime over every process in the group. /proc/<pid>/stat is read past
# the last ')' so a command name containing spaces or parentheses cannot shift
# the field offsets; utime and stime are then the 12th and 13th tokens.
group_ticks() {
    local pgid=$1 pid total=0 rest
    for pid in $(pgrep -g "$pgid" 2>/dev/null); do
        rest=$(sed 's/.*) //' "/proc/$pid/stat" 2>/dev/null) || continue
        total=$(( total + $(awk '{print $12 + $13}' <<<"$rest") ))
    done
    echo "$total"
}

group_rss_kb() {
    local pgid=$1 pid total=0 kb
    for pid in $(pgrep -g "$pgid" 2>/dev/null); do
        kb=$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null) || continue
        total=$(( total + ${kb:-0} ))
    done
    echo "$total"
}

# A killed Kit leaves its carb shared-memory segment and named semaphore
# behind. Clear only the pairs whose owning pid is gone, so a concurrent run's
# segments survive.
sweep_dead_carb_shm() {
    local f b pid
    for f in /dev/shm/carb-RStringInternals-*; do
        [ -e "$f" ] || continue
        b=$(basename "$f")
        pid="${b##*-}"
        case "$pid" in ''|*[!0-9]*) continue ;; esac
        kill -0 "$pid" 2>/dev/null || rm -f "/dev/shm/$b" "/dev/shm/sem.$b"
    done
}

# SIGTERM, then a grace period, then SIGKILL — a Kit torn down with SIGKILL
# alone abandons its shared memory and any partly-written output.
kill_group() {
    local pgid=$1 i
    # A stopped group queues every signal but SIGKILL, so continue it first or
    # the graceful half of this escalation is spent waiting on a corpse.
    kill -CONT -- "-$pgid" 2>/dev/null
    kill -TERM -- "-$pgid" 2>/dev/null
    for i in $(seq 1 20); do
        pgrep -g "$pgid" >/dev/null 2>&1 || return 0
        sleep 0.5
    done
    kill -KILL -- "-$pgid" 2>/dev/null
}

# An interrupt must reach the whole Kit tree, not just this wrapper: the
# command runs in its own process group, so without this a Ctrl-C or a caller's
# SIGTERM would leave Kit running and its shared memory held.
CHILD=""
on_signal() {
    [ -n "$CHILD" ] && kill_group "$CHILD"
    sweep_dead_carb_shm
    echo "$TAG interrupted by SIG$1"
    exit $(( 128 + $2 ))
}
trap 'on_signal INT 2' INT
trap 'on_signal TERM 15' TERM

RC=124
STALL_KIND=""
for attempt in $(seq 1 "$ATTEMPTS"); do
    ATTEMPT_LOG="${LOG}.attempt${attempt}"
    : > "$ATTEMPT_LOG"
    START=$(date +%s)

    # Job control puts the command in its own process group, so the whole Kit
    # tree can be signalled at once and the poller can read the group's counters
    # without counting itself. The log is streamed to stdout by a follower
    # started outside that group, so the run is still watchable live; what
    # differs from running the command directly is that stderr is merged into
    # stdout and neither is a terminal.
    #
    # The stdin redirect is not removable: `set -m` suppresses the </dev/null
    # bash otherwise gives a background job, and a wrapped command left reading
    # stdin blocks — or stops on SIGTTIN — with no CPU and no output, which is
    # exactly the deadlock signature below. It would be relaunched to exhaustion.
    #
    # Isaac Lab's launcher prints through an unflushed sys.stdout, so a
    # redirected boot writes nothing at all until its buffer fills. Unbuffered
    # output is what makes the log a usable progress signal.
    set -m
    PYTHONUNBUFFERED=1 "$@" </dev/null >"$ATTEMPT_LOG" 2>&1 &
    CHILD=$!
    set +m
    tail -n +1 -f --pid="$CHILD" "$ATTEMPT_LOG" 2>/dev/null &
    FOLLOWER=$!

    stall_since=$START
    last_ticks=-1
    last_bytes=-1
    STALL_KIND=""
    elapsed=0

    while kill -0 "$CHILD" 2>/dev/null; do
        sleep "$POLL"
        kill -0 "$CHILD" 2>/dev/null || break

        now=$(date +%s)
        # Past the boot window the launch is under way; stop watching and let
        # the command run for as long as it needs.
        [ $(( now - START )) -gt "$BOOT_WINDOW" ] && break

        ticks=$(group_ticks "$CHILD")
        bytes=$(stat -c%s "$ATTEMPT_LOG" 2>/dev/null || echo 0)

        if [ "$ticks" != "$last_ticks" ] || [ "$bytes" != "$last_bytes" ]; then
            last_ticks=$ticks
            last_bytes=$bytes
            stall_since=$now
            continue
        fi

        [ $(( now - stall_since )) -lt "$STALL_WINDOW" ] && continue

        rss=$(group_rss_kb "$CHILD")
        elapsed=$(( now - START ))
        STALL_KIND="boot"
        [ "$attempt" -lt "$ATTEMPTS" ] && next="relaunching" || next="no attempts left"
        echo "$TAG attempt $attempt: STALLED during boot at ${elapsed}s" \
             "(rss=${rss}kB, no CPU or output for ${STALL_WINDOW}s) -> ${next}"
        kill_group "$CHILD"
        sweep_dead_carb_shm
        break
    done

    wait "$CHILD" 2>/dev/null; CHILD_RC=$?
    wait "$FOLLOWER" 2>/dev/null

    if [ -n "$STALL_KIND" ]; then
        RC=124
        # Leave the caller a log to read even though no attempt succeeded.
        cp "$ATTEMPT_LOG" "$LOG"
        continue
    fi

    RC=$CHILD_RC
    cp "$ATTEMPT_LOG" "$LOG"
    echo "$TAG attempt $attempt: exit=$RC wall=$(( $(date +%s) - START ))s"
    break
done

[ "$STALL_KIND" = "boot" ] && [ "$ATTEMPTS" -gt 1 ] && \
    echo "$TAG every one of $ATTEMPTS attempts stalled during boot"

exit "$RC"

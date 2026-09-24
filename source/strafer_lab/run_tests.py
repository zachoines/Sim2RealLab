"""Run strafer_lab test suites with clean output.

Isaac Sim floods stdout/stderr with initialization logs, and the root
conftest's os._exit() kills the process before pytest prints its summary.
This script captures results via --junit-xml and prints a clean report.

Usage:
    cd C:\\Worspace\\IsaacLab
    isaaclab.bat -p C:\\Worspace\\source\\strafer_lab\\run_tests.py [suite ...]

Suites: terminations, events, commands, observations, curriculums,
        rewards, sensors, actions, env, noise_models, depth_noise, imu,
        obs_dump, all

Examples:
    isaaclab.bat -p ...\\run_tests.py terminations
    isaaclab.bat -p ...\\run_tests.py rewards observations
    isaaclab.bat -p ...\\run_tests.py all
"""

import importlib.util
import itertools
import signal
import subprocess
import sys
import os
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

TEST_ROOT = Path(__file__).parent / "test_sim"
XML_DIR = Path(__file__).parent  # directory for junit-xml files


# Seconds a timed-out suite gets to shut down after SIGTERM before it is killed.
# Isaac Sim releases its carb shared memory and named semaphores on the way out;
# SIGKILL alone abandons both, and the litter accumulates across runs.
TERM_GRACE = 20
# Every suite boots Kit, so every suite goes through the boot watchdog: on
# Isaac Sim 6.0.0.0 it is a no-op that costs one poll a second, and on 6.0.1.0
# it turns a boot that never starts into a relaunch plus an accounting line.
# Absent or not executable, suites run unwrapped rather than not at all.
WATCHDOG = Path(__file__).resolve().parents[2] / "tools" / "kit_boot_watchdog.sh"
WATCHDOG_ATTEMPTS = 3
WATCHDOG_STALL_WINDOW = 60
WATCHDOG_POLL = 1
# The wrapper's own SIGTERM grace before it escalates to SIGKILL. A child that
# ignores SIGTERM costs the whole of it, so the budget below has to include it.
WATCHDOG_KILL_GRACE = 10

# A suite's timeout has to cover what the watchdog may spend before the suite's
# own work starts, or a stall it could have recovered from is killed as a
# timeout instead. Only the losing attempts count: the winning one runs the
# suite, which is what the per-suite timeout already budgets for.
WATCHDOG_BUDGET = (WATCHDOG_ATTEMPTS - 1) * (
    WATCHDOG_POLL + WATCHDOG_STALL_WINDOW + WATCHDOG_KILL_GRACE + 2
)

# Per-suite timeout in seconds
# depth_noise files each need their own process (SimulationContext singleton)
# noise_models has 55 tests with GPU observation collection
SUITE_TIMEOUTS = {
    "depth_noise": 300,   # per-file timeout (3 files run sequentially)
    "noise_models": 900,  # 55 tests with heavy GPU work
    "actions": 300,
    "env": 300,
    "rewards": 300,       # per-file timeout (collision tests run physics)
    "obs_dump": 300,      # brings up a depth camera env
    "camera_jitter": 300,  # brings up an enriched depth camera env
    "command_markers": 300,  # brings up an enriched depth camera env
}
DEFAULT_TIMEOUT = 180

# How many trailing non-blank output lines to surface when a suite's JUnit XML
# is missing, so a genuine crash is diagnosable instead of opaque.
_CRASH_TAIL_LINES = 30

# Suites that must run each test file in a separate subprocess because
# each file creates its own ManagerBasedRLEnv (SimulationContext singleton).
MULTI_PROCESS_SUITES = {"depth_noise", "rewards", "imu"}

SUITES = {
    "terminations":  [str(TEST_ROOT / "terminations" / "test_terminations.py")],
    "events":        [str(TEST_ROOT / "events" / "test_events.py")],
    "commands":      [str(TEST_ROOT / "commands" / "test_commands.py")],
    "observations":  [str(TEST_ROOT / "observations" / "test_obs_functions.py")],
    "curriculums":   [str(TEST_ROOT / "curriculums" / "test_curriculums.py")],
    "rewards":       [str(TEST_ROOT / "rewards" / "test_rewards.py"),
                      str(TEST_ROOT / "rewards" / "test_collision_rewards.py")],
    "sensors":       [str(TEST_ROOT / "sensors" / "test_observations.py")],
    "actions":       [str(TEST_ROOT / "actions")],
    "env":           [str(TEST_ROOT / "env")],
    "noise_models":  [str(TEST_ROOT / "noise_models")],
    "depth_noise":   [
        str(TEST_ROOT / "sensors" / "depth_noise" / "test_gaussian.py"),
        str(TEST_ROOT / "sensors" / "depth_noise" / "test_holes.py"),
        str(TEST_ROOT / "sensors" / "depth_noise" / "test_frame_drops.py"),
    ],
    "imu":           [str(TEST_ROOT / "sensors" / "test_imu.py"),
                      str(TEST_ROOT / "sensors" / "test_imu_collision.py")],
    "obs_dump":      [str(TEST_ROOT / "bridge" / "test_obs_dump_terms.py")],
    "camera_jitter": [str(TEST_ROOT / "sensors" / "test_d555_camera_prim_jitter.py")],
    "command_markers": [str(TEST_ROOT / "sensors" / "test_command_markers.py")],
}


def _terminate(proc: subprocess.Popen) -> None:
    """SIGTERM the suite's process group, then SIGKILL what is left.

    The suite runs in its own session, so the group signal reaches Kit and not
    just the process that was spawned. Both waits are bounded: a child that
    ignores SIGTERM must not hang the runner that is trying to end it. Isaac Sim
    releases its shared memory and named semaphores on the way out, which a
    straight SIGKILL abandons.
    """
    escalation = ((signal.SIGTERM, TERM_GRACE), (getattr(signal, "SIGKILL", signal.SIGTERM), 10))
    for sig, wait in escalation:
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            proc.send_signal(sig)
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _preserve_failing_xml(xml_path: Path) -> Path | None:
    """Copy a failing suite's JUnit XML aside so the next run cannot erase it.

    Suites write to a fixed per-suite path, so re-running a suite overwrites the
    evidence of why it failed. The copy carries a timestamp, and a counter for
    the case where two failures land in the same second. The name is claimed
    with O_CREAT | O_EXCL, so the check and the claim are one operation and two
    runs preserving at the same moment cannot land on the same file.

    Nothing prunes these. They are gitignored by the same `test_results*.xml`
    rule as the live files, and a tree that has failed often enough to notice
    wants clearing by hand.
    """
    if not xml_path.is_file():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for n in itertools.count():
        suffix = f"-FAILRUN-{stamp}" + (f"-{n}" if n else "")
        kept = xml_path.with_name(f"{xml_path.stem}{suffix}{xml_path.suffix}")
        try:
            fd = os.open(kept, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        shutil.copy2(xml_path, kept)
        return kept




def _wrap(cmd: list[str], label: str) -> list[str]:
    """Put the boot watchdog in front of a suite command, if it is runnable."""
    if not (WATCHDOG.is_file() and os.access(WATCHDOG, os.X_OK)):
        return cmd
    return [str(WATCHDOG),
            "--label", label,
            "--attempts", str(WATCHDOG_ATTEMPTS),
            "--stall-window", str(WATCHDOG_STALL_WINDOW),
            "--poll", str(WATCHDOG_POLL),
            "--"] + cmd


def _timeout_for(name: str, cmd: list[str]) -> int:
    """The suite's own budget, plus the watchdog's worst case when wrapped."""
    base = SUITE_TIMEOUTS.get(name, DEFAULT_TIMEOUT)
    return base + WATCHDOG_BUDGET if cmd and cmd[0] == str(WATCHDOG) else base


def _watchdog_lines(captured: str) -> list[str]:
    """The watchdog's per-attempt accounting lines, in order.

    pytest's progress bar ends without a newline, so an accounting line can
    arrive glued to it; keep only the part from the marker on.
    """
    marker = "[kit-boot-watchdog"
    return [ln[ln.index(marker):].strip()
            for ln in captured.splitlines() if marker in ln]


def _relaunches(result: dict) -> int:
    """How many times the watchdog actually relaunched this suite's Kit boot.

    The attempt that exhausts the budget also reports a stall, but nothing is
    relaunched after it, so only the lines that say so are counted.
    """
    return sum(1 for ln in result.get("watchdog", []) if ln.endswith("-> relaunching"))




def _run_subprocess(cmd: list[str], timeout: int, xml_path: Path) -> dict:
    """Run a pytest subprocess, redirect output to temp files, return parsed results.

    Uses temp files for stdout/stderr instead of PIPE to avoid pipe-buffer
    deadlocks with Isaac Sim's heavy output.

    Every outcome — a clean run, a timeout, a crash before the XML was written —
    comes back as a result dict, so no path can report an absent result as a pass.
    """
    # Suites write to a fixed per-suite path. A run that dies before pytest
    # writes results would otherwise parse the file left there by the previous
    # run and report its counts as this run's — a crashed suite reading green.
    # Removing it first turns that case into the "XML not generated" error the
    # parse path already handles.
    xml_path.unlink(missing_ok=True)

    # Use temp files so the child process never blocks on pipe buffers
    captured = ""
    with tempfile.TemporaryFile(mode="w+b") as tmp_out, \
         tempfile.TemporaryFile(mode="w+b") as tmp_err:

        # A session of its own is what lets a timed-out suite be signalled as a
        # group; POSIX only, and the rest of this function degrades to plain
        # process signalling where it is unavailable.
        proc = subprocess.Popen(cmd, stdout=tmp_out, stderr=tmp_err,
                                start_new_session=(os.name == "posix"))

        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate(proc)
            timed_out = True
        except KeyboardInterrupt:
            # The suite runs in its own session, so Ctrl-C no longer reaches it
            # the way it did when it shared this process's group. Tear it down
            # here, or Kit outlives the runner that was watching it.
            _terminate(proc)
            raise

        # Read the child's output before the temp files are released, so a
        # missing-XML run can be diagnosed below rather than reported opaquely.
        tmp_out.seek(0)
        tmp_err.seek(0)
        captured = (tmp_out.read().decode("utf-8", "replace")
                    + tmp_err.read().decode("utf-8", "replace"))

    if timed_out:
        # Keep whatever the run produced before it was cut off.
        kept = _preserve_failing_xml(xml_path)
        details = [f"  ERROR  TIMEOUT after {timeout}s"]
        if kept is not None:
            details.append(f"  KEPT   {kept.name}")
        return {"tests": 0, "passed": 0, "failed": 0, "errors": 1, "skipped": 0,
                "watchdog": _watchdog_lines(captured),
                "details": details}

    # Parse XML results (written before os._exit kills the process)
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError):
        # No XML means the pytest subprocess died before writing it (a
        # collection-time import error, a segfault, a plugin-autoload crash, or
        # a force-exit that preempted the writer). Count it as an error — never
        # let an absent result read as a silent pass — and surface the tail of
        # the captured output so the crash is diagnosable instead of opaque.
        tail = [ln[:300] for ln in captured.splitlines() if ln.strip()][-_CRASH_TAIL_LINES:]
        details = ["  ERROR  XML not generated (subprocess crashed before writing results)"]
        details += [f"         | {ln}" for ln in tail] if tail else \
                   ["         | (no output captured)"]
        # A parse error means a partial file IS on disk, and it is the most
        # useful thing the run left behind; the next run's unlink would destroy
        # it. FileNotFoundError leaves nothing to keep and this is a no-op.
        kept = _preserve_failing_xml(xml_path)
        if kept is not None:
            details.append(f"  KEPT   {kept.name}")
        return {"tests": 0, "passed": 0, "failed": 0,
                "errors": 1, "skipped": 0,
                "watchdog": _watchdog_lines(captured),
                "details": details}

    root = tree.getroot()
    suite = root.find(".//testsuite")
    if suite is None:
        kept = _preserve_failing_xml(xml_path)
        details = ["  ERROR  No testsuite in XML"]
        if kept is not None:
            details.append(f"  KEPT   {kept.name}")
        return {"tests": 0, "passed": 0, "failed": 0,
                "errors": 1, "skipped": 0,
                "watchdog": _watchdog_lines(captured),
                "details": details}

    total = int(suite.get("tests", 0))
    errors = int(suite.get("errors", 0))
    failures = int(suite.get("failures", 0))
    skipped = int(suite.get("skipped", 0))
    passed = total - errors - failures - skipped

    details = []
    for tc in root.iter("testcase"):
        tc_name = tc.get("name", "?")
        tc_time = tc.get("time", "")
        err = tc.find("error")
        fail = tc.find("failure")
        skip = tc.find("skipped")
        if err is not None:
            msg = err.get("message", "")[:120]
            details.append(f"  ERROR  {tc_name}  ({msg})")
        elif fail is not None:
            msg = fail.get("message", "")[:120]
            details.append(f"  FAIL   {tc_name}  ({msg})")
        elif skip is not None:
            details.append(f"  SKIP   {tc_name}")
        else:
            details.append(f"  PASS   {tc_name}  ({tc_time}s)")

    if total == 0 and failures == 0 and errors == 0:
        # Zero collected with nothing recorded is the false-green shape — a
        # suite whose tests silently stopped matching/collecting. Fail loud.
        errors = 1
        details.append("  ERROR  0 tests collected")

    if failures or errors:
        kept = _preserve_failing_xml(xml_path)
        if kept is not None:
            details.append(f"  KEPT   {kept.name}")

    return {"tests": total, "passed": passed, "failed": failures,
            "errors": errors, "skipped": skipped,
            "watchdog": _watchdog_lines(captured),
            "details": details}


def run_suite(name: str, paths: list[str]) -> dict:
    """Run a test suite and return parsed results.

    For multi-process suites (e.g., depth_noise), each file is run in its
    own subprocess because they each create a ManagerBasedRLEnv and Isaac Sim
    only allows one SimulationContext per process.
    """
    timeout = SUITE_TIMEOUTS.get(name, DEFAULT_TIMEOUT)

    if name in MULTI_PROCESS_SUITES:
        return _run_multi_process(name, paths, timeout)
    return _run_single_process(name, paths, timeout)


def _run_single_process(name: str, paths: list[str], timeout: int) -> dict:
    """Run all test paths in a single pytest subprocess."""
    xml_path = XML_DIR / f"test_results_{name}.xml"
    cmd = _wrap([
        sys.executable, "-m", "pytest",
        *paths,
        "--tb=short",
        "-q",
        f"--junit-xml={xml_path}",
    ], name)

    timeout = _timeout_for(name, cmd)
    result = _run_subprocess(cmd, timeout, xml_path)
    result["name"] = name
    return result


def _run_multi_process(name: str, paths: list[str], per_file_timeout: int) -> dict:
    """Run each test file in its own subprocess and merge results.

    Required for suites where each file creates its own SimulationContext.
    """
    merged = {
        "name": name,
        "tests": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "watchdog": [],
        "details": [],
    }

    for i, path in enumerate(paths, 1):
        file_label = Path(path).stem
        xml_path = XML_DIR / f"test_results_{name}_{file_label}.xml"
        cmd = _wrap([
            sys.executable, "-m", "pytest",
            path,
            "--tb=short",
            "-q",
            f"--junit-xml={xml_path}",
        ], f"{name}:{file_label}")
        per_file_timeout = _timeout_for(name, cmd)

        print(f"    [{i}/{len(paths)}] {file_label} ...", end=" ")
        sys.stdout.flush()

        result = _run_subprocess(cmd, per_file_timeout, xml_path)

        merged["tests"] += result["tests"]
        merged["passed"] += result["passed"]
        merged["failed"] += result["failed"]
        merged["errors"] += result["errors"]
        merged["skipped"] += result["skipped"]
        merged["details"].extend(f"  {file_label}: {ln.strip()}"
                                 if ln.startswith("  ERROR  TIMEOUT") else ln
                                 for ln in result["details"])
        merged["watchdog"].extend(result.get("watchdog", []))

        status = "ok" if result["failed"] == 0 and result["errors"] == 0 else "FAIL"
        print(f"{status} ({result['passed']}/{result['tests']})")

        sys.stdout.flush()

    return merged


def main():
    # Every suite runs ``sys.executable -m pytest``. If THIS interpreter has no
    # pytest (e.g. the script was launched with conda base instead of the Isaac
    # Sim env), every suite would otherwise die with an opaque "No module named
    # pytest" and no JUnit XML. Fail fast with one actionable message.
    if importlib.util.find_spec("pytest") is None:
        print(f"[run_tests] '{sys.executable}' has no pytest — relaunch with the "
              f"Isaac Sim env's python ('make test-lab' / 'make test-lab-pure' "
              f"set this up for you).")
        sys.exit(1)

    args = sys.argv[1:]
    if not args or "all" in args:
        selected = list(SUITES.items())
    else:
        selected = []
        for a in args:
            if a in SUITES:
                selected.append((a, SUITES[a]))
            else:
                print(f"Unknown suite: {a}")
                print(f"Available: {', '.join(SUITES.keys())}, all")
                sys.exit(1)

    grand_total = 0
    grand_passed = 0
    grand_failed = 0
    grand_errors = 0
    grand_relaunches = 0
    results = []

    for i, (name, paths) in enumerate(selected, 1):
        print(f"\n{'='*60}")
        print(f" [{i}/{len(selected)}] Running: {name}")
        print(f"{'='*60}")
        sys.stdout.flush()

        result = run_suite(name, paths)
        results.append(result)

        grand_total += result["tests"]
        grand_passed += result["passed"]
        grand_failed += result["failed"]
        grand_errors += result["errors"]
        grand_relaunches += _relaunches(result)

        # The watchdog's accounting first: a suite that had to be relaunched
        # says so here, and a suite that did not leaves the count at zero —
        # which is the reading that shows the wrapper cost nothing.
        for line in result.get("watchdog", []):
            print(f"  {line}")

        # Print per-test details
        for line in result["details"]:
            print(line)

        status = "PASS" if result["failed"] == 0 and result["errors"] == 0 else "FAIL"
        print(f"\n  {status}: {result['passed']}/{result['tests']} passed", end="")
        if result["failed"]:
            print(f", {result['failed']} failed", end="")
        if result["errors"]:
            print(f", {result['errors']} errors", end="")
        print()
        sys.stdout.flush()

    # Summary
    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    print(f"{'Suite':<20} {'Tests':>6} {'Pass':>6} {'Fail':>6} {'Err':>6} {'Relaunch':>9}")
    print(f"{'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*9}")
    for r in results:
        mark = "+" if r["failed"] == 0 and r["errors"] == 0 else "X"
        print(f"{mark} {r['name']:<18} {r['tests']:>6} {r['passed']:>6} {r['failed']:>6} "
              f"{r['errors']:>6} {_relaunches(r):>9}")
    print(f"{'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*9}")
    print(f"  {'TOTAL':<18} {grand_total:>6} {grand_passed:>6} {grand_failed:>6} "
          f"{grand_errors:>6} {grand_relaunches:>9}")

    all_pass = grand_failed == 0 and grand_errors == 0
    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILURES'}")
    sys.stdout.flush()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

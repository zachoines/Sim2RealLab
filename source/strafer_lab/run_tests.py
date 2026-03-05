"""Run strafer_lab test suites with clean output.

Isaac Sim floods stdout/stderr with initialization logs, and the root
conftest's os._exit() kills the process before pytest prints its summary.
This script captures results via --junit-xml and prints a clean report.

Usage:
    cd C:\\Worspace\\IsaacLab
    isaaclab.bat -p C:\\Worspace\\source\\strafer_lab\\run_tests.py [suite ...]

Suites: terminations, events, commands, observations, curriculums,
        rewards, sensors, actions, env, noise_models, depth_noise, imu, all

Examples:
    isaaclab.bat -p ...\\run_tests.py terminations
    isaaclab.bat -p ...\\run_tests.py rewards observations
    isaaclab.bat -p ...\\run_tests.py all
"""

import subprocess
import sys
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

TEST_ROOT = Path(__file__).parent / "test"
XML_DIR = Path(__file__).parent  # directory for junit-xml files

# Per-suite timeout in seconds
# depth_noise files each need their own process (SimulationContext singleton)
# noise_models has 55 tests with GPU observation collection
SUITE_TIMEOUTS = {
    "depth_noise": 300,   # per-file timeout (3 files run sequentially)
    "noise_models": 900,  # 55 tests with heavy GPU work
    "actions": 300,
    "env": 300,
}
DEFAULT_TIMEOUT = 180

# Suites that must run each test file in a separate subprocess because
# each file creates its own ManagerBasedRLEnv (SimulationContext singleton).
MULTI_PROCESS_SUITES = {"depth_noise"}

SUITES = {
    "terminations":  [str(TEST_ROOT / "terminations" / "test_terminations.py")],
    "events":        [str(TEST_ROOT / "events" / "test_events.py")],
    "commands":      [str(TEST_ROOT / "commands" / "test_commands.py")],
    "observations":  [str(TEST_ROOT / "observations" / "test_obs_functions.py")],
    "curriculums":   [str(TEST_ROOT / "curriculums" / "test_curriculums.py")],
    "rewards":       [str(TEST_ROOT / "rewards" / "test_rewards.py")],
    "sensors":       [str(TEST_ROOT / "sensors" / "test_observations.py")],
    "actions":       [str(TEST_ROOT / "actions")],
    "env":           [str(TEST_ROOT / "env")],
    "noise_models":  [str(TEST_ROOT / "noise_models")],
    "depth_noise":   [
        str(TEST_ROOT / "sensors" / "depth_noise" / "test_gaussian.py"),
        str(TEST_ROOT / "sensors" / "depth_noise" / "test_holes.py"),
        str(TEST_ROOT / "sensors" / "depth_noise" / "test_frame_drops.py"),
    ],
    "imu":           [str(TEST_ROOT / "sensors" / "test_imu.py")],
}


def _run_subprocess(cmd: list[str], timeout: int, xml_path: Path) -> dict | None:
    """Run a pytest subprocess, redirect output to temp files, return parsed results.

    Uses temp files for stdout/stderr instead of PIPE to avoid Windows pipe
    buffer deadlocks with Isaac Sim's heavy output.

    Returns:
        Parsed result dict, or None on timeout/XML failure.
    """
    # Use temp files so the child process never blocks on pipe buffers
    with tempfile.TemporaryFile(mode="w+b") as tmp_out, \
         tempfile.TemporaryFile(mode="w+b") as tmp_err:

        proc = subprocess.Popen(cmd, stdout=tmp_out, stderr=tmp_err)

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return None  # caller handles timeout

    # Parse XML results (written before os._exit kills the process)
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError):
        return {"tests": 0, "passed": 0, "failed": 0,
                "errors": 0, "skipped": 0, "details": ["  ERROR  XML not generated"]}

    root = tree.getroot()
    suite = root.find(".//testsuite")
    if suite is None:
        return {"tests": 0, "passed": 0, "failed": 0,
                "errors": 0, "skipped": 0, "details": ["  ERROR  No testsuite in XML"]}

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

    return {"tests": total, "passed": passed, "failed": failures,
            "errors": errors, "skipped": skipped, "details": details}


def run_suite(name: str, paths: list[str]) -> dict:
    """Run a test suite and return parsed results.

    For multi-process suites (e.g., depth_noise), each file is run in its
    own subprocess because they each create a ManagerBasedRLEnv and Isaac Sim
    only allows one SimulationContext per process.
    """
    timeout = SUITE_TIMEOUTS.get(name, DEFAULT_TIMEOUT)

    if name in MULTI_PROCESS_SUITES:
        return _run_multi_process(name, paths, timeout)
    else:
        return _run_single_process(name, paths, timeout)


def _run_single_process(name: str, paths: list[str], timeout: int) -> dict:
    """Run all test paths in a single pytest subprocess."""
    xml_path = XML_DIR / f"test_results_{name}.xml"
    cmd = [
        sys.executable, "-m", "pytest",
        *paths,
        "--tb=short",
        "-q",
        f"--junit-xml={xml_path}",
    ]

    result = _run_subprocess(cmd, timeout, xml_path)
    if result is None:
        return {"name": name, "tests": 0, "passed": 0, "failed": 0,
                "errors": 1, "skipped": 0,
                "details": [f"  ERROR  TIMEOUT after {timeout}s"]}

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
        "details": [],
    }

    for i, path in enumerate(paths, 1):
        file_label = Path(path).stem
        xml_path = XML_DIR / f"test_results_{name}_{file_label}.xml"
        cmd = [
            sys.executable, "-m", "pytest",
            path,
            "--tb=short",
            "-q",
            f"--junit-xml={xml_path}",
        ]

        print(f"    [{i}/{len(paths)}] {file_label} ...", end=" ")
        sys.stdout.flush()

        result = _run_subprocess(cmd, per_file_timeout, xml_path)

        if result is None:
            merged["errors"] += 1
            merged["details"].append(
                f"  ERROR  {file_label}  (TIMEOUT after {per_file_timeout}s)"
            )
            print("TIMEOUT")
        else:
            merged["tests"] += result["tests"]
            merged["passed"] += result["passed"]
            merged["failed"] += result["failed"]
            merged["errors"] += result["errors"]
            merged["skipped"] += result["skipped"]
            merged["details"].extend(result["details"])

            status = "ok" if result["failed"] == 0 and result["errors"] == 0 else "FAIL"
            print(f"{status} ({result['passed']}/{result['tests']})")

        sys.stdout.flush()

    return merged


def main():
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
    print(f"{'Suite':<20} {'Tests':>6} {'Pass':>6} {'Fail':>6} {'Err':>6}")
    print(f"{'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for r in results:
        mark = "+" if r["failed"] == 0 and r["errors"] == 0 else "X"
        print(f"{mark} {r['name']:<18} {r['tests']:>6} {r['passed']:>6} {r['failed']:>6} {r['errors']:>6}")
    print(f"{'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    print(f"  {'TOTAL':<18} {grand_total:>6} {grand_passed:>6} {grand_failed:>6} {grand_errors:>6}")

    all_pass = grand_failed == 0 and grand_errors == 0
    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILURES'}")
    sys.stdout.flush()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

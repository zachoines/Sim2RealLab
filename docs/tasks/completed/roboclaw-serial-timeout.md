# Read and set the RoboClaws' serial timeout, so the controllers stop the motors when the link goes quiet

**Status:** Shipped 2026-10-17 in `cb2ea64` and 2026-10-18 in `455e2d9` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/249

**Type:** task (a controller safety setting the tree never sets, reads or documents)
**Owner:** Jetson (`diagnose_roboclaw.py`, `strafer_driver`, `strafer_shared`)
**Priority:** P1 for the chassis bring-up. The driver's 0.5 s `cmd_vel` watchdog is software, so it
stops with the node. After a SIGKILL (Docker's after a stop, or the OOM killer), a native crash, a
hung node or a dropped USB link, only the controllers' own packet-serial timeout can stop the
motors. Each controller's value is held in its flash, and neither has been read.
**Estimate:** S
**Branch:** `task/roboclaw-serial-timeout`

## Story

As **whoever brings up the chassis**, I want **each RoboClaw's packet-serial timeout read, and set
to 0.5 s and saved on each controller**, so that **the controllers stop the motors themselves when
the serial traffic stops, whatever happened to the program driving them.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)
- [`chassis-driver-port-fence-2026-10-14`](../../measurements/chassis-driver-port-fence-2026-10-14/README.md)
  §7: the on-site check this step now joins.
- [`jetson-untether`](../active/reliability/jetson-untether.md), "Camera, RoboClaw and boot hazards" and its boxes on the
  camera's serial interface.

## Context

Code read at `0a975ef`. The protocol is from the RoboClaw user manual, Revision 6.0
(`downloads.basicmicro.com/docs/roboclaw_user_manual.pdf`, sha256 `a0568558…`, fetched
2026-10-17).

- **The setting.**
  - Command 14, Set Serial Timeout: "Sets the serial communication timeout in 100 ms increments. If
    no serial data is received within the specified time, both motors will stop automatically. The
    valid range is 0 to 255 (0 to 25.5 seconds)." Send `[Address, 14, Value, CRC16]`, receive
    `[0xFF]`.
  - Command 15, Read Serial Timeout: send `[Address, 15]`, receive `[Value, CRC16]`.
  - The manual's Timeout setting, the same value in Motion Studio, says 0 disables it.
- **Only command 94 saves it.** Command 94, Write Settings to NVM: "Writes all configuration
  settings to non-volatile flash memory. Saved values are automatically loaded at power-up.
  Settings are only written to flash when this command is issued." It takes the 4-byte key
  `0xE22EAB7A`, and "will cause a unit reset after settings have been written". So command 14
  changes the active value until the next power-up, and a save saves every active setting.
  The reset should also re-enumerate the controller's USB port; that is an inference, not yet seen
  on these controllers.
- **Command 95 undoes unsaved changes.** Command 95, Read Settings from NVM: "Reloads all
  configuration settings from non-volatile flash memory into active memory, overwriting any unsaved
  changes." The active settings may hold changes nobody meant to save:
  - trial gains from `tune_pid.py --tune` or `--set`;
  - the PID constants the node writes at every start.

  Sending command 95 first means a save carries only the saved settings and the new timeout.
- **The tree never touches it.** At `0a975ef` no file in `source/` or `docs/` sends or names
  command 14 or 15, or mentions the controllers' serial timeout.
- **The node's own stop is software.**
  - `WATCHDOG_TIMEOUT_SEC = 0.5` (`strafer_driver/roboclaw_node.py`:75) zeroes the twist on each
    50 Hz tick after 0.5 s without `cmd_vel` (:325-334). It runs only while the node runs.
  - The node's teardown stops the motors (`destroy_node`, :537-550, from `main`'s `finally`). A
    Python exception still reaches it. On `main` a container stop does not: compose sends SIGTERM,
    which never reaches the node, and Docker's SIGKILL follows (PR #248 changes the stop signal).
  - A SIGKILL, a native crash, a hung node or a dropped USB link skips both.
- **Why the node does not write it.** It is a bring-up setting, held in the controllers' flash and
  loaded at power-up, and it does not change at runtime. Saved once, it covers every program that
  drives the controllers, the bench tools included. A write at node start would cover only the
  node, and only within one power cycle.
- **Which tools a 0.5 s timeout affects.**
  - `diagnose_roboclaw.py`'s duty test holds each motor for 1 s with no serial traffic, so the
    controller stops the motor at 0.5 s. The tool says so, and on site that stop shows the timeout
    working.
  - `tune_pid.py` drives a motor only inside its step response, which reads the speed every 5 ms
    through the 0.5 s step (`SAMPLE_PERIOD_S`, `STEP_DURATION_S`), and stops with `hard_stop()`
    before each pause.
  - The node sends a drive packet to each controller every 20 ms.

## Acceptance criteria

- [x] **`diagnose_roboclaw.py` prints the timeout, read by command 15, for each controller it
      diagnoses**, with the bring-up's expected value beside any other reading. `--read-only`
      reads without the duty test, so the read drives nothing.
      *2026-10-17, in `cb2ea64`: after the temperature read, "Serial timeout (cmd 15): …".*
      *2026-10-18: command 15 reads the active value only, so a read alone cannot show what is
      saved; the on-site step reads again after a power cycle.*
- [x] **It sets the timeout only behind `--set-serial-timeout <seconds>`, and a second read echoes
      the value back.** The value must be a whole number of tenths from 0 to 25.5 s; anything else
      is a usage error, so no request rounds to another value. A set that is not acknowledged, or
      reads back different, ends that controller's run there. Nothing is written after the
      read-back, and the tool exits 1. A set that takes, without `--write-nvm`, sends no motor
      command.
      *2026-10-17, in `cb2ea64`, where a set that took went on to the duty test.*
      *2026-10-18, in `455e2d9`: it no longer does.*
- [x] **Saving is a separate, explicit flag** *(added 2026-10-17, since command 14 alone is lost at
      power-up)*. `--write-nvm` is accepted only with `--set-serial-timeout`. It works in this
      order:
      - it first reloads the saved settings (command 95), and stops if that is not acknowledged;
      - it skips the set and the save when the value already matches;
      - otherwise it sets and reads back, then sends command 94 only after a matching read-back;
      - then it closes the port: no PID read and no drive follow.

      Running the tool again after the reset reads the saved value.
      *2026-10-17, in `cb2ea64`. Commands 94 and 95 live in the tool only.*
      *2026-10-18: that holds for the same `--write-nvm` line, which reloads the saved settings
      (95) before it reads. A `--read-only` read shows the active value, which an unsaved set
      also leaves at the target, so the on-site step proves the save by a read after a
      main-power cycle. In `455e2d9` the tool's advice after command 94 says to run the same
      command again; it had said `--read-only`.*
- [x] **The duty test always stops its motor.** *(added 2026-10-17.)* A serial error or Ctrl-C
      during a hold still sends that motor duty 0. The tool also refuses one device for both
      controllers, as the node does.
      *2026-10-17, in `cb2ea64`.*
- [x] **It only reads unless given `--drive`** *(added 2026-10-18)*. The duty test runs only
      with `--drive`. Before the first motor packet one line names the wheels, the duty and the
      hold, and says the wheels must be lifted; `--help` says so too. `--read-only`, `--drive` and
      `--set-serial-timeout` exclude each other.
      *2026-10-18, in `455e2d9`: `TestDiagnoseRead`, `TestDiagnoseDrive`.*
- [x] **A port that does not answer fails the run** *(added 2026-10-18)*. A port that opens but
      never answers the battery read, or does not open, exits 1 in every mode. Before, only a
      requested set made either fail, so `--read-only` on a silent port exited 0. A controller
      that auto-detect does not find still fails only a requested set.
      *2026-10-18, in `455e2d9`: `TestDiagnoseUnreached`.*
- [x] **ModemManager leaves the RoboClaws' ports alone** *(added 2026-10-18)*.
      `99-strafer.rules`' 03eb:2404 rule sets `ENV{ID_MM_DEVICE_IGNORE}="1"`, and a test pins it.
      *2026-10-18, in `455e2d9`: `test_udev_rule_names_roboclaws_and_keeps_modemmanager_off`.*
- [x] **Fake-serial tests for the read and for the flagged set.**
      *2026-10-17: `strafer_driver/test/test_serial_timeout.py`, 53 tests on `test_port_fence.py`'s
      fake RoboClaw. The fake now models commands 14, 15, 94 and 95, with a saved and an active
      register, and checks packet lengths.*
      - *Run on `main` (`0a975ef`) with `cb2ea64`'s two test modules copied in, 49 fail. The 4
        that pass are controls: no duty-test note at 0 s or 1.1 s, one controller tested alone,
        and the node sending none of 14, 15, 94 or 95.*
      - *The 27 fence tests pass there.*
      - *At `cb2ea64` the driver suite passes 140 of 140.*
      - *In mutation runs on a throwaway copy, every mutant tried of the reload, save, rounding,
        duty-stop, read-only and one-device paths failed at least one test.*

      *2026-10-18, deposited as the third pass of the port-fence record's deposit:
      Sim2RealLab-Artifacts `0c964910`,
      `chassis-driver-port-fence-2026-10-14/record-files/pass3/`
      ([the record's evidence section](../../measurements/chassis-driver-port-fence-2026-10-14/README.md#evidence)):*
      - *The first three counts above were run again and hold: 49 of the 53 fail on `main` with
        the same 4 controls passing, the 27 fence tests pass there, and `cb2ea64`'s driver suite
        passes 140 of 140.*
      - *The mutation runs above were not kept, and their list was not recorded, so they remain
        unreproduced as stated. The deposited list is `pass3/tests/mutation_runs.txt`.*
      - *At `455e2d9` there are 66 serial-timeout tests and 28 fence tests. On `main` with
        `455e2d9`'s two test modules, 67 of the 154 driver tests fail: 65 serial-timeout tests and
        2 fence tests. On `9489177` with them, 28 fail: 15 test cases the fix round added and 13
        it changed. Every test left unchanged passes. At `455e2d9` the driver suite passes 154 of
        154.*
      - *35 mutations at `455e2d9` each fail at least one test: 23 on the fix round's paths (the
        drive gate, its notice and help, the exit codes, the advice after command 94, the udev
        rule) and 12 on the first round's (reload, set, save, rounding, the duty test's stop, the
        one-device refusal, the node). The first round's read-only path is covered in the first
        group.*
- [x] **`ROBOCLAW_SERIAL_TIMEOUT_S = 0.5`** in `strafer_shared/constants.py`, as the value the
      bring-up expects, with a one-line comment naming the command.
- [x] **The node does not write it.** The reason is under Context, and
      `test_node_never_sends_timeout_or_flash_commands` pins it.
- [x] **On site, for the maintainer.** Do this before `base` first drives the chassis, with both
      RoboClaws connected, their main power on before their USB:
      - read each controller's timeout with `diagnose_roboclaw.py --read-only`. Command 15 reads the
        active value only;
      - run `--set-serial-timeout 0.5 --write-nvm` on each controller, whatever it read. On a
        controller with 0.5 s saved it sets and saves nothing. On exit 1, run the same line again,
        never `--read-only` in its place;
      - read both again after a main-power cycle, with the USB unplugged first, so the read is the
        saved value;
      - optionally, with the wheels lifted, run the duty test (`--drive`);
      - keep the output in a file outside the tree, deposit it with the record of the bring-up
        that runs the step, and record the readings per controller in a dated note on the
        serial-timeout box in `jetson-untether`, citing that deposit.

      The commands, the emergency stop and the plug order's reason are in the port-fence record's
      [§7, step 0](../../measurements/chassis-driver-port-fence-2026-10-14/README.md#7-the-on-site-check).
      Until the values are read, the chassis bring-up carries this step first.
      *2026-10-19, on the robot:
      [`chassis-bringup-2026-10-19`](../../measurements/chassis-bringup-2026-10-19/README.md) §2.*
      - *Both controllers read 0.0 s, then took 0.5 s and saved it.*
      - *Both read 0.5 s after a power cycle with the USB out first. The main-power cut is the
        maintainer's report.*
      - *The readings are in `jetson-untether`'s box.*
- [x] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance).
      *2026-10-17:*
      - *the dated step 0 in the port-fence record's §7;*
      - *a box in `jetson-untether`'s camera-fence group, after the udev box;*
      - *the flags in `source/strafer_ros/README.md`'s bench-tool block.*
- [x] No regression: `make test-ros` passes.
      *2026-10-17: the whole ROS suite at `cb2ea64`, 925 JUnit cases: 913 passed, 11 skipped,
      1 xfailed, 0 failed.*
      *2026-10-18: the same counts at `cb2ea64` again, and at `455e2d9` 939 cases: 927 passed,
      11 skipped, 1 xfailed, 0 failed; both in `pass3/`.*

## Investigation pointers

- `source/strafer_ros/diagnose_roboclaw.py` (`diagnose_roboclaw`, `test_duty_drive`, `main`).
- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_interface.py` (`read_serial_timeout`,
  `set_serial_timeout`).
- `source/strafer_ros/strafer_driver/test/test_port_fence.py` (`FakeRoboClaw`) and
  `test_serial_timeout.py`.

## Out of scope

- **A runtime write by the node.** It would cover only the node, and only within one power cycle.
- **The other controller settings in flash:** current limits, voltage limits, PID gains. Command
  95 keeps a save from adding unsaved changes to them.
- **`docs/WIRING_GUIDE.md` §5**, whose Motion Studio "General Settings → Serial" pane holds the
  same Timeout. A line there is a follow-up for that guide.
- **`strafer_shared/constants.py`'s PID comment.** It says the gains are "written to RAM on every
  startup by driver and test scripts". The node writes them; `tune_pid.py` writes its own
  defaults, and only on `--tune` or `--set`. This is a follow-up.
- **ModemManager on the D555's CDC port.** On the robot host, ModemManager 1.20.0 took up
  `/dev/ttyACM0` (8086:0b56, whose CDC interface reports the AT protocol) as a modem candidate
  each time the camera enumerated: its journal shows "creating modem with plugin 'generic'", then
  "Failed to find primary AT port", on 2026-10-04 and 2026-10-11. It logged nothing when the
  RoboClaws (protocol 00) enumerated on 2026-10-18. The rule covers the RoboClaws only.
- **Error and battery visibility:**
  [`roboclaw-error-visibility-and-low-battery`](../parked/reliability/roboclaw-error-visibility-and-low-battery.md).

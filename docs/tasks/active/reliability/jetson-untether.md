# Untether the robot: bracket the Jetson to the chassis, power it from a LiPo, and reach it over Wi-Fi with both Ethernet cables out

**Type:** task (rig bring-up — compute mounting, onboard power, network)
**Owner:** Jetson + DGX — the host, compose, DDS and driver work is Jetson-lane; the topology docs,
`.env.example` and the repo-root `Makefile` are DGX-lane files
([ownership-boundaries.md](../../context/ownership-boundaries.md):26-28, :36-39).
**Priority:** P2. Some P1 briefs need it, but it is not what blocks them today.
[`domain-randomization-audit`](../trained-policy/domain-randomization-audit.md)'s Phase 1 items 1, 2
and 5 and [`d555-depth-decode-validity`](../trained-policy/d555-depth-decode-validity.md)'s
`inferences` item are also blocked on a chassis bringup and on `d555-l4t-stream-integrity`. The
texture capture that first needed the robot on the floor was taken with the Jetson moved beside the
robot on longer Ethernet runs (PR #233).
**Promote to P1 when [`d555-l4t-stream-integrity`](d555-l4t-stream-integrity.md) closes and a
chassis bringup (RoboClaws connected and driving) is under way:** closed-loop driving on the real
robot needs all three.
**Estimate:** L (an adapter and a regulator to choose and fit, a power measurement, a DDS default
that reaches every entry point, a bracket, then cold-boot verification)
**Branch:** `task/jetson-untether`

## Story

As the **real-robot lane**, I need **the Jetson riding on the chassis, powered from a LiPo and
reachable over Wi-Fi with neither Ethernet cable attached**, so that **the robot can be placed,
carried and eventually driven anywhere in a room instead of within reach of two cables and a mains
supply.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/deploy-env-config.md](../../context/deploy-env-config.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`real-d555-depth-texture-2026-09-26`](../../../measurements/real-d555-depth-texture-2026-09-26/README.md),
  "Setup": how the rig was arranged for the floor capture, and the mount geometry measured from the
  data.
- [`real-d555-hardware-readback-2026-09-25`](../../../measurements/real-d555-hardware-readback-2026-09-25/README.md):
  the camera's USB identity and its CDC interface.

## Context

**What ties the robot down today** (read on the host, 2026-09-27):

- **Compute.** The Jetson (Orin NX 16 GB, L4T R36.4.3) is not attached to the robot. For the
  2026-09-26 capture it was moved next to the robot on longer Ethernet runs (texture record:39-44).
  The D555 is on the robot's mount, and the only cord that works with it is a 6-inch USB cable at
  5000 Mbit/s. A 3 m Thunderbolt cable did not power the camera; a longer run needs a powered
  cable. So the Jetson has to ride within reach of that cord.
- **Network.** Two Ethernet links. `enP8p1s0` runs through the wall-modem adaptor and carries the
  LAN address, the default route and the SSH / VS Code path. `enP7p1s0` is the point-to-point
  cable to the DGX on a private /24, which the deploy stack's DDS is pinned to (`ip -br addr`,
  `ip route`, `nmcli -t -f DEVICE,TYPE,STATE,CONNECTION dev`). `sshd` listens on all addresses
  (`ss -ltn`: `0.0.0.0:22`). The host has no Wi-Fi interface: `iw dev` prints nothing and
  `/sys/class/net` has no `wl*`.
- **Power.** A mains DC supply. The INA3221 reads `VDD_IN` 12.06 V at 0.43–0.44 A, about 5.2 W,
  with perception stopped and the camera unplugged; a burst of shell commands read about 8.4 W.
  No repo file records a full-stack figure.
- **Motors.** No RoboClaw is connected (`lsusb` has no `03eb:2404`).
- At filing the D555 is unplugged (`usb 2-2: USB disconnect`, 2026-09-27 10:56 CDT) and
  `perception` is stopped.

### Wi-Fi: the last adapter wedged, and nothing replaced it

- [`enriched-lane-rig-stability`](enriched-lane-rig-stability.md) §3 (:199-213) records the NETGEAR
  A8000 (`0846:9060`, MT7921, `mt7921u`) going dead for 27 minutes under load until it was
  replugged, with the host still up. The journal shows the whole day, 2026-08-02: five firmware
  timeouts, each followed by a chip reset. Three reconnected within about 11 s; two were outages.
  - The first was at 16:06 CDT on `tegra-xusb`, behind the carrier's USB5744 hub (`2-1.3`).
  - The dongle was then moved to the Renesas uPD720201 PCIe USB controller (`usb 4-2`, 16:51)
    and reset there four more times (17:22, 18:19, 19:21, 21:57). After the last one it never
    reconnected, and it was pulled on 2026-08-04 at 11:22 CDT. It has not enumerated since
    (`journalctl --since 2026-08-02 --until 2026-08-05 | grep -E 'mt7921u|usb [0-9]-'`; that August boot is
    not listed by `journalctl --list-boots`). The other controller did not help.
- The DKMS module `mt7921-a8000/1.0` still builds for `5.15.148-tegra` and loads at boot
  (`dkms status`, `/etc/modules-load.d/mt7921u.conf`). Driver upkeep was kept out of the repo
  ([`nx-docker-bringup`](../../completed/nx-docker-bringup.md):107).
- The saved NetworkManager Wi-Fi profile is bound by interface name to the old adapter's
  `wlx<MAC>` name, with no MAC binding (`nmcli con show <profile>`). A replacement adapter, even
  another A8000, will not pick it up. Power save is on globally
  (`/etc/NetworkManager/conf.d/default-wifi-powersave-on.conf`: `wifi.powersave = 3`), and the
  profile inherits it.
- One PCIe x1 controller (`pcie@14100000`) is enabled with nothing on it: "Phy link never came up"
  (`journalctl -b 0 -k`; `lspci` lists only NVMe, the two NICs and the Renesas controller). That is
  consistent with an empty M.2 Key E slot. Whether this carrier brings one out physically is not
  known.
- `avahi-daemon` publishes the Docker bridge addresses too. On the Jetson,
  `getent ahostsv4 strafer-nx.local` returns `172.17.0.1` (`docker0`), and
  `/etc/avahi/avahi-daemon.conf` has no `allow-interfaces` or `deny-interfaces`.
- The DGX's gitignored `.env` and the tracked `.env.example`:151-152 name the Jetson by an address
  and by user `jetson`. The address was the Jetson's own Wi-Fi DHCP lease in August
  (NetworkManager journal). Whether the router reserves it is unknown; a new adapter has a new MAC,
  so a MAC-bound reservation would not carry over.
- `enriched-lane-rig-stability.md`:226-227 asks to "keep wired as the SSH path regardless". This
  brief reverses that ask, so it has to be superseded there.

### DDS: the stack is pinned to a cable it is about to lose

- **The deploy stack was built for same-host DDS over UDP loopback** (`docker-compose.yml`:3). On
  the real-robot lane the only traffic that crosses hosts is HTTP to the VLM and planner. DDS
  crosses hosts only on the sim lanes.
- **The tracked profile pins no interface.** `strafer_bringup/config/cyclonedds.xml`:29-44 carries
  receive tuning only, so Cyclone chooses an interface itself. Which one it picks on a host with
  `docker0`, a Docker bridge and a Wi-Fi interface was not checked.
- **The pin is host-local.** The untracked, not-gitignored `deploy/docker-compose.override.yml`
  remaps the bind-mount source of `/opt/strafer/config/cyclonedds.xml` to
  `~/.config/cyclonedds/direct-link.xml`. It does so for `base`, `perception`, `slam`,
  `navigation`, `autonomy`, `inference`, `viewer` and `sim-perception` (:33-69), but not for
  `zenoh-bridge`. The profile itself:
  - pins `enP7p1s0` with `presence_required="true"` (:47-48);
  - adds `127.0.0.1` with `multicast="true"` (:54-55). That flag is required: `lo` reports flags
    `0x9`, with no `IFF_MULTICAST`, so without it Cyclone would not run discovery on loopback;
  - copies the tracked `<Internal>` block (:58-72).
- **What the pin costs when the cable goes.** While `enP7p1s0` is absent or has no address, every
  new participant should fail at creation: container starts, `docker exec` probes and
  `ros2 bag record` alike. This is inferred from `presence_required="true"`, which was measured to
  fail creation with a wrong address. It was not measured with the cable out. NetworkManager removes the address on carrier loss. That was seen on 2026-09-25 from
  20:30 to 20:55 CDT, and on 2026-09-27 at 10:56 CDT for 23 s (avahi and NetworkManager journal).
  What already-running participants do when the pinned interface loses carrier has never been
  measured.
- **Most entry points never load the override.** `Makefile`:156 is
  `FULL_COMPOSE := docker compose -f $(DEPLOY_DIR)/docker-compose.yml`. The explicit `-f` means
  `make up` (:193), `make launch-autonomy` (:199), `make viewer` (:190) and `make submit-deploy`
  (:207) never load it. Neither does the cheatsheet's sim-bridge chain
  ([`sim_bridge_autonomy_cheatsheet.md`](../../../sim_bridge_autonomy_cheatsheet.md):56-61,
  :74-79). Making a robot-local profile "the auto-loaded override" therefore does not make it the
  default.
- **What the env check enforces.** `deploy/tests/check_env_sync.py` checks only the
  `CYCLONEDDS_URI` literal (:59) and the bind-mount *target* (:142-150), and only in
  `docker-compose.yml` and `docker-compose.sim.yml` (:81). It never reads the mount source, and it
  cannot see untracked overlays (:83-84;
  [deploy-env-config.md](../../context/deploy-env-config.md):98-100). A loopback-only profile names
  no rig address, so it could be tracked. The direct-link profile names rig addresses and stays
  host-local.
- **The DGX has a matching pin.** Its gitignored `.env` defaults `CYCLONEDDS_URI` to its own
  direct-link profile (`enP7s7`, `presence_required="true"`), so DGX-side ROS participants fail the
  same way when the cable is out. The VLM and planner are served over HTTP (`Makefile`:246-252).
  Whether either one creates a DDS participant was not checked.
- **The sim-bridge lane cannot move to Wi-Fi.** The deployed census needs 65–211 Mbit/s. With the
  DGX's end on Wi-Fi, the path delivered 20–60 Mbit/s loss-free, which gave 0.26 Hz of depth
  against 30.00 Hz on the cable
  ([`sim-bridge-link-transport-capacity`](sim-bridge-link-transport-capacity.md):5-20, :65-99). The
  cable therefore stays, as a re-dock for sim-bridge sessions. That is a requirement of this brief,
  not an option.
- **Zenoh is untested against the new profile.** `zenoh-bridge` deliberately gets no
  `CYCLONEDDS_URI` (`docker-compose.yml`:186-187). Whether it discovers loopback-pinned nodes is
  untested.

### The executor's services over Wi-Fi

- The DGX serves the VLM on `:8100` and the planner on `:8200`, bound to `0.0.0.0`
  (`Makefile`:246-252), at the LAN address its Wi-Fi interface carries. The sim lanes already use
  that address (`deploy/tests/gen_env.py`:63-66).
- The deploy autonomy mirror leaves both URLs empty (`compose/autonomy.env`:10-19;
  `gen_env.py`:85-88). The executor exits 1 while either URL is unset (`executor/main.py`:173-182)
  or unreachable ([`executor-startup-health-check-contract`](executor-startup-health-check-contract.md):17-20).
  URLs are deploy-only keys and must stay out of canon (`check_env_sync.py`:152-160).
- The only host-local file that sets the URLs today is
  `deploy/docker-compose.override.autonomy-local.yml`, gitignored by `.gitignore`:52. It also sets
  two keys that make it wrong for the real robot:
  - `STRAFER_USE_SIM_TIME: "true"` (:74). The comment at :35-39 retires only the copy under the
    commented-out `inference` block;
  - `STRAFER_NAV_BACKEND` (:75). That silently shadows canon `env_autonomy.env`, where
    `check_env_sync` cannot see it.
- Neither service was running on 2026-09-27: health GETs were refused on both paths. The DGX
  answers ping over the LAN at 6–64 ms and over the cable at 1.0–1.8 ms (`ping -c3`).
- A bare `docker compose up` does not start `inference`, which is under `profiles: [ policy ]`
  (`docker-compose.yml`:155). A policy backend in canon therefore needs `--profile policy` on
  whatever boot path is chosen.

### Power

- **The carrier.** `/etc/nv_tegra_release`:5 names the flashed image
  `recomputer-robo-orin-nx-16g-j401-gmsl`, while `/proc/device-tree/model` reads "NVIDIA Jetson
  Orin Nano Seeed recomputer classic Robotics". The carrier's DC input range is in no repo file.
- **The only power doc is for a different board.**
  [`WIRING_GUIDE.md`](../../../WIRING_GUIDE.md):233-239 describes an Orin Nano barrel jack at
  7–20 V and offers 4S direct or 3S plus a boost converter.
- **The repo contradicts itself on cell count.** `WIRING_GUIDE.md` leaves it open: :222 reads "3S
  or 4S", and :237-238 offer 4S direct or 3S plus a boost. The conflict is elsewhere:
  - 3S, or 12 V: the parked low-battery brief (:15; thresholds at :99-101, :116, :128).
  - 4S, 14.0–16.8 V: [`domain-randomization-audit`](../trained-policy/domain-randomization-audit.md):143
    and :197, and `strafer_lab/.../navigation/sim_real_cfg.py`:135, :140.

  The pack choice decides which of these change.
- **The on-board monitor does not see the whole robot.** INA3221 channel 1
  (`/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/in1_*`, `curr1_*`) reads without root. Its
  alert limit `curr1_crit` is 3.32 A, which is an alert setting and not the carrier's rating.
  `VDD_IN` is taken to be the module's input only. That is an inference, not a measurement. If it
  holds, the carrier's USB VBUS (camera and Wi-Fi adapter), the USB hub and the fan sit outside it,
  so a pack budget has to be metered at the carrier's DC input.
- **Power mode and boot target are unmeasured.** `nvpmodel -q` reports `MAXN_SUPER`, a persisted
  choice; the conf's default is mode 4 (40 W). The host boots to `graphical.target` with `gdm`
  running, and the X session registers the D555's HID interface as a keyboard. The power cost of
  either was not measured.
- **Nothing monitors a Jetson pack yet.** The parked
  [`roboclaw-error-visibility-and-low-battery`](../../parked/reliability/roboclaw-error-visibility-and-low-battery.md)
  watches only the motor pack, through the RoboClaw's `read_main_battery` (:96-101). A separate
  Jetson pack falls outside it.

### Camera, RoboClaw and boot hazards

- **Without RoboClaws, `base` writes to the camera.** `99-strafer.rules` is not installed:
  `/etc/udev/rules.d` holds only NVIDIA rules. The sysctl and `modules-load.d` drop-ins from the
  same installer are present (`deploy/host-setup/install-host-prereqs.sh`:17, :22-25, :44). The
  rules would not protect the camera on their own anyway, because the driver never uses them when
  no RoboClaw is attached:
  - With no RoboClaw, `/dev/roboclaw*` is empty. `detect_roboclaws` then falls back to
    `/dev/ttyACM*` (`strafer_driver/roboclaw_interface.py`:415-417) and sends each port a
    `read_main_battery` probe (:367-380).
  - After "no RoboClaws found" the node opens `front_port` (`roboclaw_node.py`:136-157) and writes
    PID configuration to it (:156, :225). `front_port` defaults to `/dev/ttyACM0` at
    `base.launch.py`:30-33, and `driver.launch.py`:22-26, :38-43 passes it after the params file,
    so it beats `driver_params.yaml`:6.
  - With the D555 attached, `/dev/ttyACM0` is the camera's CDC interface (hardware readback
    record:57-58).
  - `base` mounts `/dev`, joins `dialout`, carries the ttyACM cgroup rule
    (`docker-compose.yml`:74-78), and is in the default `up` set. So "Device mounts are inert until
    the hardware is connected" (:32-34) is false for `base` whenever the camera is attached.
- **Nothing starts on boot.** Only `strafer_perception` and `strafer_inference` exist, both
  stopped (`Exited (137)`, restart `unless-stopped`; `docker ps -a`, `docker inspect`). `docker`
  is enabled. A cold boot on battery would start no strafer service.
- **The camera does not come up across a boot.** Every listed boot with a device on the camera's
  root port (`usb 2-2`) at power-on has the same shape: 5 of 5.
  - A SuperSpeed device enumerates in the first seconds and binds no UVC, HID or CDC driver until it
    is unplugged. `cdc_acm` is not even registered.
  - The command is, per boot:
    `journalctl -b <id> -k | grep -E 'usb 2-2|Found UVC|cdc_acm'`.
  - Every hot-plug seen binds normally: 2026-08-04, 2026-08-06, 2026-08-16, 2026-09-25 and
    2026-09-26. Each enumerates first at SuperSpeed Plus, then settles at 5000 Mbit/s.
  - The last boot was 2026-09-10 at 16:57 CDT. The camera stayed driverless from then until the
    replug on 2026-09-25 at 11:59:30 CDT.
  - [`d555-depth-decode-validity`](../trained-policy/d555-depth-decode-validity.md):192-194 records
    the stuck state as `8086:0bdc` "Intel RealSense Generic Device" with no `/dev/video*`, and :200
    records the power cycle that cleared it. No record ties it to the boot. The kernel lines
    themselves carry no product ID.
  - The camera sits on `tegra-xusb` root port 2 (`2-2`), not behind the USB5744 hub (`2-1`), and
    that port has no software power switch. A cold boot on battery is a reboot with the camera
    attached.
- **The clock is not held across power-off.** The current boot started with the clock unset and
  restored a saved time 15 days stale. NTP corrected it 40 s in
  (`journalctl -b 0 -u systemd-timesyncd`). At every listed boot the system RTC (`rtc0`,
  `nvvrs-pseq-rtc`) sets the clock to 1970 (`journalctl -b <id> -k | grep 'setting system clock'`), so
  the RTC holds no time across power-off. Whether it lacks a backup cell is unknown. A cold boot
  before Wi-Fi is up would stamp logs and bags with the stale time.
- **`deploy/.env` still carries sim-lane values.** This gitignored file sets
  `STRAFER_SLAM_TASK_ID=enrich` (:107), `STRAFER_SLAM_SCENE_TOKEN=rigv3gate1` (:117) and the v3
  artifact path (:42). The SLAM keys are passed through at `docker-compose.yml`:105-106, so a
  real-robot bring-up with them would key RTAB-Map to a sim scene's database.

### Mass and camera mount

- **Mass.** `strafer_shared/constants.py`:41 counts `MASS_MISC` = 0.5 kg as "LiPo + wires + buck
  converter + mounting hw", which leaves out the Jetson. `MASS_TOTAL` is about 5.108 kg (:43), but
  the DR audit (:142) says about 4.5 kg. Electronics masses in the USD are deferred
  ([`DEFERRED_WORK.md`](../../DEFERRED_WORK.md):158-170).
- **The modelled camera position.** The camera is modelled at (0.20, 0.0, 0.25) m
  (`constants.py`:117-119) on a `d555_mount` joint whose parent is `base_link`, at the wheel axle
  0.048 m up (`strafer_description/urdf/strafer.urdf.xacro`:9, :234-238). That puts the lens about
  0.298 m up, level.
- **The measured camera position.** The 2026-09-26 plane fit puts the lens at 0.2781–0.2791 m,
  pitched −0.83° to −0.92° and rolled +0.18° to +0.19° (texture record:51-54). X and Y were not
  measured.
- **The rules for changing them.** A constants change goes through the DR audit's rule (:468-473),
  and constants are append-only across the lane boundary (ownership-boundaries.md:44-49). A remount
  is itself a rebolt cycle, and the DR audit's Phase 1 item 5 (:211-224) asks for the delta across
  one.

### What this unblocks

- Mount-height captures anywhere in a room, not only where the cables reach.
- [`real-d555-depth-range-survey`](../investigations/real-d555-depth-range-survey.md), which
  translates the robot through 1–12 m in a deployment room (:71-79).
- Real closed-loop driving, together with `d555-l4t-stream-integrity` (IMU, aligned depth) and a
  chassis bringup. The decode brief's hardware item is blocked because "the rig has no motor
  controller" (`d555-depth-decode-validity.md`:207-210).
- The two parked briefs that trigger on real-robot bringup, which this brief is a precondition of:
  `roboclaw-error-visibility-and-low-battery` (:5-7) and
  [`d555-usb-dropout-framerate-collapse`](../../parked/reliability/d555-usb-dropout-framerate-collapse.md)
  (:5-8).
- The DR audit's Phase 1 inputs: the chassis weight (:192-194), battery voltage over a 5-minute
  mission (:195-197), and the mount position with a rebolt delta.

### Open questions

- **The carrier.** Which carrier is it, and what DC input range does it take? The device tree and
  the flashed image disagree, and only the board's label or datasheet settles it.
- **The Wi-Fi hardware.** Does the carrier have a physical M.2 Key E slot behind the empty
  `pcie@14100000` controller? Which adapter goes in, and where is the A8000 now?
- **The pack.** Does the Jetson share one pack with the motors or get its own? What cell count,
  what regulator, and how is the Jetson's supply cut off at low voltage?
- **The wall-modem adaptor.** Is it the powerline adaptor that `enriched-lane-rig-stability`
  measured (:220-222)? The host cannot tell. It links at 1000/full to the same router the old Wi-Fi
  associated with.
- **The camera across a boot.** Is there a recovery that does not need the cable pulled, or a boot
  order that avoids the failure? It failed at 5 of 5 boots it was attached through.
- **The camera cord.** Is a powered long USB cable for the D555 wanted? It would relax where the
  Jetson must sit.

## Acceptance criteria

The boxes are in working order, so the robot is never unreachable: each link is proven before the
one it replaces comes out.

**Wi-Fi, with both cables still in**

- [ ] A Wi-Fi interface is on the Jetson through an adapter chosen against the 2026-08-02 history.
      The A8000 on `mt7921u` is not reused unless it passes the soak below. The adapter, its
      driver and the bus it sits on are recorded.
- [ ] A NetworkManager profile that follows the chosen adapter autoconnects at boot, and the stale
      `wlx…` profile is removed or re-bound. Power save is set deliberately on the profile, and the
      choice is recorded with the round-trip time it gives.
- [ ] The Jetson has a Wi-Fi address that survives DHCP: a reservation, or an mDNS name that
      resolves from the DGX to the Wi-Fi address. `avahi` no longer publishes `docker0` or `br-*`.
- [ ] SSH and VS Code reach the Jetson over Wi-Fi with the wall-modem link disconnected. The direct
      cable stays in meanwhile, as the rescue path through the DGX.
- [ ] A soak of at least 60 minutes, run twice, under sustained load at the robot's measured Wi-Fi
      need: at least the Foxglove viewer with its camera topics open, plus an SSH session. SSH stays
      responsive, and `journalctl -k` has no `mt76` timeout or chip-reset lines.

**DDS that needs neither cable**

- [ ] A robot-local Cyclone profile is the default for every documented real-robot entry point:
      bare `docker compose up`, `make up`, `make launch-autonomy`, `make viewer` and
      `make submit-deploy`. The profile runs same-host discovery on `127.0.0.1` with
      `multicast="true"`, keeps the tracked `<Internal>` tuning, and depends on neither Ethernet
      interface. Shown by the effective mount source of each service (`docker inspect`), and by
      bringing `perception`, `slam`, `navigation` and `autonomy` up with `enP7p1s0` unplugged.
- [ ] Whether that profile is tracked or host-local is decided, with the reason recorded;
      `check_env_sync.py` passes.
- [ ] The direct-link pin stays available as an explicit re-dock for sim-bridge sessions. One
      documented command selects it and names what must be recreated, and one returns the robot to
      its own profile. A re-docked sim-bridge session reproduces the link brief's wired delivery:
      the inference node's `depth rx` climbs about one per `ticks timer`.
- [ ] What running participants on the direct-link pin do when `enP7p1s0` loses carrier
      mid-session is measured once and recorded.
- [ ] Remote inspection over Wi-Fi works through one documented path: the Foxglove viewer on
      `:8765`, or `zenoh-bridge` once it is shown to discover the loopback-pinned nodes. Camera DDS
      traffic does not cross Wi-Fi.

**The executor's services over Wi-Fi**

- [ ] The real-robot `autonomy` container gets `VLM_URL` and `PLANNER_URL` for the DGX's Wi-Fi
      address through a mechanism that carries no other key, in particular neither
      `STRAFER_USE_SIM_TIME` nor `STRAFER_NAV_BACKEND`. The URLs stay out of canon.
- [ ] With both services up on the DGX and both cables out of the Jetson, the executor passes its
      startup health check, and the health round-trip time is recorded. The VLM and planner start
      and answer `/health` with the DGX's end of the cable unplugged, whatever its `.env` pin.
- [ ] `.env.example`:143-152 and `strafer_bringup/config/env_sim_in_the_loop.env`:8-12 name the
      Jetson in a form that survives DHCP, preferably its mDNS name, with its real user. (DGX-lane
      file.)

**Power, measured before the pack is chosen**

- [ ] The carrier's model and DC input range are read off the board and recorded.
- [ ] Draw is measured at the carrier's DC input on the current mains supply, with `VDD_IN` logged
      alongside. The stack is fully up: `perception` with the D555 streaming, plus `slam`,
      `navigation`, `autonomy` and `inference`, with Wi-Fi active. The record gives mean and peak
      over at least 10 minutes, and the boot inrush.
- [ ] A power mode and boot target are chosen for battery operation, each with its measured draw
      difference. Today the host runs `MAXN_SUPER` and boots to `graphical.target` with `gdm`.
- [ ] The pack decision is recorded: whether it is shared with the motors, the cell count, the
      regulator and its rating against the measured peak, and runtime at the measured mean. The 3S
      and 4S statements in `WIRING_GUIDE.md`, the parked low-battery brief and the DR audit's
      motor-strength row are made to agree with it, or the conflict is flagged in each.
- [ ] How the Jetson's supply is monitored and cut off at low voltage is decided. If
      `roboclaw-error-visibility-and-low-battery` should own it, it is handed to that brief.

**Bracket and remount**

- [ ] The Jetson is bracketed to the chassis within reach of the camera's working cord. The camera
      is re-seated as the bracket requires. The Wi-Fi adapter's placement is chosen with the RSSI
      it gives in the room.
- [ ] After the remount the camera position is re-measured and recorded, against both the model
      and the 2026-09-26 figures:
      - lens height, pitch and roll, by the 2026-09-26 plane fit (texture record:51-53);
      - X and Y, measured physically against `body_link`.

      The difference from 2026-09-26 is recorded as a before-and-after mount change for the DR
      audit's Phase 1 item 5. The constants change only through that brief's rule.
- [ ] The robot is weighed as bracketed, with its pack or packs, and recorded against `MASS_TOTAL`
      and the DR audit's 4.5 kg, for Phase 1 item 1 and `DEFERRED_WORK.md`'s electronics masses.
- [ ] Over 10 minutes of full-stack load in the bracketed position, `tegrastats` shows no thermal
      throttling: no clock cap, and the SoC below its throttle trip point. Peak temperatures are
      recorded.

**The camera's serial interface, fenced before any RoboClaw is plugged in**

- [ ] `99-strafer.rules` is installed and survives a reboot.
- [ ] With the D555 attached and no RoboClaw, nothing opens or writes to the camera's
      `/dev/ttyACM0`. Either `base` is left out of every boot and `up` path until the controllers
      are attached, or the driver's fallback stops probing `/dev/ttyACM*`. Shown on a
      camera-attached host with `fuser /dev/ttyACM0` and the node's log.
      `docker-compose.yml`:32-34 is corrected.

**Cutover and cold boot**

- [ ] An explicit boot path starts the intended service set on power-on, including what the canon
      backend needs: `inference` under `--profile policy` for a policy backend. `base` stays out
      until the controllers are attached. `deploy/.env`'s sim-lane SLAM keys are reset, and its
      model path is set deliberately.
- [ ] Three cold boots on battery with both Ethernet cables out. Each one records:
      - SSH reachable over Wi-Fi, with the time to reachable;
      - the time to NTP sync;
      - the intended containers up;
      - the D555 enumerating as `8086:0b56` at 5000 Mbit/s, with `/dev/video*` and its CDC present.

      If the camera comes up without its interfaces, as it did after the 2026-09-10 boot, one of two
      things is recorded before cutover counts as done: the cause and a recovery that does not need
      the cable pulled, or a written boot procedure that avoids the failure.
- [ ] The robot is carried through the range survey's 12 m span in a deployment room with SSH and
      the viewer staying up. The RSSI and round-trip time at the far end are recorded.
- [ ] The rescue path is written down: which cable to plug back in, and what then works. The
      wall-modem profile autoconnects with no interface binding; the direct-link profile is bound
      to `enP7p1s0`.

**Docs**

- [ ] The topology surfaces say what is true after cutover:
      - `repo-topology.md`:3-12;
      - `Readme.md`:72, :84, :107-111 and :125, including that the host is an Orin NX 16 GB on
        L4T R36.4.3;
      - `WIRING_GUIDE.md`: power (:233-239), USB and udev (:243-259), and the wire count
        (:307-316). Its serial-keyed `roboclaw_front`/`roboclaw_rear` snippet contradicts
        `99-strafer.rules`:20, and the D555's CDC takes `/dev/ttyACM0`;
      - the cheatsheet's SSH target (:105) and wired pre-flight (:116-128);
      - the link brief's transport table (:52-58);
      - `source/strafer_ros/README.md`:3, :124 and :169;
      - the `jetson-desktop` SSH target at `Readme.md`:349 and
        `docs/example_commands_cheatsheet.md`:383;
      - "Orin Nano" at `WIRING_GUIDE.md`:3 and :26;
      - the `/dev/roboclaw_front` examples at `strafer_bringup/launch/base.launch.py`:10 and
        `strafer_bringup/launch/perception.launch.py`:15, a name no rule creates.

      `docs/INTEGRATION_SIM_IN_THE_LOOP.md` is exempt: see its INTERIM banner. Its successor is the
      cheatsheet. The docs, `.env.example` and `Makefile` items are DGX-lane files.

      `enriched-lane-rig-stability.md`:226-227 is superseded there, with a pointer here.
- [ ] The reboot hazard to the camera and the stale clock at boot are recorded where an operator
      reads before powering the robot off.
- [ ] No direct-link address and no Wi-Fi network name appear in tracked text.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports. `check_env_sync.py` passes, and
      the re-docked sim-bridge session above stands for the sim-bridge lane.

## Investigation pointers

- **Host reads used above:**
  - `ip -br addr`, `ip route`, `nmcli -t -f NAME,TYPE,AUTOCONNECT con show`, `iw dev`,
    `lsusb -t`;
  - `journalctl --since 2026-08-02 --until 2026-08-05 | grep -E 'mt7921u|usb [0-9]-'` (without `-k`),
    `journalctl -b 0 -k | grep 'usb 2-2'`, `journalctl --list-boots`,
    `journalctl -b 0 -u systemd-timesyncd`;
  - `getent ahostsv4 strafer-nx.local`, `nvpmodel -q`, `tegrastats`, `/etc/nv_tegra_release`;
  - the INA3221 hwmon node above;
  - `docker inspect -f '{{ index .Config.Labels "com.docker.compose.project.config_files" }}' <c>`
    for which compose files built a container.
- **Possible cable rescue path.** L4T's USB device-mode bridge `l4tbr0` exists, with `usb0` and
  `usb1` down. Whether the carrier's device-mode port gives a cable rescue path was not checked.
- **Free USB ports.**
  - The USB5744's four downstream ports and the Renesas controller's four ports are free.
  - Of the `tegra-xusb` root ports, only HS port 3 (`usb2-2`, USB 2.0 only) is enabled in the
    device tree (`/proc/device-tree/bus@0/padctl@3520000/ports/*/status`).
  - The device-mode pad `usb2-0` is the camera port's USB 2.0 companion.
  - `connect_type` reads `unknown` on every port, so the physical layout cannot be read from sysfs.
- **Compose and DDS.** `source/strafer_ros/deploy/docker-compose.yml` (header :1-38, `base`
  :62-79, `zenoh-bridge` :171-190); `deploy/docker-compose.override.yml` and
  `~/.config/cyclonedds/direct-link.xml` (host-local); `deploy/tests/check_env_sync.py`; the
  `Makefile`'s `FULL_COMPOSE` (:156) and `udev` (:238-242).
- **Driver.** `strafer_driver/strafer_driver/roboclaw_interface.py` (`probe_address`,
  `detect_roboclaws`), `roboclaw_node.py`:106-165, `strafer_driver/launch/driver.launch.py`, and
  `source/strafer_ros/99-strafer.rules` (the RoboClaw rule at :20; the D555 IIO and hidraw rules at
  :26 and :29).
- **The link brief's §"Wired path — measured 2026-08-17"** (:87-99) gives the pin's shape on both
  hosts and the wired figures the re-dock has to reproduce.
- **The hardware lanes `nx-docker-bringup` left unticked** (:97-101): RoboClaw passthrough, D555
  passthrough and the IMU modules. Its deferred choice between USB and Ethernet for the camera is
  at :106.

## Out of scope

- **Chassis bringup:** connecting the RoboClaws, validating front/rear auto-detect, and driving. No
  brief is filed for it; its lanes are the unticked hardware boxes of
  [`nx-docker-bringup`](../../completed/nx-docker-bringup.md):97-101. This brief only fences the
  camera's serial interface ahead of it.
- **`enriched-lane-rig-stability`'s other modes.** Mode 2 (Nav2 lifecycle recovery after a
  partition) and its pre-arm health gate stay there. For its mode 3 (the `mt7921u` wedge), this
  brief's adapter choice and soak are the Jetson-side fix; that brief's mode-3 box points here.
- **Runtime camera-dropout detection and any automatic USB or UVC reset.** These stay with
  `d555-usb-dropout-framerate-collapse`. This brief covers only enumeration across a cold boot, and
  a boot procedure that avoids the failure is enough to close it.
- **The D555 stream defects:** the IMU, aligned depth, stamps and duplicate frames, owned by
  `d555-l4t-stream-integrity`. Its host change must survive the reboots this brief adds (:89).
- **Carrying the sim-bridge lane over Wi-Fi**, or tuning the DGX's Wi-Fi. Both are owned by
  `sim-bridge-link-transport-capacity`.
- **Implementing battery monitoring and a low-battery mode.** That belongs to the parked
  `roboclaw-error-visibility-and-low-battery`. This brief only decides the pack and how it is
  watched.
- **The DR audit's other Phase 1 measurements, all of its analysis, and any change to
  `sim_real_cfg.py` or the constants.** This brief hands over one as-bracketed weight and one
  post-remount mount position.
- **Remote DDS over Wi-Fi.** A remote `ros2` CLI is not supported. DDS stays on loopback, and
  inspection goes through the one documented path in the acceptance criteria.
- **Peer-side (DGX) work** is named in each box that needs it: the `Makefile` default,
  `.env.example`, the DGX `.env` pin and the `docs/` surfaces. The Jetson agent hands these off
  through the operator.
- **Electronics masses in the USD** (`DEFERRED_WORK.md`:158-170).
- **Moving the D555 to Ethernet, or a powered long USB cable.** Either one is recorded here as an
  option only.

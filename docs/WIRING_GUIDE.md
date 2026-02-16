# Strafer Robot: Wiring Guide

Complete wiring reference for connecting 4x GoBilda 5203 motors to 2x RoboClaw ST 2x45A motor controllers and the Jetson Orin Nano. This configuration exactly matches the USD simulation in `source/strafer_lab/`.

**Board variant**: RoboClaw **ST** (Screw Terminal) -- all connections are screw terminals, no pin headers. Reference: [RoboClaw ST 2x45A Datasheet (PDF)](https://downloads.basicmicro.com/docs/roboclaw_datasheet_ST_2x45A.pdf), page 6.

---

## Physical Layout

```
              FRONT OF ROBOT
    ┌─────────────────────────────┐
    │                             │
    │  FL (wheel_1)     FR (wheel_2)
    │  ●                 ●        │
    │  │                 │        │
    │  │  ┌───────────┐  │        │
    │  └──│ RoboClaw  │──┘        │
    │     │    #1      │          │
    │     │  (0x80)   │          │
    │     └─────┬─────┘          │
    │           │ USB            │
    │     ┌─────┴─────┐          │
    │     │  Jetson   │          │
    │     │ Orin Nano │          │
    │     └─────┬─────┘          │
    │           │ USB            │
    │     ┌─────┴─────┐          │
    │     │ RoboClaw  │          │
    │     │    #2      │          │
    │     │  (0x81)   │          │
    │  ┌──│           │──┐        │
    │  │  └───────────┘  │        │
    │  ●                 ●        │
    │  RL (wheel_3)     RR (wheel_4)
    │                             │
    └─────────────────────────────┘
              REAR OF ROBOT
```

### Simulation-to-Physical Mapping

From [strafer_env_cfg.py:183](source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py#L183) and [observations.py:42](source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py#L42):

| Position | Joint Name | Kinematic Index | RoboClaw | Channel | Axis Sign |
|----------|-----------|----------------|----------|---------|-----------|
| Front-Left | `wheel_1_drive` | 0 | #1 (0x80) | M1 / Enc 1 | -1.0 |
| Front-Right | `wheel_2_drive` | 1 | #1 (0x80) | M2 / Enc 2 | +1.0 |
| Rear-Left | `wheel_3_drive` | 2 | #2 (0x81) | M1 / Enc 1 | -1.0 |
| Rear-Right | `wheel_4_drive` | 3 | #2 (0x81) | M2 / Enc 2 | +1.0 |

---

## RoboClaw ST 2x45A Terminal Layout

The ST version has **three terminal areas**, a **Micro-USB port**, and a **status LED**:

1. **Power/Motor terminals** (large phillips-head screw terminals) -- one end of the board
2. **Encoder terminals** (small screw terminals) -- side of the board
3. **Control terminal block** (small screw terminals) -- opposite end from power/motor

```
                      RoboClaw ST 2x45A (top view)

         POWER/MOTOR END                        CONTROL END
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  POWER/MOTOR TERMINALS        CONTROL TERMINAL BLOCK     │
    │  (large phillips screws)      (small screw terminals)    │
    │  ┌───┬───┬──┬──┬───┬───┐     ┌──┬───┬───┬──┬──┬──┬──┬──┬──┬──┐
    │  │M2A│M2B│ -│ +│M1B│M1A│     │LB│GND│GND│5+│5+│S1│S2│S3│S4│S5│
    │  └───┴───┴──┴──┴───┴───┘     └──┴───┴───┴──┴──┴──┴──┴──┴──┴──┘
    │                                                          │
    │          [USB Micro-B]                                   │
    │                                                          │
    │  ENCODER TERMINALS (small screw terminals, along side)   │
    │  ┌──┬──┬──┬──┐                                           │
    │  │1B│1A│2B│2A│                                           │
    │  └──┴──┴──┴──┘                                           │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
```

### Power/Motor Terminals (large phillips-head screws, per datasheet p4)

| Terminal | Label | Function |
|----------|-------|----------|
| 1 | **M2A** | Motor 2 lead A |
| 2 | **M2B** | Motor 2 lead B |
| 3 | **-** | Battery negative (GND) |
| 4 | **+** | Battery positive (6-34V) |
| 5 | **M1B** | Motor 1 lead B |
| 6 | **M1A** | Motor 1 lead A |

### Encoder Terminals (small screw terminals, side of board)

| Terminal | Label | Function |
|----------|-------|----------|
| 1 | **1B** | Encoder 1, Channel B |
| 2 | **1A** | Encoder 1, Channel A |
| 3 | **2B** | Encoder 2, Channel B |
| 4 | **2A** | Encoder 2, Channel A |

### Control Terminal Block (small screw terminals, opposite end from power)

| Terminal | Label | Function |
|----------|-------|----------|
| 1 | **LB** | Logic battery input (optional, for separate logic power) |
| 2 | **GND** | Ground |
| 3 | **GND** | Ground |
| 4 | **5+** | 5V output from BEC (3A max) |
| 5 | **5+** | 5V output from BEC (3A max) |
| 6 | **S1** | Signal input 1 |
| 7 | **S2** | Signal input 2 |
| 8 | **S3** | Signal input 3 |
| 9 | **S4** | Signal input 4 |
| 10 | **S5** | Signal input 5 |

**Key points for our use:**
- Motor leads connect to the **power/motor terminals** (large phillips screws)
- Battery connects to the **power/motor terminals** (- and + between the motor pairs)
- Encoder **signals** (A/B) connect to the **encoder terminals** (small side screws: 1A, 1B, 2A, 2B)
- Encoder **power** (5V and GND) connects to the **control terminal block** (5+ and GND)
- M1 motor always pairs with encoder 1 (1A, 1B). M2 motor always pairs with encoder 2 (2A, 2B).

---

## 1. Motor Power Wires

Each GoBilda 5203 motor has 2 power leads with 3.5mm bullet connectors (18 AWG, 470mm length). They are not color-coded for polarity -- direction depends on which way you connect them and is corrected in software via `wheel_axis_signs`.

### RoboClaw #1 (address 0x80) -- Front Axle

| Power/Motor Terminal | Motor | Simulation Joint |
|---|---|---|
| **M1A** | FL motor wire 1 | `wheel_1_drive` |
| **M1B** | FL motor wire 2 | |
| **M2A** | FR motor wire 1 | `wheel_2_drive` |
| **M2B** | FR motor wire 2 | |

### RoboClaw #2 (address 0x81) -- Rear Axle

| Power/Motor Terminal | Motor | Simulation Joint |
|---|---|---|
| **M1A** | RL motor wire 1 | `wheel_3_drive` |
| **M1B** | RL motor wire 2 | |
| **M2A** | RR motor wire 1 | `wheel_4_drive` |
| **M2B** | RR motor wire 2 | |

Strip the bullet connectors, insert bare wire into the large phillips-head screw terminals, and tighten.

---

## 2. Encoder Wires

Each GoBilda 5203 has a hall-effect quadrature encoder (537.7 PPR at output shaft) with a 4-pin JST XH connector. You will need a [GoBilda Encoder Breakout Cable](https://www.gobilda.com/encoder-breakout-cable-4-pos-jst-xh-mh-fc-to-4-x-1-pos-tjc8-mh-fc-300mm-length/) per motor to split the JST XH into individual wires for the RoboClaw screw terminals.

### GoBilda JST XH Encoder Pinout

| JST XH Pin | Function |
|---|---|
| Pin 1 | GND |
| Pin 2 | Encoder Channel B |
| Pin 3 | Encoder Channel A |
| Pin 4 | VCC (3.3-5V) |

### Where Each Encoder Wire Goes on the RoboClaw ST

Each encoder has 4 wires. Two (signal) go to the **encoder terminals** (side), two (power) go to the **control terminal block**:

```
GoBilda Motor Encoder              RoboClaw ST 2x45A
(JST XH breakout)                  ─────────────────
                                   ENCODER TERMINALS (side of board):
  Channel A (pin 3)  ──────────>   1A  (encoder 1) or 2A (encoder 2)
  Channel B (pin 2)  ──────────>   1B  (encoder 1) or 2B (encoder 2)

                                   CONTROL TERMINAL BLOCK:
  VCC (pin 4)        ──────────>   5+  (5V BEC output)
  GND (pin 1)        ──────────>   GND
```

### Wiring: RoboClaw #1 (0x80) -- Front Encoders

| Motor | Encoder Wire | RoboClaw Terminal | Terminal Area |
|---|---|---|---|
| **FL (wheel_1)** | Channel A (pin 3) | **1A** | Encoder (side) |
| | Channel B (pin 2) | **1B** | Encoder (side) |
| | VCC (pin 4) | **5+** | Control block |
| | GND (pin 1) | **GND** | Control block |
| **FR (wheel_2)** | Channel A (pin 3) | **2A** | Encoder (side) |
| | Channel B (pin 2) | **2B** | Encoder (side) |
| | VCC (pin 4) | **5+** | Control block |
| | GND (pin 1) | **GND** | Control block |

### Wiring: RoboClaw #2 (0x81) -- Rear Encoders

| Motor | Encoder Wire | RoboClaw Terminal | Terminal Area |
|---|---|---|---|
| **RL (wheel_3)** | Channel A (pin 3) | **1A** | Encoder (side) |
| | Channel B (pin 2) | **1B** | Encoder (side) |
| | VCC (pin 4) | **5+** | Control block |
| | GND (pin 1) | **GND** | Control block |
| **RR (wheel_4)** | Channel A (pin 3) | **2A** | Encoder (side) |
| | Channel B (pin 2) | **2B** | Encoder (side) |
| | VCC (pin 4) | **5+** | Control block |
| | GND (pin 1) | **GND** | Control block |

**Rules:**
- M1 motor always pairs with encoder 1 (1A, 1B). M2 motor always pairs with encoder 2 (2A, 2B).
- The control terminal block has two 5+ and two GND screw terminals -- enough for two encoders per RoboClaw.
- The 5+ terminals output 5V from the RoboClaw's built-in BEC (3A max). The GoBilda encoders accept 3.3-5V, so this is correct.

---

## 3. Battery Power

Battery connects to the **power/motor terminals** (large phillips-head screws). The `-` and `+` terminals sit between the M1 and M2 motor pairs.

```
12V LiPo Battery (3S or 4S, 5000+ mAh recommended)
    │
    ├── (+) ──> RoboClaw #1 (+) power/motor terminal
    ├── (-) ──> RoboClaw #1 (-) power/motor terminal
    │
    ├── (+) ──> RoboClaw #2 (+) power/motor terminal
    └── (-) ──> RoboClaw #2 (-) power/motor terminal
```

Use a wiring harness or terminal block to split battery power to both RoboClaws. Keep leads as short as possible to reduce inductance.

### Jetson Power

The Jetson Orin Nano carrier board requires 7-20V DC input (barrel jack). Options:

- **4S LiPo (14.8V nominal)**: Can power the Jetson directly via barrel jack, and the RoboClaws via the same battery.
- **3S LiPo (11.1V nominal)**: Within range but at the low end. A 12V-to-19V boost converter is safer for the Jetson.
- **Separate supply**: Use a dedicated 19V adapter for the Jetson during bench testing.

---

## 4. USB Connections (RoboClaws + RealSense -> Jetson)

| Jetson USB Port | Cable | Device |
|---|---|---|
| USB 3.1 Type-A #1 | USB-A to Micro-B | RoboClaw #1 (front, 0x80) |
| USB 3.1 Type-A #2 | USB-A to Micro-B | RoboClaw #2 (rear, 0x81) |
| USB 3.1 Type-C (or remaining Type-A) | USB-C cable | RealSense D555 |

On Linux, RoboClaws appear as `/dev/ttyACM0` and `/dev/ttyACM1`. Order is not guaranteed across reboots -- use udev rules for stable naming:

```bash
# /etc/udev/rules.d/99-strafer.rules
SUBSYSTEM=="tty", ATTRS{idVendor}=="03eb", ATTRS{serial}=="<SERIAL_1>", SYMLINK+="roboclaw_front"
SUBSYSTEM=="tty", ATTRS{idVendor}=="03eb", ATTRS{serial}=="<SERIAL_2>", SYMLINK+="roboclaw_rear"
```

To find the serial numbers, run `udevadm info -a /dev/ttyACM0 | grep serial` with each RoboClaw connected individually.

---

## 5. RoboClaw Address Configuration

Both RoboClaws ship at address 0x80 (128). One must be changed to 0x81 (129) using [Motion Studio](https://resources.basicmicro.com/general-settings-in-motion-studio/) (Windows software from BasicMicro).

**Do this before connecting both to the Jetson.**

### For RoboClaw #2 (rear):

1. Connect **only** RoboClaw #2 to your Windows PC via USB
2. Open Motion Studio, connect to the device
3. Go to **General Settings** -> **Serial** pane
4. Set address to **129 (0x81)**
5. Set mode to **Packet Serial**
6. Set baud rate to **115200**
7. Write settings to device

### For RoboClaw #1 (front):

1. Verify address is **128 (0x80)** (factory default)
2. Set mode to **Packet Serial**
3. Set baud rate to **115200**
4. Write settings to device

---

## 6. Direction Verification

After all wiring is complete, verify that motor directions match the simulation. The `wheel_axis_signs = [-1, 1, -1, 1]` from [strafer_env_cfg.py](source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py) means:

- A **positive velocity command** through `strafer_shared.mecanum_kinematics` should drive FR and RR wheels **forward**
- A **positive velocity command** should drive FL and RL wheels in the **opposite shaft direction** (the sign inversion handles this)

### Calibration procedure:

1. Send a low positive velocity (e.g., 500 ticks/sec) to RoboClaw #1 M1 (FL motor)
2. Observe which direction the wheel spins
3. If the robot would move **backward** with this command after sign correction is applied, **swap the M1A/M1B wires** on the screw terminal
4. Repeat for each motor
5. For each encoder: if counts go **negative** when the motor spins in its "forward" direction, **swap Channel A and Channel B** on that ENC header

This is a one-time calibration done during Phase 1 with a standalone test script on the Jetson.

---

## 7. Complete Wire Count

| Component | Wires | Type |
|---|---|---|
| 4x Motor power (2 wires each) | 8 | 18 AWG, bullet connectors -> screw terminals |
| 4x Encoder (4 wires each) | 16 | JST XH breakout -> screw terminals |
| Battery -> 2x RoboClaw | 4 | Heavy gauge (12-14 AWG) |
| 2x USB (RoboClaw -> Jetson) | 2 cables | USB-A to Micro-B |
| 1x USB (RealSense -> Jetson) | 1 cable | USB-C |
| **Total** | **28 wires + 3 USB cables** | |

---

## References

- [RoboClaw User Manual (PDF)](https://downloads.basicmicro.com/docs/roboclaw_user_manual.pdf)
- [RoboClaw Quick Start Guide](https://resources.basicmicro.com/dual-channel-roboclaw-quick-start-guide/)
- [RoboClaw Basics - Encoder Wiring](https://resources.basicmicro.com/roboclaw-basics/)
- [GoBilda 5203 Motor (19.2:1)](https://www.gobilda.com/5203-series-yellow-jacket-planetary-gear-motor-19-2-1-ratio-24mm-length-8mm-rex-shaft-312-rpm-3-3-5v-encoder/)
- [GoBilda Encoder Breakout Cable](https://www.gobilda.com/encoder-breakout-cable-4-pos-jst-xh-mh-fc-to-4-x-1-pos-tjc8-mh-fc-300mm-length/)
- [Motion Studio - Address Configuration](https://resources.basicmicro.com/general-settings-in-motion-studio/)

"""Minimal vendored RoboClaw packet serial interface.

Implements the subset of the RoboClaw packet serial protocol needed by the
strafer_driver node.  Uses only pyserial -- no third-party RoboClaw libraries.

Protocol reference:
    https://downloads.basicmicro.com/docs/roboclaw_user_manual.pdf

Packet format (send):  [address, command, ...data, crc16_hi, crc16_lo]
Packet format (recv):  [...data, crc16_hi, crc16_lo]  (no address/command echo)
"""

from __future__ import annotations

import glob
import struct
import threading
import time
from typing import Optional

import serial


# ---------------------------------------------------------------------------
# CRC-16 (same algorithm as BasicMicro's reference code)
# ---------------------------------------------------------------------------

def _crc16(data: bytes) -> int:
    """Compute CRC-16 for RoboClaw packet serial protocol."""
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


# ---------------------------------------------------------------------------
# Command IDs (from RoboClaw User Manual)
# ---------------------------------------------------------------------------

CMD_M1_FORWARD = 0
CMD_M1_BACKWARD = 1
CMD_M2_FORWARD = 4
CMD_M2_BACKWARD = 5

CMD_READ_ENC_M1 = 16
CMD_READ_ENC_M2 = 17
CMD_READ_SPEED_M1 = 18
CMD_READ_SPEED_M2 = 19

CMD_DRIVE_M1_SPEED = 35
CMD_DRIVE_M2_SPEED = 36
CMD_DRIVE_M1M2_SPEED = 37

CMD_READ_MAIN_BATTERY = 24

CMD_SET_M1_VEL_PID = 28
CMD_SET_M2_VEL_PID = 29
CMD_READ_M1_VEL_PID = 55
CMD_READ_M2_VEL_PID = 56

CMD_RESET_ENCODERS = 20

CMD_READ_TEMP = 82


class RoboClawError(Exception):
    """Base exception for RoboClaw communication errors."""


class RoboClawChecksumError(RoboClawError):
    """CRC mismatch on received data."""


class RoboClawTimeoutError(RoboClawError):
    """No response within the read timeout."""


class RoboClawInterface:
    """Low-level serial interface to a single RoboClaw controller.

    Thread-safe: all serial I/O is guarded by a lock. However, the intended
    usage pattern is a single 50 Hz timer callback doing all reads/writes
    sequentially.

    Args:
        port: Serial port path (e.g. ``/dev/ttyACM0``).
        address: RoboClaw packet serial address (0x80 or 0x81).
        baud_rate: Serial baud rate.
        read_timeout: Per-command read timeout in seconds.
        max_retries: Number of retries on checksum/timeout failure.
    """

    def __init__(
        self,
        port: str,
        address: int,
        baud_rate: int = 115200,
        read_timeout: float = 0.1,
        max_retries: int = 1,
    ) -> None:
        self.port = port
        self.address = address
        self.baud_rate = baud_rate
        self.read_timeout = read_timeout
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._serial: Optional[serial.Serial] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the serial port."""
        with self._lock:
            if self._serial is not None and self._serial.is_open:
                return
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.read_timeout,
                interCharTimeout=0.01,
            )

    def close(self) -> None:
        """Close the serial port."""
        with self._lock:
            if self._serial is not None and self._serial.is_open:
                self._serial.close()
            self._serial = None

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._serial is not None and self._serial.is_open

    def reconnect(self) -> bool:
        """Close and reopen the serial port. Returns True on success."""
        self.close()
        try:
            self.open()
            return True
        except serial.SerialException:
            return False

    # ------------------------------------------------------------------
    # Low-level protocol
    # ------------------------------------------------------------------

    def _send_command(self, command: int, data: bytes = b"") -> None:
        """Send a command packet (address + command + data + CRC16)."""
        packet = bytes([self.address, command]) + data
        crc = _crc16(packet)
        packet += struct.pack(">H", crc)
        self._serial.write(packet)

    def _read_bytes(self, count: int) -> bytes:
        """Read exactly *count* bytes, raising on timeout."""
        data = self._serial.read(count)
        if len(data) != count:
            raise RoboClawTimeoutError(
                f"Expected {count} bytes, got {len(data)} "
                f"(port={self.port}, addr=0x{self.address:02x})"
            )
        return data

    def _send_and_recv(self, command: int, data: bytes, recv_bytes: int) -> bytes:
        """Send command, read response, verify CRC. Returns payload (no CRC)."""
        header = bytes([self.address, command])
        packet = header + data
        crc = _crc16(packet)
        packet += struct.pack(">H", crc)

        self._serial.reset_input_buffer()
        self._serial.write(packet)

        # Response: recv_bytes of payload + 2 bytes CRC
        response = self._read_bytes(recv_bytes + 2)
        payload = response[:recv_bytes]
        recv_crc = struct.unpack(">H", response[recv_bytes:])[0]

        # CRC is computed over address + command + payload
        expected_crc = _crc16(header + payload)
        if recv_crc != expected_crc:
            raise RoboClawChecksumError(
                f"CRC mismatch: got 0x{recv_crc:04x}, expected 0x{expected_crc:04x}"
            )
        return payload

    def _send_with_ack(self, command: int, data: bytes = b"") -> bool:
        """Send a write command and read the 0xFF ACK byte."""
        packet = bytes([self.address, command]) + data
        crc = _crc16(packet)
        packet += struct.pack(">H", crc)

        self._serial.reset_input_buffer()
        self._serial.write(packet)

        ack = self._serial.read(1)
        return ack == b"\xff"

    def _retry(self, func, *args):
        """Execute *func* with retry logic."""
        last_exc = None
        for attempt in range(1 + self.max_retries):
            try:
                with self._lock:
                    return func(*args)
            except (RoboClawError, serial.SerialException) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(0.005)  # brief pause before retry
        raise last_exc

    # ------------------------------------------------------------------
    # Motor commands
    # ------------------------------------------------------------------

    def drive_m1_speed(self, speed: int) -> bool:
        """Drive motor 1 at *speed* ticks/sec (signed int32).

        Uses command 35 (DRIVE_M1_SPEED): signed 32-bit velocity PID target.
        """
        data = struct.pack(">i", speed)
        return self._retry(self._send_with_ack, CMD_DRIVE_M1_SPEED, data)

    def drive_m2_speed(self, speed: int) -> bool:
        """Drive motor 2 at *speed* ticks/sec (signed int32)."""
        data = struct.pack(">i", speed)
        return self._retry(self._send_with_ack, CMD_DRIVE_M2_SPEED, data)

    def drive_m1m2_speed(self, speed_m1: int, speed_m2: int) -> bool:
        """Drive both motors at the specified tick/sec speeds."""
        data = struct.pack(">ii", speed_m1, speed_m2)
        return self._retry(self._send_with_ack, CMD_DRIVE_M1M2_SPEED, data)

    def stop_motors(self) -> bool:
        """Stop both motors by sending zero velocity."""
        return self.drive_m1m2_speed(0, 0)

    def forward_m1(self, duty: int) -> bool:
        """Forward motor 1 at duty (0-127). 0 = stop."""
        data = bytes([duty])
        return self._retry(self._send_with_ack, CMD_M1_FORWARD, data)

    def forward_m2(self, duty: int) -> bool:
        """Forward motor 2 at duty (0-127). 0 = stop."""
        data = bytes([duty])
        return self._retry(self._send_with_ack, CMD_M2_FORWARD, data)

    # ------------------------------------------------------------------
    # Encoder reads
    # ------------------------------------------------------------------

    def read_speed_m1(self) -> tuple[int, int]:
        """Read motor 1 encoder speed.

        Returns:
            (speed_ticks_per_sec, status) -- speed is signed int32.
            status: 0=forward, 1=backward.
        """
        payload = self._retry(self._send_and_recv, CMD_READ_SPEED_M1, b"", 5)
        speed = struct.unpack(">i", payload[:4])[0]
        status = payload[4]
        return speed, status

    def read_speed_m2(self) -> tuple[int, int]:
        """Read motor 2 encoder speed."""
        payload = self._retry(self._send_and_recv, CMD_READ_SPEED_M2, b"", 5)
        speed = struct.unpack(">i", payload[:4])[0]
        status = payload[4]
        return speed, status

    def read_encoder_m1(self) -> tuple[int, int]:
        """Read motor 1 encoder count.

        Returns:
            (count, status) -- count is unsigned int32, status is byte.
        """
        payload = self._retry(self._send_and_recv, CMD_READ_ENC_M1, b"", 5)
        count = struct.unpack(">I", payload[:4])[0]
        status = payload[4]
        return count, status

    def read_encoder_m2(self) -> tuple[int, int]:
        """Read motor 2 encoder count."""
        payload = self._retry(self._send_and_recv, CMD_READ_ENC_M2, b"", 5)
        count = struct.unpack(">I", payload[:4])[0]
        status = payload[4]
        return count, status

    # ------------------------------------------------------------------
    # PID configuration
    # ------------------------------------------------------------------

    def set_velocity_pid_m1(self, p: int, i: int, d: int, qpps: int) -> bool:
        """Set velocity PID constants for motor 1.

        Args:
            p: Proportional gain (RoboClaw uses fixed-point, typically 0-65536).
            i: Integral gain.
            d: Derivative gain.
            qpps: Quadrature Pulses Per Second at max speed.
        """
        data = struct.pack(">IIII", d, p, i, qpps)
        return self._retry(self._send_with_ack, CMD_SET_M1_VEL_PID, data)

    def set_velocity_pid_m2(self, p: int, i: int, d: int, qpps: int) -> bool:
        """Set velocity PID constants for motor 2."""
        data = struct.pack(">IIII", d, p, i, qpps)
        return self._retry(self._send_with_ack, CMD_SET_M2_VEL_PID, data)

    def read_velocity_pid_m1(self) -> dict:
        """Read velocity PID constants for motor 1.

        Returns:
            Dict with keys 'P', 'I', 'D', 'QPPS'.
        """
        payload = self._retry(self._send_and_recv, CMD_READ_M1_VEL_PID, b"", 16)
        p = struct.unpack(">I", payload[0:4])[0]
        i = struct.unpack(">I", payload[4:8])[0]
        d = struct.unpack(">I", payload[8:12])[0]
        qpps = struct.unpack(">I", payload[12:16])[0]
        return {"P": p, "I": i, "D": d, "QPPS": qpps}

    def read_velocity_pid_m2(self) -> dict:
        """Read velocity PID constants for motor 2."""
        payload = self._retry(self._send_and_recv, CMD_READ_M2_VEL_PID, b"", 16)
        p = struct.unpack(">I", payload[0:4])[0]
        i = struct.unpack(">I", payload[4:8])[0]
        d = struct.unpack(">I", payload[8:12])[0]
        qpps = struct.unpack(">I", payload[12:16])[0]
        return {"P": p, "I": i, "D": d, "QPPS": qpps}

    def reset_encoders(self) -> bool:
        """Reset both encoder counters to zero."""
        return self._retry(self._send_with_ack, CMD_RESET_ENCODERS)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def read_main_battery(self) -> float:
        """Read main battery voltage in volts."""
        payload = self._retry(self._send_and_recv, CMD_READ_MAIN_BATTERY, b"", 2)
        raw = struct.unpack(">H", payload)[0]
        return raw / 10.0

    def read_temperature(self) -> float:
        """Read board temperature in degrees Celsius."""
        payload = self._retry(self._send_and_recv, CMD_READ_TEMP, b"", 2)
        raw = struct.unpack(">H", payload)[0]
        return raw / 10.0


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def probe_address(port: str, address: int, baud_rate: int = 115200) -> bool:
    """Try to communicate with a RoboClaw at *address* on *port*.

    Sends a ``read_main_battery`` command and checks for a valid CRC
    response.  Returns True if the controller responds, False otherwise.
    Does not raise on failure.
    """
    try:
        rc = RoboClawInterface(port, address, baud_rate=baud_rate,
                               read_timeout=0.15, max_retries=0)
        rc.open()
        try:
            rc.read_main_battery()
            return True
        except (RoboClawError, serial.SerialException):
            return False
        finally:
            rc.close()
    except (serial.SerialException, OSError):
        return False


def detect_roboclaws(
    front_address: int,
    rear_address: int,
    baud_rate: int = 115200,
    candidate_ports: list[str] | None = None,
) -> dict[int, str]:
    """Scan serial ports and identify which RoboClaw is on which port.

    Probes each candidate port for both ``front_address`` and
    ``rear_address``.  Returns a dict mapping address -> port for every
    address that was found.

    Args:
        front_address: Packet serial address of the front controller.
        rear_address: Packet serial address of the rear controller.
        baud_rate: Serial baud rate.
        candidate_ports: Explicit list of ports to scan.  If ``None``,
            auto-discovers by looking for ``/dev/roboclaw_*`` first, then
            ``/dev/ttyACM*``.

    Returns:
        ``{address: port}`` for each controller found.  May be empty, or
        contain only one entry if only one controller is reachable.
    """
    if candidate_ports is None:
        candidate_ports = sorted(glob.glob("/dev/roboclaw*"))
        if not candidate_ports:
            candidate_ports = sorted(glob.glob("/dev/ttyACM*"))

    addresses = [front_address, rear_address]
    result: dict[int, str] = {}

    for port in candidate_ports:
        for addr in addresses:
            if addr in result:
                continue  # already found this one
            if probe_address(port, addr, baud_rate):
                result[addr] = port
        if len(result) == len(addresses):
            break  # both found

    return result

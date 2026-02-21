"""Tests for strafer_driver.roboclaw_interface.

Covers CRC16, packet assembly/parsing, connection management, retry logic.
"""

import struct
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def crc16():
    from strafer_driver.roboclaw_interface import _crc16
    return _crc16


@pytest.fixture
def make_interface():
    """Factory that creates a RoboClawInterface with a mock serial port."""
    from strafer_driver.roboclaw_interface import RoboClawInterface

    def _make(address=0x80):
        rc = RoboClawInterface("/dev/fake", address, baud_rate=115200)
        mock_serial = MagicMock()
        mock_serial.is_open = True
        rc._serial = mock_serial
        return rc, mock_serial

    return _make


class TestCRC16:
    def test_known_value(self, crc16):
        data = bytes([0x80, 0x12])
        crc = crc16(data)
        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF

    def test_empty_data(self, crc16):
        assert crc16(b"") == 0

    def test_single_byte(self, crc16):
        crc = crc16(b"\x00")
        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF

    def test_deterministic(self, crc16):
        data = bytes([0x80, 0x25, 0x00, 0x01])
        assert crc16(data) == crc16(data)

    def test_different_data_different_crc(self, crc16):
        assert crc16(b"\x80\x12") != crc16(b"\x80\x13")

    def test_crc_changes_with_all_bytes(self, crc16):
        """Verify CRC is sensitive to every byte position."""
        base = bytes([0x80, 0x25, 0x00, 0x00, 0x01, 0x00])
        crcs = set()
        for i in range(len(base)):
            modified = bytearray(base)
            modified[i] ^= 0x01
            crcs.add(crc16(bytes(modified)))
        assert len(crcs) == len(base)


class TestPacketProtocol:
    def test_send_with_ack_packet_format(self, make_interface, crc16):
        """Verify _send_with_ack builds correct packet: [addr, cmd, data, crc_hi, crc_lo]."""
        rc, mock_serial = make_interface()
        mock_serial.read.return_value = b"\xff"

        rc.drive_m1_speed(1000)

        written = mock_serial.write.call_args[0][0]
        assert len(written) == 8
        assert written[0] == 0x80
        assert written[1] == 35  # CMD_DRIVE_M1_SPEED
        payload = written[:-2]
        expected_crc = crc16(payload)
        actual_crc = struct.unpack(">H", written[-2:])[0]
        assert actual_crc == expected_crc

    def test_send_with_ack_returns_true_on_ack(self, make_interface):
        rc, mock_serial = make_interface()
        mock_serial.read.return_value = b"\xff"
        assert rc.forward_m1(64) is True

    def test_send_with_ack_returns_false_on_nack(self, make_interface):
        rc, mock_serial = make_interface()
        mock_serial.read.return_value = b"\x00"
        assert rc.forward_m1(64) is False

    def test_send_and_recv_parses_speed(self, make_interface, crc16):
        """Verify read_speed_m1 correctly parses a mock response."""
        rc, mock_serial = make_interface()

        speed_bytes = struct.pack(">i", 1500)
        status_byte = b"\x00"
        payload = speed_bytes + status_byte
        header = bytes([0x80, 18])
        resp_crc = crc16(header + payload)
        response = payload + struct.pack(">H", resp_crc)
        mock_serial.read.return_value = response

        speed, status = rc.read_speed_m1()
        assert speed == 1500
        assert status == 0

    def test_send_and_recv_negative_speed(self, make_interface, crc16):
        """Verify negative speed is correctly parsed."""
        rc, mock_serial = make_interface()

        speed_bytes = struct.pack(">i", -800)
        status_byte = b"\x01"
        payload = speed_bytes + status_byte
        header = bytes([0x80, 18])
        resp_crc = crc16(header + payload)
        response = payload + struct.pack(">H", resp_crc)
        mock_serial.read.return_value = response

        speed, status = rc.read_speed_m1()
        assert speed == -800
        assert status == 1

    def test_checksum_error_raised(self, make_interface):
        """Bad CRC in response should raise RoboClawChecksumError."""
        from strafer_driver.roboclaw_interface import RoboClawChecksumError
        rc, mock_serial = make_interface()
        rc.max_retries = 0

        payload = struct.pack(">i", 1000) + b"\x00"
        bad_crc = struct.pack(">H", 0xDEAD)
        mock_serial.read.return_value = payload + bad_crc

        with pytest.raises(RoboClawChecksumError):
            rc.read_speed_m1()

    def test_timeout_error_raised(self, make_interface):
        """Short response should raise RoboClawTimeoutError."""
        from strafer_driver.roboclaw_interface import RoboClawTimeoutError
        rc, mock_serial = make_interface()
        rc.max_retries = 0

        mock_serial.read.return_value = b"\x00"

        with pytest.raises(RoboClawTimeoutError):
            rc.read_speed_m1()

    def test_drive_m1m2_speed_packet(self, make_interface, crc16):
        """Verify drive_m1m2_speed sends correct dual-speed packet."""
        rc, mock_serial = make_interface()
        mock_serial.read.return_value = b"\xff"

        rc.drive_m1m2_speed(500, -300)

        written = mock_serial.write.call_args[0][0]
        assert len(written) == 12
        assert written[1] == 37  # CMD_DRIVE_M1M2_SPEED
        m1, m2 = struct.unpack(">ii", written[2:10])
        assert m1 == 500
        assert m2 == -300

    def test_read_encoder_unsigned(self, make_interface, crc16):
        """Verify encoder count is parsed as unsigned int32."""
        rc, mock_serial = make_interface()

        count = 4294900000
        payload = struct.pack(">I", count) + b"\x00"
        header = bytes([0x80, 16])
        resp_crc = crc16(header + payload)
        mock_serial.read.return_value = payload + struct.pack(">H", resp_crc)

        enc, status = rc.read_encoder_m1()
        assert enc == count

    def test_read_battery_voltage(self, make_interface, crc16):
        """Verify battery voltage conversion (raw / 10.0)."""
        rc, mock_serial = make_interface()

        raw = 122
        payload = struct.pack(">H", raw)
        header = bytes([0x80, 24])
        resp_crc = crc16(header + payload)
        mock_serial.read.return_value = payload + struct.pack(">H", resp_crc)

        voltage = rc.read_main_battery()
        assert abs(voltage - 12.2) < 0.01

    def test_read_temperature(self, make_interface, crc16):
        """Verify temperature conversion (raw / 10.0)."""
        rc, mock_serial = make_interface()

        raw = 273
        payload = struct.pack(">H", raw)
        header = bytes([0x80, 82])
        resp_crc = crc16(header + payload)
        mock_serial.read.return_value = payload + struct.pack(">H", resp_crc)

        temp = rc.read_temperature()
        assert abs(temp - 27.3) < 0.01

    def test_set_velocity_pid_packet(self, make_interface, crc16):
        """Verify PID set command packs D, P, I, QPPS in correct order."""
        rc, mock_serial = make_interface()
        mock_serial.read.return_value = b"\xff"

        rc.set_velocity_pid_m1(p=15000, i=750, d=100, qpps=2796)

        written = mock_serial.write.call_args[0][0]
        assert len(written) == 20
        assert written[1] == 28  # CMD_SET_M1_VEL_PID
        d_val, p_val, i_val, qpps_val = struct.unpack(">IIII", written[2:18])
        assert d_val == 100
        assert p_val == 15000
        assert i_val == 750
        assert qpps_val == 2796

    def test_read_velocity_pid(self, make_interface, crc16):
        """Verify PID read parses P, I, D, QPPS correctly."""
        rc, mock_serial = make_interface()

        p, i, d, qpps = 15000, 750, 100, 2796
        payload = struct.pack(">IIII", p, i, d, qpps)
        header = bytes([0x80, 55])
        resp_crc = crc16(header + payload)
        mock_serial.read.return_value = payload + struct.pack(">H", resp_crc)

        pid = rc.read_velocity_pid_m1()
        assert pid == {"P": p, "I": i, "D": d, "QPPS": qpps}


class TestConnectionManagement:
    def test_is_open_false_initially(self):
        from strafer_driver.roboclaw_interface import RoboClawInterface
        rc = RoboClawInterface("/dev/fake", 0x80)
        assert rc.is_open is False

    def test_close_idempotent(self):
        from strafer_driver.roboclaw_interface import RoboClawInterface
        rc = RoboClawInterface("/dev/fake", 0x80)
        rc.close()
        rc.close()

    def test_reconnect_failure(self):
        from strafer_driver.roboclaw_interface import RoboClawInterface
        rc = RoboClawInterface("/dev/nonexistent_port", 0x80)
        assert rc.reconnect() is False


class TestRetryLogic:
    def test_retry_succeeds_on_second_attempt(self, make_interface, crc16):
        """Retry logic should recover from a single transient failure."""
        rc, mock_serial = make_interface()
        rc.max_retries = 1

        speed_bytes = struct.pack(">i", 500) + b"\x00"
        header = bytes([0x80, 18])
        resp_crc = crc16(header + speed_bytes)
        valid_response = speed_bytes + struct.pack(">H", resp_crc)

        mock_serial.read.side_effect = [b"\x00", valid_response]

        speed, status = rc.read_speed_m1()
        assert speed == 500

    def test_retry_exhausted_raises(self, make_interface):
        """After all retries exhausted, the last exception is raised."""
        from strafer_driver.roboclaw_interface import RoboClawTimeoutError
        rc, mock_serial = make_interface()
        rc.max_retries = 2

        mock_serial.read.return_value = b""

        with pytest.raises(RoboClawTimeoutError):
            rc.read_speed_m1()

        assert mock_serial.read.call_count == 3

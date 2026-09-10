import struct
import unittest

import can

from probe_dm_j4340 import Register
from probe_dm_j4340_canable import (
    build_disable_request,
    build_register_request,
    decode_register_message,
    decode_status_message,
)


class CanableProtocolTest(unittest.TestCase):
    def test_disable_request(self):
        message = build_disable_request(3)
        self.assertEqual(message.arbitration_id, 3)
        self.assertFalse(message.is_extended_id)
        self.assertEqual(bytes(message.data), bytes([0xFF] * 7 + [0xFD]))

        velocity_mode = build_disable_request(3, 3)
        self.assertEqual(velocity_mode.arbitration_id, 0x203)

    def test_read_request(self):
        message = build_register_request(0x123, 0x3C)
        self.assertEqual(message.arbitration_id, 0x7FF)
        self.assertFalse(message.is_extended_id)
        self.assertEqual(bytes(message.data), bytes([0x23, 0x01, 0x33, 0x3C]))

    def test_float_reply(self):
        payload = bytes([2, 0, 0x33, 0x3C]) + struct.pack("<f", 24.5)
        message = can.Message(arbitration_id=0x12, data=payload, is_extended_id=False)
        result = decode_register_message(message, 2, Register(0x3C, "bus_voltage", "f32"))
        self.assertIsNotNone(result)
        assert result is not None
        value, raw = result
        self.assertAlmostEqual(value, 24.5)
        self.assertEqual(raw, "012#0200333C0000C441")

    def test_transmit_echo_is_ignored(self):
        payload = bytes([2, 0, 0x33, 0x3C]) + struct.pack("<f", 24.5)
        message = can.Message(
            arbitration_id=0x7FF, data=payload, is_extended_id=False, is_rx=False
        )
        result = decode_register_message(message, 2, Register(0x3C, "bus_voltage", "f32"))
        self.assertIsNone(result)

    def test_status_reply(self):
        reply = can.Message(
            arbitration_id=0x12,
            data=bytes([0x12, 0x80, 0x00, 0x80, 0x08, 0x00, 35, 36]),
            is_extended_id=False,
            is_rx=True,
        )
        status = decode_status_message(reply, 2)
        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status["error"], 1)
        self.assertEqual(status["response_can_id"], 0x12)
        self.assertAlmostEqual(status["velocity_rad_s"], 45 / 4095, places=6)

    def test_can_error_frame_is_not_motor_feedback(self):
        message = can.Message(
            arbitration_id=0,
            data=bytes.fromhex("0000000000000800"),
            is_extended_id=False,
            is_error_frame=True,
            is_rx=True,
        )
        self.assertIsNone(decode_status_message(message, 0))


if __name__ == "__main__":
    unittest.main()

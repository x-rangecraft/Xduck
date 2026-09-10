import unittest

import can

from collect_i2rt import (
    Command,
    MotorProfile,
    build_parser,
    decode_feedback,
    float_to_uint,
    pack_mit_command,
    validate_args,
)


PROFILE = MotorProfile(13, 29, 122, 1, 12.5, 10.0, 28.0)


class CollectProtocolTest(unittest.TestCase):
    def test_midpoint_mapping(self):
        self.assertEqual(float_to_uint(0.0, -10.0, 10.0, 12), 2048)

    def test_zero_mit_command(self):
        message = pack_mit_command(PROFILE, Command())
        self.assertEqual(message.arbitration_id, 13)
        self.assertEqual(message.dlc, 8)
        self.assertEqual(bytes(message.data), bytes.fromhex("8000800000000800"))

    def test_feedback_uses_discovered_ranges(self):
        message = can.Message(
            arbitration_id=29,
            data=bytes.fromhex("0D80008008002423"),
            is_extended_id=False,
            is_rx=True,
        )
        decoded = decode_feedback(message, PROFILE)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertAlmostEqual(decoded["qd_feedback"], 10.0 / 4095, places=6)
        self.assertAlmostEqual(decoded["tau_feedback"], 28.0 / 4095, places=6)
        self.assertEqual(decoded["mos_temperature"], 36)
        self.assertEqual(decoded["motor_temperature"], 35)
        self.assertEqual(decoded["driver_state"], 0)
        self.assertEqual(decoded["fault_state"], 0)

    def test_enabled_state_is_not_a_fault(self):
        message = can.Message(
            arbitration_id=29,
            data=bytes.fromhex("1D80008008002423"),
            is_extended_id=False,
            is_rx=True,
        )
        decoded = decode_feedback(message, PROFILE)
        assert decoded is not None
        self.assertEqual(decoded["driver_state"], 1)
        self.assertEqual(decoded["fault_state"], 0)

    def test_wrong_feedback_id_is_ignored(self):
        message = can.Message(arbitration_id=30, data=bytes(8), is_extended_id=False, is_rx=True)
        self.assertIsNone(decode_feedback(message, PROFILE))

    def test_gf43x40_10_physical_peak_is_enforced(self):
        args = build_parser().parse_args(
            ["torque_pulse", "--ids", "13", "--arm", "--high-torque-arm", "--torques", "23.6"]
        )
        with self.assertRaisesRegex(ValueError, "23.5"):
            validate_args(args)

    def test_gf43x40_10_table_speed_limit_is_enforced(self):
        args = build_parser().parse_args(
            ["velocity_scan", "--ids", "13", "--arm", "--speeds", "5.82"]
        )
        with self.assertRaisesRegex(ValueError, "5.8095"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()

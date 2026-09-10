import unittest

from build_j4340_params import build_torque_speed_envelope, fit_peak_region_extrapolation


class PeakRegionFitTest(unittest.TestCase):
    def test_fit_is_continuous_monotonic_and_constrained(self):
        fit = fit_peak_region_extrapolation()
        points = fit["points"]
        self.assertAlmostEqual(points[0]["torque_Nm"], 17.86)
        self.assertAlmostEqual(points[0]["speed_rpm_nominal"], 30.6, places=9)
        self.assertAlmostEqual(points[-1]["torque_Nm"], 23.5)
        speeds = [point["speed_rpm_nominal"] for point in points]
        self.assertTrue(all(a >= b >= 0.0 for a, b in zip(speeds, speeds[1:])))
        self.assertTrue(
            all(point["mechanical_power_W_nominal"] <= 70.0 for point in points)
        )

    def test_cubic_fit_quality(self):
        fit = fit_peak_region_extrapolation()
        self.assertGreater(fit["full_table_fit_r_squared"], 0.999)
        self.assertLess(fit["full_table_fit_rmse_rpm"], 0.2)

    def test_inverse_torque_speed_envelope_is_not_rectangular(self):
        envelope = build_torque_speed_envelope(fit_peak_region_extrapolation())
        speeds = envelope["speed_rad_s_ascending"]
        torques = envelope["max_accelerating_torque_Nm"]
        self.assertTrue(all(a < b for a, b in zip(speeds, speeds[1:])))
        self.assertTrue(all(a >= b for a, b in zip(torques, torques[1:])))
        self.assertEqual(torques[0], 23.5)
        self.assertEqual(torques[-1], 0.0)


if __name__ == "__main__":
    unittest.main()

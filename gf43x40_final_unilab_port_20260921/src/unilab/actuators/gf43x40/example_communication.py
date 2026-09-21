"""Standalone transport example; requires NumPy and the two adjacent JSON files.

The zero observations below represent a stationary motor, not a physics model.
Run this file directly from a copy of the four-file sharing bundle.
"""

import numpy as np

from communication import GF43X40Communication, MITCommand


def main() -> None:
    link = GF43X40Communication(
        num_envs=1,
        num_motors=1,
        command_delay_s=0.0,
        feedback_delay_s=0.0,
        jitter_mode="measured_20260918",
        command_jitter_ms=(0, 0),  # Uniform ranges remain zero in measured mode.
        feedback_jitter_ms=(0, 0),
        jitter_seed=0,
    )
    zeros = np.zeros((1, 1))
    previous_seq = 0
    previous_feedback_time = None
    for tick in range(60):
        t = tick * .001
        if tick % 20 == 0:
            target = np.full((1, 1), .1 * (tick // 20 + 1))
            link.submit(MITCommand(target, zeros, np.full((1, 1), 60.),
                                   np.full((1, 1), 2.), zeros), at_s=t)
        active = link.advance(t)
        if link.active_seq != previous_seq:
            print(f"{t * 1000:5.0f} ms: command {link.active_seq}, target={active.q_des[0, 0]:.4f}")
            previous_seq = link.active_seq
        # In a simulator: integrate physics with active, then supply the actual
        # post-step position, observed velocity and estimated torque here.
        feedback = link.sample(t + .001, zeros, zeros, zeros)
        if feedback is not None and feedback.sample_time_s != previous_feedback_time:
            print(f"{(t + .001) * 1000:5.0f} ms: feedback sampled at "
                  f"{feedback.sample_time_s * 1000:.0f} ms")
            previous_feedback_time = feedback.sample_time_s


if __name__ == "__main__":
    main()

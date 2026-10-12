"""Shape-preserving C2 joint paths with nonzero through-waypoint velocities."""

import numpy as np


class ContinuousMotorPath:
    """Batch of quintic Hermite paths; position, velocity and acceleration match.

    Interior speeds use a harmonic secant mean, capped at 1.2 times the smaller
    secant. The resulting quintic Bezier control points remain ordered within
    each monotone segment (v0 + v1 <= 2.4 * secant < 2.5 * secant), so there is
    no position overshoot. Only actual direction reversals stop that joint.
    """

    def __init__(self, nodes, durations):
        self.nodes = np.asarray(nodes, dtype=float)
        self.durations = np.asarray(durations, dtype=float)
        if self.nodes.ndim == 2:
            self.nodes = self.nodes[None]
        if self.durations.ndim == 1:
            self.durations = self.durations[None]
        if self.nodes.shape[:2] != (len(self.durations), self.durations.shape[1] + 1):
            raise ValueError("path dimensions do not agree")
        if (
            not np.all(np.isfinite(self.nodes))
            or not np.all(np.isfinite(self.durations))
            or np.any(self.durations <= 0)
        ):
            raise ValueError("path requires finite nodes and positive finite durations")
        self.knots = np.c_[np.zeros(len(self.nodes)), np.cumsum(self.durations, axis=1)]
        secant = np.diff(self.nodes, axis=1) / self.durations[:, :, None]
        velocity = np.zeros_like(self.nodes)
        a, b = secant[:, :-1], secant[:, 1:]
        same = a * b > 0
        harmonic = np.divide(2 * a * b, a + b, out=np.zeros_like(a), where=same)
        velocity[:, 1:-1] = np.sign(harmonic) * np.minimum(
            abs(harmonic), 1.2 * np.minimum(abs(a), abs(b))
        )
        self.velocity = velocity
        h = self.durations[:, :, None]
        delta = np.diff(self.nodes, axis=1)
        v0, v1 = h * velocity[:, :-1], h * velocity[:, 1:]
        self.coefficients = (
            self.nodes[:, :-1],
            v0,
            10 * delta - 6 * v0 - 4 * v1,
            -15 * delta + 8 * v0 + 7 * v1,
            6 * delta - 3 * v0 - 3 * v1,
        )

    def sample(self, elapsed):
        elapsed = np.atleast_1d(elapsed).astype(float)
        n = max(len(elapsed), len(self.nodes))
        q = np.broadcast_to(self.nodes[:, 0], (n, self.nodes.shape[-1])).copy()
        for j, _ in enumerate(self.durations[0]):
            s = np.clip((elapsed - self.knots[:, j]) / self.durations[:, j], 0, 1)[:, None]
            p0, v0, c3, c4, c5 = [v[:, j] for v in self.coefficients]
            value = p0 + v0 * s + c3 * s**3 + c4 * s**4 + c5 * s**5
            active = elapsed >= self.knots[:, j]
            q[active] = value[active]
        return q

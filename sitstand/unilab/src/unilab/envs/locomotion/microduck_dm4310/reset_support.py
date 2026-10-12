"""Cold-compiled collision geometry for physically valid randomized reset poses.

The MuJoCo model is consumed only during construction. Reset-time forward
kinematics uses detached NumPy arrays; no asset or backend metadata is accessed.
"""

from __future__ import annotations

import numpy as np


def quaternion_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*q.shape[:-1], 3, 3)


class ResetGroundClearance:
    def __init__(self, model_file: str, ground_name: str):
        import mujoco

        self._free_joint_type = int(mujoco.mjtJoint.mjJNT_FREE)
        model = mujoco.MjModel.from_xml_path(model_file)
        ground = model.geom(ground_name).id
        self.ground_z = float(model.geom_pos[ground, 2])
        if model.geom_type[ground] != mujoco.mjtGeom.mjGEOM_PLANE or not np.allclose(
            model.geom_quat[ground], [1, 0, 0, 0]
        ):
            raise ValueError("MicroDuck reset clearance requires a horizontal ground plane")
        self.bodies = []
        for body in range(1, model.nbody):
            joints = range(
                model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]
            )
            joint_data = []
            for joint in joints:
                kind = int(model.jnt_type[joint])
                adr = int(model.jnt_qposadr[joint])
                if kind not in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE):
                    raise ValueError("MicroDuck clearance supports free and hinge joints only")
                axis = model.jnt_axis[joint].copy()
                x, y, z = axis
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                joint_data.append(
                    (
                        kind,
                        adr,
                        float(model.qpos0[adr]),
                        model.jnt_pos[joint].copy(),
                        skew,
                        skew @ skew,
                    )
                )
            points = []
            for geom in range(
                model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body]
            ):
                collides = (model.geom_contype[geom] & model.geom_conaffinity[ground]) or (
                    model.geom_conaffinity[geom] & model.geom_contype[ground]
                )
                if not collides:
                    continue
                if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = model.geom_dataid[geom]
                    start = model.mesh_vertadr[mid]
                    count = model.mesh_vertnum[mid]
                    vertices = model.mesh_vert[start : start + count]
                elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
                    # Every linear height extremum of a box is at a corner.
                    vertices = np.array(
                        [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
                    ) * model.geom_size[geom]
                else:
                    raise ValueError(
                        "Re-audit reset support when robot collision primitives change"
                    )
                rotation = quaternion_matrix(model.geom_quat[geom])
                points.append(
                    vertices @ rotation.T + model.geom_pos[geom]
                )
            cloud = np.concatenate(points) if points else np.empty((0, 3))
            self.bodies.append(
                (
                    int(model.body_parentid[body]),
                    model.body_pos[body].copy(),
                    quaternion_matrix(model.body_quat[body]),
                    tuple(joint_data),
                    cloud,
                )
            )

    def minimum_z(self, qpos: np.ndarray) -> np.ndarray:
        n = len(qpos)
        rotations = [np.broadcast_to(np.eye(3), (n, 3, 3))]
        positions = [np.zeros((n, 3))]
        minimum = np.full(n, np.inf)
        for parent, local_pos, local_rot, joints, cloud in self.bodies:
            rotation = rotations[parent] @ local_rot
            position = positions[parent] + np.einsum("nij,j->ni", rotations[parent], local_pos)
            for kind, adr, reference, anchor, skew, skew2 in joints:
                if kind == self._free_joint_type:
                    position = qpos[:, adr : adr + 3].copy()
                    rotation = quaternion_matrix(qpos[:, adr + 3 : adr + 7])
                else:
                    angle = qpos[:, adr] - reference
                    change = (
                        np.eye(3)
                        + np.sin(angle)[:, None, None] * skew
                        + (1 - np.cos(angle))[:, None, None] * skew2
                    )
                    rotated_anchor = np.einsum("nij,j->ni", change, anchor)
                    position += np.einsum("nij,nj->ni", rotation, anchor - rotated_anchor)
                    rotation = rotation @ change
            rotations.append(rotation)
            positions.append(position)
            if len(cloud):
                # Bound temporary memory even when all 4096 environments reset.
                for start in range(0, len(cloud), 256):
                    z = rotation[:, 2, :] @ cloud[start : start + 256].T
                    minimum = np.minimum(minimum, np.min(z, axis=1) + position[:, 2])
        return minimum

    def lift_to_clearance(self, qpos: np.ndarray, clearance_m: float = 0.005) -> None:
        qpos[:, 2] += np.maximum(self.ground_z + clearance_m - self.minimum_z(qpos), 0.0)

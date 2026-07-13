"""Synthetic 'capture system' — stands in for a real RGB-D depth-camera
pose estimator. Uses simulator ground truth plus injected Gaussian noise
in place of an actual vision pipeline; documented explicitly so the
scope is never overstated.

Environment-specific note (AdroitHandRelocateSparse-v1, gymnasium==0.29.1,
gymnasium-robotics==1.2.4, mujoco==2.3.7):

The target object in this env/version combo is NOT attached to the model
with a free/ball (quaternion) joint. It has 6 independent DOF: three
translational slide joints (OBJTx, OBJTy, OBJTz) followed by three
rotational hinge joints (OBJRx, OBJRy, OBJRz), each a plain scalar angle
-- not quaternion components. This was verified in Task 1 by inspecting
joint names/types directly on the MuJoCo model; there is no 7-DOF
free joint for the object anywhere in the model.

Consequently the last 6 entries of `qpos` decompose as:
    qpos[-6:-3] -> object position (x, y, z)
    qpos[-3:]   -> object rotation angles (OBJRx, OBJRy, OBJRz), in that order

`shared/schema/capture_schema.json` is read-only and requires `orientation`
to be a 4-element (quaternion-shaped) array. To satisfy the schema exactly
as written, the 3 scalar hinge angles are converted into a derived unit
quaternion below. This quaternion is NOT a native simulator quantity --
it is reconstructed from 3 independent hinge angles for schema compliance
only. The specific axis-order convention chosen is documented at the
call site in `estimate_object_pose`.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def estimate_object_pose(env, noise_std: float = 0.005, seed: int | None = None) -> tuple[list[float], list[float]]:
    """Estimate the target object's pose from simulator ground truth plus
    injected Gaussian noise, standing in for a real depth-camera pose
    estimator.

    Position comes from qpos[-6:-3] (OBJTx, OBJTy, OBJTz slide joints).
    Orientation comes from qpos[-3:] (OBJRx, OBJRy, OBJRz hinge joints,
    each a scalar angle in radians -- not quaternion components). Those
    3 angles are converted to a unit quaternion via
    `Rotation.from_euler("xyz", angles)`, i.e. treating (OBJRx, OBJRy,
    OBJRz) as extrinsic Euler angles applied in x, then y, then z order
    (scipy's lowercase axis string denotes extrinsic composition;
    uppercase would mean intrinsic).
    This is the best available interpretation for schema compliance; it
    has not been verified against MuJoCo's exact joint-axis semantics
    beyond the confirmed axis order OBJRx, OBJRy, OBJRz.
    """
    rng = np.random.default_rng(seed)
    unwrapped = env.unwrapped
    qpos = unwrapped.data.qpos

    position = qpos[-6:-3].copy()
    position = position + rng.normal(0, noise_std, size=3)

    rotation_angles = qpos[-3:].copy()
    # Explicit, documented axis-order choice: 'xyz' (lowercase = extrinsic
    # in scipy's convention) Euler angles applied to (OBJRx, OBJRy, OBJRz)
    # in that order. MuJoCo's exact intrinsic/extrinsic convention for
    # these 3 independent hinge joints is not verified beyond their axis
    # order; this is a reasonable, explicit choice made solely to produce
    # a schema-valid quaternion.
    quaternion_xyzw = Rotation.from_euler("xyz", rotation_angles).as_quat()
    orientation = quaternion_xyzw

    return position.tolist(), orientation.tolist()


def capture_frame(
    env,
    frame_id: int,
    timestamp: float,
    object_id: str = "target_object",
    width: int = 128,
    height: int = 128,
    fov_deg: float = 45.0,
    noise_std: float = 0.005,
    seed: int | None = None,
) -> dict:
    position, orientation = estimate_object_pose(env, noise_std=noise_std, seed=seed)
    confidence = float(np.clip(1.0 - noise_std * 20, 0.5, 0.99))
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "camera": {"width": width, "height": height, "fov_deg": fov_deg},
        "detections": [
            {
                "object_id": object_id,
                "position": position,
                "orientation": orientation,
                "confidence": confidence,
            }
        ],
    }

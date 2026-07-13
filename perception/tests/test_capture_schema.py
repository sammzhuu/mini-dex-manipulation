# perception/tests/test_capture_schema.py
import json
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "shared" / "schema" / "capture_schema.json"
FIXTURE_PATH = REPO_ROOT / "shared" / "fixtures" / "capture_sample.json"


def test_fixture_matches_schema():
    schema = json.loads(SCHEMA_PATH.read_text())
    fixture = json.loads(FIXTURE_PATH.read_text())
    jsonschema.validate(instance=fixture, schema=schema)


def test_capture_frame_matches_schema():
    gym = pytest.importorskip("gymnasium")
    gymnasium_robotics = pytest.importorskip("gymnasium_robotics")
    gym.register_envs(gymnasium_robotics)
    from capture import capture_frame

    env = gym.make("AdroitHandRelocateSparse-v1")
    env.reset(seed=0)
    schema = json.loads(SCHEMA_PATH.read_text())
    frame = capture_frame(env, frame_id=0, timestamp=0.0, seed=0)
    jsonschema.validate(instance=frame, schema=schema)
    env.close()


def test_pose_matches_named_joints():
    """Regression guard for the Task 1 qpos-slicing bug: looks up each
    object joint's qpos address by NAME (independent of capture.py's own
    slice indices), so a future accidental reversion to the wrong slice
    (e.g. qpos[-7:-4]/qpos[-4:]) would fail this test even though it still
    produces a 3-number position and a 4-number orientation.

    Steps the env with a few seeded actions first: at a fresh reset every
    object DOF (including the hand joints a wrong slice would contaminate
    position/orientation with) is exactly zero, so a norm-only orientation
    check would trivially pass regardless of which 3 qpos entries were
    read. Non-zero angles make the by-name comparison meaningful."""
    gym = pytest.importorskip("gymnasium")
    gymnasium_robotics = pytest.importorskip("gymnasium_robotics")
    gym.register_envs(gymnasium_robotics)
    from capture import capture_frame

    env = gym.make("AdroitHandRelocateSparse-v1")
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.action_space.sample())
    model = env.unwrapped.model
    expected_position = np.array(
        [env.unwrapped.data.qpos[model.joint(name).qposadr[0]] for name in ("OBJTx", "OBJTy", "OBJTz")]
    )
    expected_angles = np.array(
        [env.unwrapped.data.qpos[model.joint(name).qposadr[0]] for name in ("OBJRx", "OBJRy", "OBJRz")]
    )
    expected_orientation = Rotation.from_euler("xyz", expected_angles).as_quat()

    noise_std = 0.005
    frame = capture_frame(env, frame_id=0, timestamp=0.0, noise_std=noise_std, seed=0)
    position = np.array(frame["detections"][0]["position"])
    orientation = np.array(frame["detections"][0]["orientation"])

    # Position must be within a small multiple of the injected noise std of
    # the named-joint ground truth -- proves capture_frame reads OBJTx/Ty/Tz,
    # not some other 3 qpos entries (e.g. a hand joint).
    assert np.all(np.abs(position - expected_position) < 10 * noise_std)

    # Orientation must match the quaternion derived from the named OBJRx/Ry/Rz
    # joints specifically -- a norm-only check would pass for ANY 3 angles
    # (Rotation.as_quat() always returns a unit quaternion), so this proves
    # capture_frame reads the correct 3 rotation joints, not e.g. a hand joint.
    assert orientation.shape == (4,)
    assert np.isclose(np.linalg.norm(orientation), 1.0, atol=1e-6)
    assert np.allclose(orientation, expected_orientation, atol=1e-9)

    env.close()

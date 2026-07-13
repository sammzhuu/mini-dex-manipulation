import argparse
import json
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics

from capture import capture_frame

gym.register_envs(gymnasium_robotics)

TASK_ID = "AdroitHandRelocateSparse-v1"


def dump_sequence(num_frames: int, seed: int, out_path: Path) -> Path:
    env = gym.make(TASK_ID)
    env.reset(seed=seed)
    frames = []
    for i in range(num_frames):
        action = env.action_space.sample()
        env.step(action)
        frames.append(capture_frame(env, frame_id=i, timestamp=i / 30.0, seed=seed * 1000 + i))
    env.close()
    out_path.write_text(json.dumps(frames, indent=2))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "artifacts" / "capture_sequence.json")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    path = dump_sequence(args.frames, args.seed, args.out)
    print(f"Wrote {path}")

# Mini-Dex — Simulated Dexterous Manipulation PoC Toolkit

A simulated, end-to-end toolkit for deploying a dexterous-manipulation policy the way a forward-deployed robotics engineer would package one for a customer proof-of-concept. A trained policy, a perception "capture system," a robotics-middleware bridge, and internal ops tooling; built entirely in simulation, with zero physical hardware.

## Why Build?

Robotics foundation models don't ship value on their own; someone has to wrap them with the perception, middleware, and internal tooling a real deployment needs: sensor capture, robot integration, monitoring, and reporting back to the people who built the model. This project is a piece built around that idea, inspired by forward-deployed robotics engineering roles that sit between research and customer-facing deployment.

## Demo

<video src="media/demo.mp4" controls width="480"></video>

One evaluation episode of the policy attempting the AdroitHand dexterous relocate task in MuJoCo, rendered directly from the same rollout that produced the metrics below. (If the video doesn't render here, it's at `media/demo.mp4`.)

## How it works

Four components form a pipeline, each communicating through one shared JSON contract (`shared/schema/`) rather than importing each other directly — that's what let all four be built independently and in parallel instead of serially:

1. **`sim_policy`** trains a policy in a MuJoCo dexterous-hand environment (Gymnasium-Robotics' `AdroitHand`), evaluates it over a batch of episodes, and exports a schema-validated `rollout.json` plus a recorded demo video.
2. **`perception`** reads the same simulator state and emits synthetic sensor frames — object position, orientation, and a confidence score — in the exact schema a real RGB-D depth-camera pose estimator would produce.
3. **`ros2_bridge`** packages a ROS2 (Humble) node pair — a capture publisher and a policy-command subscriber — over that same schema, containerized in Docker, so the pipeline's message contract is something that could plug into a real ROS2-based robot stack.
4. **`ops_dashboard`** reads the exported rollout, visualizes success rate / reward / episode length, replays the demo video, and generates a markdown "PoC deployment report" — the kind of internal tool an FDE uses to package a demo for stakeholders and hand results back to the research team.

## Architecture

```
mini-dex-manipulation/
├── sim_policy/       # MuJoCo (AdroitHand) env + PPO policy training + rollout export
├── perception/        # Synthetic RGB-D "capture system" (pose estimation from sim state)
├── ros2_bridge/         # ROS2 (Humble) package: capture + policy-command nodes over topics
├── ops_dashboard/         # Streamlit dashboard: rollout metrics + PoC deployment report generator
├── shared/                 # Cross-lane JSON contract (schemas + fixtures) all 4 components validate against
├── media/                    # Demo video referenced by this README
└── docs/                       # Design spec, implementation plan, per-component task breakdowns
```

## Training & results

The task is `AdroitHandRelocate-v1` — the dense-reward variant of Gymnasium-Robotics' Adroit relocate task, where a 24-DOF simulated hand has to pick up an object and move it within 0.1m of a target position floating up to 0.35m in the air. This repo tried two approaches, and the honest story of what did and didn't work is worth stating plainly:

**From-scratch PPO (didn't reach success).** Training with `stable-baselines3`'s PPO plus `VecNormalize`, a 256x256 network, gSDE exploration, and true multi-process parallelism (`SubprocVecEnv`, ~700 fps sustained on a laptop CPU) for 10M timesteps got mean reward to +50.9 (up from -20 on an early sparse-reward attempt, then +30.3 on a smaller first dense-reward run) — a clear, measured sign of learning. But **success rate stayed 0/25** the entire time. Per-episode diagnostics showed why: the policy reliably lifts the object (20/25 episodes) and gets reward from approaching it, but its closest approach to the target stayed at a statistically unchanged ~0.33m mean regardless of 6.7x more training data and a larger network — a genuine capacity/exploration ceiling on the actual hard part of the task (precise 3D placement), not a reward-shaping or tuning problem. This matches the literature: published Adroit baselines generally rely on demonstration data rather than pure from-scratch on-policy RL to reach non-trivial success.

**Behavior cloning from expert demonstrations (solved it).** Rather than keep tuning from-scratch RL, the shipped policy is trained via behavior cloning on 2000 episodes from Farama's [Minari](https://minari.farama.org/) `D4RL/relocate/expert-v2` dataset — 5000 expert human-generated trajectories for this exact environment (same `AdroitHandRelocate-v1` id, same 39-dim observation and 30-dim action spaces, no adaptation needed), 96.5% of which reach the target at some point. Supervised regression toward the demonstrated actions (`sim_policy/bc_pretrain.py`), with no RL fine-tuning on top, reaches **25/25 success** over the same 25-episode evaluation, with a mean closest-approach distance of 0.0125m — an order of magnitude inside the 0.1m threshold. PPO fine-tuning on top of the BC checkpoint was deliberately not attempted: the same PPO configuration had shown late-training instability on the from-scratch run (reward regressions, exploding KL-divergence), and applying it to an already-solved policy risked degrading it for an uncertain gain rather than improving it.

The result is genuinely a demonstration-augmented policy, not an RL-discovered one — worth being explicit about, since it's a different (and here, more effective) technique than what the rest of this section describes trying first.

## Sensor data sourcing (perception)

`perception`'s capture module reads the manipulated object's pose directly from the MuJoCo simulation state. This particular environment represents the object with 6 separate joints (3 slide + 3 hinge) rather than a single free/quaternion joint, so the module converts the 3 hinge angles into a quaternion to match the shared schema. Gaussian noise and a derived confidence score are then added on top of that ground truth, standing in for the measurement error a real depth-camera pose estimator would have. The point of this design is that the rest of the pipeline (the ROS2 bridge, the dashboard) consumes exactly the same JSON shape a real vision pipeline would produce, so swapping in an actual image-based estimator later is a drop-in replacement, not a redesign.

## ROS2 bridge design

The `capture_node` and `policy_node` in `ros2_bridge` communicate over `std_msgs/String` topics carrying JSON payloads that match the shared schema, rather than typed custom ROS2 `.msg` interfaces — a deliberate simplification for this project's scope (a production version would define the messages via a `rosidl` interfaces package). The pipeline is exercised through a `replay.py` script that feeds recorded capture data through the real, built nodes, demonstrating the full message contract and node wiring end-to-end without needing a second live simulator process running alongside it.

## Setup

Each component has its own `requirements.txt` and is meant to be developed independently:

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; `source .venv/bin/activate` on macOS/Linux
pip install -r sim_policy/requirements.txt      # or perception/, ops_dashboard/
```

`ros2_bridge` requires ROS2 Humble, which doesn't install natively on Windows — see `docs/tasks/03-ros2-bridge.md` for the Docker-based setup used here.

## License

MIT — see `LICENSE`.

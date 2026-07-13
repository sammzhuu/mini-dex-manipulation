# Mini-Dex — Simulated Dexterous Manipulation PoC Toolkit

A simulated, end-to-end toolkit for deploying a dexterous-manipulation policy the way a forward-deployed robotics engineer would package one for a customer proof-of-concept — a trained policy, a perception "capture system," a robotics-middleware bridge, and internal ops tooling — built entirely in simulation, with zero physical hardware.

## Why

Robotics foundation models don't ship value on their own — someone has to wrap them with the perception, middleware, and internal tooling a real deployment needs: sensor capture, robot integration, monitoring, and reporting back to the people who built the model. This project is a portfolio piece built around that idea, inspired by forward-deployed robotics engineering roles that sit between research and customer-facing deployment. It intentionally isn't a research project — it's the scaffolding an FDE actually builds.

## Architecture

```
mini-dex-manipulation/
├── sim_policy/       # MuJoCo (AdroitHand) env + PPO policy training + rollout export
├── perception/        # Synthetic RGB-D "capture system" (pose estimation from sim state)
├── ros2_bridge/         # ROS2 (Humble) package: capture + policy-command nodes over topics
├── ops_dashboard/         # Streamlit dashboard: rollout metrics + PoC deployment report generator
├── shared/                 # Cross-lane JSON contract (schemas + fixtures) all 4 components validate against
└── docs/                    # Design spec, implementation plan, per-component task breakdowns
```

The four components agree on a fixed JSON contract (`shared/schema/`) so each could be built independently in parallel rather than serially.

## Status

This is an active work in progress, not a finished product. Current state, honestly:

| Component | Status |
|---|---|
| `sim_policy` | PPO trained on `AdroitHandRelocate-v1` (dense reward + `VecNormalize`, 1.5M timesteps, ~3.7h on CPU); `rollout.json` + `demo.mp4` exported and schema-validated. Mean reward improved from -20 (the original 200k-timestep `AdroitHandRelocateSparse-v1` MVP run) to +30, but **success rate is still 0% over 25 eval episodes** — the policy learns to approach and lift the object but hasn't reliably crossed the goal-distance success threshold within the 200-step episode budget |
| `perception` | Environment set up; core capture/pose-estimation module not yet implemented |
| `ros2_bridge` | MVP complete: package builds via `colcon` with no errors; capture, policy, and replay nodes implemented and running; pure-logic tests passing (`pytest`, no Docker needed); live capture→policy topic flow and single-command replay both verified end-to-end inside Docker |
| `ops_dashboard` | Data-loading and report-generation logic implemented and unit-tested (all passing); Streamlit UI written, pending a manual run-through |

## Setup

Each component has its own `requirements.txt` and is meant to be developed independently:

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; `source .venv/bin/activate` on macOS/Linux
pip install -r sim_policy/requirements.txt      # or perception/, ops_dashboard/
```

`ros2_bridge` requires ROS2 Humble, which doesn't install natively on Windows — see `docs/tasks/03-ros2-bridge.md` for the Docker-based setup used here.

## What's live vs. simulated/stubbed

In the interest of not overclaiming:

- **Perception** uses simulator ground-truth object pose plus injected Gaussian noise as a stand-in for a real depth-camera pose estimator, not an actual vision pipeline (yet — see that component's task file for the stretch goal of a real image-based estimator).
- **`ros2_bridge`** uses `std_msgs/String` carrying JSON payloads rather than typed custom ROS2 messages, and is exercised via fixture replay rather than a live two-process loop against a running simulator (see `docs/tasks/03-ros2-bridge.md`'s Scope section for the reasoning). `policy_node` publishes zero-vector joint targets — it demonstrates the message contract and node wiring, not a live policy; real joint targets from `sim_policy` are a future integration-pass item. All three are documented, deliberate scope decisions for a short build timeline, not oversights.
- **`ops_dashboard`** auto-detects and displays real `sim_policy` output once it exists; until then it runs against a checked-in fixture so the rest of the pipeline isn't blocked on training.

## License

MIT — see `LICENSE`.

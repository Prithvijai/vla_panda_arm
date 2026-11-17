# Panda ManiSkill VLA

A ManiSkill environments for controlling a Panda robot and connecting VLA clients through ZeroMQ.

<img src=docs/sim_env.png width="450">

## Simulation Server

Run the Panda simulation at 20 Hz with Protobuf action chunks over REQ/REP and RGB observations over PUB/SUB:

```bash
pms-sim-server
```

## View Observations in Simulation

<img src=docs/image_obs.png width="300">

```bash
pms-image-viewer
```


## For Teleoperation

Run the Panda simulation at 20 Hz with Protobuf action chunks over REQ/REP and RGB observations over PUB/SUB:

```bash
pms-teleop
```
<img src=docs/teleop_pick.png width="450">


## Run SmolVLA in the Simulation

The policy runs in its own Python environment (LeRobot) and talks to the sim only through ZeroMQ.

```bash
# one-time: separate env for the policy
conda create -n smolvla python=3.12 -y
conda run -n smolvla pip install "lerobot[smolvla]" pyzmq "protobuf>=6.33.5"
conda run -n smolvla pip install --no-deps -e .

# terminal 1: sim (lockstep = only step when actions arrive, hold finished episodes)
pms-sim-server --lockstep --max-episode-steps 200

# terminal 2 (optional): watch base + wrist camera
pms-image-viewer

# terminal 3: SmolVLA (HuggingFaceVLA/smolvla_libero) on 20 unseen seeds
conda run -n smolvla --live-stream pms-vla --policy smolvla --episodes 20 --videos 3
```

`HuggingFaceVLA/smolvla_libero` is SmolVLA fine-tuned on LIBERO (Franka Panda, two cameras,
7-D delta end-effector actions). The client maps ManiSkill observations into LIBERO's format
(`base_camera`/`hand_camera` → `image`/`image2`, 8-D state) and maps LIBERO actions
(±0.05 m, ±0.5 rad, gripper −1 = open) to ManiSkill's `pd_ee_delta_pose` (±0.1 m, ±0.1 rad,
gripper +1 = open). No ManiSkill data is used, so this is a **zero-shot cross-simulator** test.

### Evaluation

Each run uses fixed seeds (default 1000…), writes `results/<name>.json`, and reports:

| Metric | Meaning |
|---|---|
| Success rate (95% Wilson CI) | `info["success"]`: cube at goal and robot static |
| Grasp rate | episodes where the cube was grasped at least once |
| Progress | mean of the best normalized dense reward per episode (0 to 1, reach → grasp → place) |
| Inference p50 / p95 | wall-clock time per SmolVLA chunk |
| Steps/s | effective control rate of the whole loop |

Run everything (server included) with `scripts/run_benchmark.sh 20`, or by hand:

```bash
pms-vla --policy zero   --episodes 20          # lower bound
pms-vla --policy random --episodes 20          # lower bound
for k in 5 10 25 50; do pms-vla --policy smolvla --episodes 20 --actions-per-chunk $k; done
pms-vla-report results/*.json                  # markdown table
```

### Results: zero-shot `smolvla_libero` on PickCube-v1

20 unseen seeds (1000–1019) per row, 200-step limit, RTX 4070 Laptop (8 GB). "Plain" = `pick up the red cube`,
"LIBERO" = `pick up the red block and place it on the plate`; k = actions executed per predicted chunk.

| Run | Success (95% CI) | Grasp | Progress | Inference p50 / p95 | Steps/s |
|---|---|---|---|---|---|
| Zero actions | 0% (0–16%) | 0% | 0.06 | – | 20.0 |
| Random actions | 0% (0–16%) | 0% | 0.11 | – | 20.0 |
| SmolVLA, plain prompt, k=10 | 0% (0–16%) | 0% | 0.08 | 428 / 480 ms | 10.9 |
| SmolVLA, LIBERO prompt, k=5 | 0% (0–16%) | 0% | 0.08 | 439 / 494 ms | 7.4 |
| SmolVLA, LIBERO prompt, k=10 | 0% (0–16%) | 0% | 0.08 | 435 / 483 ms | 10.8 |
| SmolVLA, LIBERO prompt, k=25 | 0% (0–16%) | 0% | 0.08 | 435 / 493 ms | 14.9 |
| SmolVLA, LIBERO prompt, k=50 | 0% (0–16%) | 0% | 0.08 | 436 / 483 ms | 17.0 |

Findings:

- **Zero-shot cross-simulator transfer fails.** SmolVLA trained on LIBERO does not grasp the
  cube in ManiSkill (0/20 for every setting), even with the same robot and action type. Scene,
  objects, camera placement and frame conventions all differ from its training data.
- **The prompt changes behaviour, not the score.** With a LIBERO-style instruction the policy
  performs a recognisable pick primitive (descend → close gripper → lift) but closes ~13 cm
  above the cube, so the dense reward does not move.
- **Chunk length trades reactivity for control rate.** Inference is ~435 ms per chunk, so
  re-planning every 5 actions gives 7.4 steps/s, every 50 gives 17.0 steps/s (sim target 20 Hz).
  (An earlier run without GPU contention measured ~270 ms per chunk.)
- **Next step:** fine-tune on ~100 ManiSkill motion-planner demos (no teleoperation needed) and
  re-run the same benchmark with `--checkpoint <your model>`.

Videos of the first two LIBERO-prompt episodes: `results/videos/`.

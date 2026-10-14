# Panda ManiSkill VLA

A ManiSkill environments for controlling a Panda robot and connecting VLA clients through ZeroMQ.

<img src=docs/sim_env.png width="450">

## Environment

Display the Panda environment with zero actions:

```bash
pms-env-server
```

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


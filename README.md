# Panda ManiSkill VLA Base

A modular ManiSkill environment for Panda control and VLA experiments. The simulator can
run as a real-time ZeroMQ process while teleoperation, policy, and recording processes
connect independently.

## Install

Create and activate a Python 3.10+ environment, then install the project:

```bash
python -m pip install -e ".[dev]"
```

## Usage

### Environment Server

Start the simulator manually in its own terminal:

```bash
pms-env-server
```

The server runs at 20 Hz by default. It binds a `ROUTER` control socket to
`tcp://127.0.0.1:5555` and a `PUB` observation socket to `tcp://127.0.0.1:5556`.

Configure the environment from the command line:

```bash
pms-env-server \
  --env-id PickCube-v1 \
  --robot panda \
  --obs-mode rgbd \
  --control-mode pd_ee_delta_pose \
  --hz 20
```

Use `pms-env-server --help` to see all options. Stop the server with `Ctrl+C`.

The control channel accepts versioned `hello`, `status`, `action`, `reset`, `pause`, and
`resume` messages from a ZeroMQ `DEALER` client. Observations are published as multipart
messages with the `observation` topic, JSON metadata, and raw NumPy array frames. Actions
are consumed once; the server applies a neutral action when no fresh action is queued.

### Direct Access

```python
import numpy as np

from panda_maniskill_vla import EnvironmentConfig, EnvironmentRunner

runner = EnvironmentRunner(EnvironmentConfig())

try:
    observation, info = runner.reset()
    action = np.zeros(7, dtype=np.float32)
    observation, reward, terminated, truncated, info = runner.step(action)
finally:
    runner.close()
```

`EnvironmentRunner` remains independent of ZeroMQ and can still be used directly when
process separation is unnecessary.

## Development

```bash
ruff check .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

# Vision2Action

Vision2Action is a Vision -> Language -> Action robotics prototype built with MuJoCo.

Current capability
- Baseline color-based perception (blue, red, green, brown).
- Natural language command parsing (for example: "go to blue").
- Search -> center -> forward navigation loop with safe stopping.
- Viewer-first runtime: explore scene while sending commands in terminal.

Future goal
- Replace color-threshold perception with real ML-based object grounding and VLA models without changing the control/navigation pipeline.

## Project structure

- `vision2action/` - main package
- `vision2action/perception/` - detector interface + color detector baseline
- `vision2action/control/` - robot kinematics and pose updates
- `vision2action/navigation/` - centering/search policy
- `vision2action/env/scene.xml` - MuJoCo scene
- `tests/` - basic tests
- `examples/` - runnable examples

## Quick start

```bash
# 1) Clone
git https://github.com/Savaliya-Ravi/Vision2Action.git
cd vision2action

# 2) Create and activate environment
python -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run
python -m vision2action.main
```

## Commands in runtime

```text
go to blue
go to red
go to green
go to brown
stop
```

## Version

Current version: `v1.0.0`.

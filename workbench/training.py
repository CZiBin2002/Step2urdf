from pathlib import Path

from .model import identifier

TEMPLATES = Path(__file__).with_name("templates")


def add_training(archive, project, backend, asset_name, *, start_at_zero=False):
    settings = {
        identifier(joint.name): (
            joint.effort,
            joint.lower + (joint.upper - joint.lower) * 0.65 if joint.type != "continuous" else 0.4,
            0.0 if start_at_zero or joint.type == "continuous" else (joint.lower + joint.upper) * 0.5,
        )
        for joint in project.joints if joint.mimic is None and joint.type in {"continuous", "revolute", "prismatic"}
    }
    script = (TEMPLATES / f"{backend}.py").read_text(encoding="utf-8")
    script = script.replace('"__ASSET__"', repr(asset_name)).replace('"__JOINTS__"', repr(settings))
    compile(script, f"train_{backend}.py", "exec")
    archive.writestr(f"train_{backend}.py", script)
    instructions = {
        "isaac": (
            "Isaac Lab v2.3.2 + Isaac Sim 5.1, Stable-Baselines3",
            "<IsaacLab>/isaaclab.sh -p train_isaac.py --headless --num-envs 128 --steps 100000",
            "Windows uses isaaclab.bat in place of isaaclab.sh.",
        ),
        "mjlab": (
            "MJLab (current actor/critic configuration API) + MuJoCo Warp + RSL-RL",
            "uv run --project <mjlab checkout> python train_mjlab.py --num-envs 128 --iterations 200",
            "Run from the extracted ZIP folder with a CUDA-capable MJLab installation.",
        ),
    }
    title, command, notes = instructions[backend]
    archive.writestr("TRAINING.md", f"""# Joint pose tracking with {title}

```sh
{command}
```

{notes} Keep the script and model assets in the same folder.

The PPO baseline observes joint angles/displacements, velocities and target positions.
Actions are scaled by each joint's configured effort limit. The reward encourages
joint-pose tracking and penalizes velocity and control effort. Change JOINTS in the
script to adjust targets or configure task-specific observations and rewards.
This example provides a joint-control training task. Locomotion, manipulation and
other robot behaviors require their own task definitions and reward tuning.
""")

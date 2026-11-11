"""MJLab / MuJoCo Warp. Train RSL-RL PPO on joint pose tracking."""

import argparse
from pathlib import Path

import mujoco
import torch
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, time_out
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg

JOINTS = "__JOINTS__"
ASSET = "__ASSET__"
ASSET_PATH = Path(__file__).with_name(ASSET + ".xml")


def robot_spec():
    return mujoco.MjSpec.from_file(str(ASSET_PATH))


def target_position(env, asset_cfg):
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    selected = [robot.joint_names[index] for index in ids] if not isinstance(ids, slice) else robot.joint_names[ids]
    target = torch.tensor([JOINTS[name][1] for name in selected], device=env.device)
    return target.expand(env.num_envs, -1)


def tracking_reward(env, asset_cfg):
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    target = target_position(env, asset_cfg)
    position = robot.data.joint_pos[:, ids]
    velocity = robot.data.joint_vel[:, ids]
    action = env.action_manager.action
    return (torch.exp(-8 * (position - target).square().mean(dim=-1))
            - 0.001 * velocity.square().mean(dim=-1) - 0.01 * action.square().mean(dim=-1))


def env_cfg(num_envs):
    names = tuple(JOINTS)
    robot = EntityCfg(
        spec_fn=robot_spec,
        articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=names),)),
        init_state=EntityCfg.InitialStateCfg(joint_pos={name: values[2] for name, values in JOINTS.items()},
                                             joint_vel={".*": 0.0}),
    )
    selection = SceneEntityCfg("robot", joint_names=names)
    observations = {
        "position": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": selection}),
        "velocity": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": selection}),
        "target": ObservationTermCfg(func=target_position, params={"asset_cfg": selection}),
    }
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": robot},
                       num_envs=num_envs, env_spacing=4.0),
        observations={"actor": ObservationGroupCfg(observations), "critic": ObservationGroupCfg(dict(observations))},
        actions={"effort": JointEffortActionCfg(entity_name="robot", actuator_names=names,
                                                scale={name: values[0] for name, values in JOINTS.items()})},
        rewards={"tracking": RewardTermCfg(func=tracking_reward, weight=1.0, params={"asset_cfg": selection})},
        terminations={"time_out": TerminationTermCfg(func=time_out, time_out=True)},
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=1 / 120)),
        decimation=4, episode_length_s=6.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    task_id = "Mjlab-Step2urdf-PoseTracking"
    config = RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(128, 128), distribution_cfg={
            "class_name": "GaussianDistribution", "init_std": 0.8, "std_type": "scalar",
        }),
        critic=RslRlModelCfg(hidden_dims=(128, 128)),
        algorithm=RslRlPpoAlgorithmCfg(learning_rate=3e-4),
        max_iterations=args.iterations, num_steps_per_env=32, experiment_name="step2urdf_pose",
        clip_actions=1.0,
    )
    register_mjlab_task(task_id, env_cfg(args.num_envs), env_cfg(1), config)
    launch_training(task_id, TrainConfig(env=env_cfg(args.num_envs), agent=config))


if __name__ == "__main__":
    main()

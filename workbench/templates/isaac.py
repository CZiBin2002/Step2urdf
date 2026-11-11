"""Isaac Lab 2.3.2 / Isaac Sim 5.1. Train PPO on joint pose tracking."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=100000)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
from stable_baselines3 import PPO

import isaaclab.sim as sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

JOINTS = "__JOINTS__"
ASSET = "__ASSET__"
ASSET_PATH = Path(__file__).with_name(ASSET + ".usda")


@configclass
class PoseTrackingCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 6.0
    action_space = len(JOINTS)
    observation_space = 3 * len(JOINTS)
    state_space = 0
    sim = sim.SimulationCfg(dt=1 / 120, render_interval=decimation)
    scene = InteractiveSceneCfg(num_envs=128, env_spacing=4.0, replicate_physics=True)
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim.UsdFileCfg(usd_path=str(ASSET_PATH), activate_contact_sensors=False),
        actuators={
            name: ImplicitActuatorCfg(joint_names_expr=[name], stiffness=0, damping=0,
                                      effort_limit_sim=values[0])
            for name, values in JOINTS.items()
        },
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={name: values[2] for name, values in JOINTS.items()},
            joint_vel={".*": 0},
        ),
    )


class PoseTrackingEnv(DirectRLEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.joint_ids, ordered = self.robot.find_joints(list(JOINTS), preserve_order=True)
        if ordered != list(JOINTS):
            raise RuntimeError(f"Missing controlled joints: {set(JOINTS) - set(ordered)}")
        self.effort = torch.tensor([JOINTS[name][0] for name in ordered], device=self.device)
        self.target = torch.tensor([JOINTS[name][1] for name in ordered], device=self.device)
        self.actions = torch.zeros((self.num_envs, len(ordered)), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        sim.GroundPlaneCfg().func("/World/Ground", sim.GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/Ground"])
        self.scene.articulations["robot"] = self.robot
        light = sim.DomeLightCfg(intensity=1200.0)
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions):
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        self.robot.set_joint_effort_target(self.actions * self.effort, joint_ids=self.joint_ids)

    def _get_observations(self):
        pos = self.robot.data.joint_pos[:, self.joint_ids]
        vel = self.robot.data.joint_vel[:, self.joint_ids]
        return {"policy": torch.cat([pos, vel, self.target.expand_as(pos)], dim=-1)}

    def _get_rewards(self):
        pos = self.robot.data.joint_pos[:, self.joint_ids]
        vel = self.robot.data.joint_vel[:, self.joint_ids]
        error = (pos - self.target).square().mean(dim=-1)
        return torch.exp(-8 * error) - 0.001 * vel.square().mean(dim=-1) - 0.01 * self.actions.square().mean(dim=-1)

    def _get_dones(self):
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        failed = ~torch.isfinite(self.robot.data.joint_pos).all(dim=-1)
        return failed, timed_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        state = self.robot.data.default_root_state[env_ids].clone()
        state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(state[:, 7:], env_ids)
        pos = self.robot.data.default_joint_pos[env_ids].clone()
        vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(pos, vel, env_ids=env_ids)


def main():
    cfg = PoseTrackingCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = Sb3VecEnvWrapper(PoseTrackingEnv(cfg))
    try:
        model = PPO("MlpPolicy", env, n_steps=32, batch_size=32, seed=args.seed, verbose=1, device="cpu")
        model.learn(total_timesteps=args.steps)
        model.save("pose_tracking_isaac")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

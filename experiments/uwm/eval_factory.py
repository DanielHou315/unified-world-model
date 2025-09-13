import os

import hydra
import numpy as np
import torch
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

from experiments.dp.train import maybe_resume_checkpoint
from experiments.utils import set_seed, is_main_process

import pytorch_lightning as pl
from omegaconf import OmegaConf


def get_nut_pose(env_uw):
    nut_state = env_uw.scene["nut"].data.root_state_w
    nut_state[:, :3] = nut_state[:, :3] - env_uw.scene.env_origins
    return nut_state

def collect_rollout(cfg, model, device):
    model.eval()
    model = getattr(model, "module", model)  # unwrap DDP

    from force_tool.utils.isaac_utils import (
        get_default_cfg,
        create_isaac_env
    )
    from force_tool.utils.isaac_utils import PolicyObsManager
    from force_tool.policy.probing_policy import FourStepHardCodeProbingPolicy
    from force_tool.utils.data_utils import stack_dict, to_numpy
    from force_tool.visualization.plot_utils import save_numpy_video

    sim_app, env_cfg = get_default_cfg(cfg)
    env = create_isaac_env(env_cfg)
    env_uw = env.unwrapped
    device = env.device

    # Collect rollouts
    video_dir = os.path.join(cfg.logdir, "videos")
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)

    probing_policy = FourStepHardCodeProbingPolicy(env_cfg)
    pl.seed_everything(env_cfg.seed)

    cam = env_uw.scene["obs_camera"]
    num_envs = env_uw.num_envs
    epi = 0

    # Initial reset
    obs, extras = env.reset()
    obs_mgr = PolicyObsManager(env_uw, env_cfg) 
    
    env_uw.sim.step(render=False)
    # extras["reward"] = env_uw.reward_manager.compute(dt=env_uw.step_dt)
    # accumulated_rewards = []

    def get_action_from_obs(obs_cur):
        modalities = env_cfg.modalities
        obs_tensor = {
            m: torch.tensor(obs_cur[m], device=device) for m in modalities
        }
        # Sample action from model
        actions = model.sample(obs_tensor)
        actions = actions[:,0]
        return actions

    with torch.inference_mode():
        while epi < env_cfg.num_episodes:
            # Pre-step preparation
            frames = []
            obs, extras = env.reset()
            actions = torch.zeros_like(torch.from_numpy(env.action_space.sample()), device=env.device)
            obs, reward, done, extras = env.step(actions)
            extras["reward"] = reward
            obs_mgr.reset(actions, reward)
            if cam is not None:
                frames.append(to_numpy(cam.data.output["rgb"]))

            # Rollout
            for i in tqdm(range(env_cfg.mating_steps), desc="Mating steps"):
                actions = get_action_from_obs(obs_mgr.obs_cur)
                # Sim and process
                obs, reward, done, extras = env.step(actions)
                extras["reward"] = reward
                obs_mgr.process_step(actions, reward)
                if cam is not None:
                    frames.append(to_numpy(cam.data.output["rgb"]))

            # Probing
            if env_cfg.task.startswith("Isaac-Factory"):
                trial_success = env_uw._get_curr_successes(
                    success_threshold=env_uw.cfg_task.success_threshold, check_rot=False
                )
                env_success = trial_success
            else:
                pre_probe_nut_state = get_nut_pose(env_uw)
                probing_actions = probing_policy(obs)
                probe_length = probing_actions.shape[1]
                for i in range(probe_length):
                    obs, reward, done, extras = env.step(probing_actions[:, i])
                    if cam is not None:
                        frames.append(to_numpy(cam.data.output["rgb"]))
                post_probe_nut_state = get_nut_pose(env_uw)
                nut_pose_diff = torch.norm(
                    pre_probe_nut_state[:, :3] - post_probe_nut_state[:, :3], dim=-1
                )
                trial_success = nut_pose_diff < 0.003
                env_success = env.unwrapped.reward_manager._episode_sums["success"] > 0
            obs, extras = env.reset()
            env_uw.sim.step(render=False)
            # reward_dict = {
            #     k: v if torch.is_tensor(v) else torch.tensor(v) for k, v in extras["log"].items()
            # }
            # reward_dict["probe_success"] = trial_success.float().mean()
            # accumulated_rewards.append(reward_dict)
            print(
                f"Trial success: {trial_success.float().mean()}, "
                f"Env success: {env_success.float().mean()}"
            )
            # save frames
            if cam is not None:
                all_frames = np.stack(frames, 1)
                video_path = os.path.join(video_dir, f"interact_video_{epi}.mp4")
                print(f"Saving video to {video_path}")
                save_numpy_video(
                    all_frames, video_path, fps=15, format='mp4'
                )
            print("Finish episode ", epi)
            del frames
            epi += num_envs
    # accumulated_rewards = stack_dict(accumulated_rewards)
    # accumulated_rewards = dict(accumulated_rewards.apply(lambda x: torch.mean(x.float()).item()))
    # # self.logger.experiment.log(accumulated_rewards)
    # print(accumulated_rewards)
    print("Interact finished")
    return {}

    # successes = []
    # for e in trange(
    #     config.num_rollouts, desc="Collecting rollouts", disable=not is_main_process()
    # ):
    #     env.seed(e)
    #     obs = env.reset()
    #     done = False

    #     while not done:
    #         obs_tensor = {
    #             k: torch.tensor(v, device=device)[None] for k, v in obs.items()
    #         }

    #         # Sample action from model
    #         action = model.sample(obs_tensor)[0].cpu().numpy()

    #         # Step environment
    #         obs, reward, done, info = env.step(action)

    #     successes.append(info["success"])
    #     video = env.get_video()
    #     imageio.mimwrite(os.path.join(video_dir, f"{e}.mp4"), video, fps=30)
    #     print(
    #         f"Episode {e} success: {info['success']}, cumulative: {np.mean(successes):.2f}"
    #     )

    # # Compute success rate
    # success_rate = sum(successes) / len(successes)
    # return success_rate


def maybe_collect_rollout(config, step, model, device):
    """Collect rollouts on the main process if it's the correct step."""
    # Skip rollout rollection for pretraining
    if "libero_90" in config.dataset.name:
        return

     # Determine environment to collect in
    task = config.dataset.name
    if task == "factory_nut":
        env_cfg = config.kuka_eval_env
        OmegaConf.set_struct(env_cfg, False)
    elif task == "factory_gear":
        env_cfg = config.gear_eval_env
        OmegaConf.set_struct(env_cfg, False)
    else:
        raise ValueError(f"Unknown task {task}")
    env_cfg.logdir = config.logdir

    if is_main_process() and (
        step % config.rollout_every == 0 or step == (config.num_steps - 1)
    ):
        success_rate = collect_rollout(env_cfg, model, device)
        print(f"Step: {step} success rate: {success_rate}")


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_uwm_factory.yaml"
)
def main(config):
    # Resolve hydra config
    # print(config)
    # config.dataset = "factory_nut"
    # config.exp_id = "0428_nut_his4_a7"
    OmegaConf.resolve(config)
    set_seed(1234)
    device = torch.device(f"cuda:0")

    # Create model
    model = instantiate(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **config.optimizer)
    scheduler = get_scheduler(optimizer=optimizer, **config.scheduler)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    # Resume from checkpoint
    config.resume = True
    step = maybe_resume_checkpoint(config, model, optimizer, scheduler, scaler)
    maybe_collect_rollout(config, 0, model, device)

if __name__ == "__main__":
    main()

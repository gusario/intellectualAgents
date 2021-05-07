from tqdm import tqdm
import time
import retro
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from src.callbacks import ProgressBarManager_new, SaveOnBestTrainingRewardCallback
import argparse
import json
import time
from src.config_processing import GameConfig
from src.agent_loader import AgentLoader


def main(config: str, agent: str):
    with open(config) as fp:
        json_data = json.load(fp)

    config = GameConfig.deserialize(json_data)
    log_dir = config.agents_config[agent]["save_path"]
    if agent == "DQN":
        env = make_atari_env(config.game_name, n_envs=1,
                             seed=0, monitor_dir=log_dir)

    elif agent == "PPO":
        env = make_atari_env(config.game_name, n_envs=8,
                             seed=0, monitor_dir=log_dir)

    else:
        env = make_atari_env(config.game_name, n_envs=16,
                             seed=0, monitor_dir=log_dir)

    env = VecFrameStack(env, n_stack=4)

    agent = AgentLoader.get_agent(agent, config.agents_config, env)

    reward_callback = SaveOnBestTrainingRewardCallback(
        check_freq=100, log_dir=log_dir)

    start_time = time.time()
    steps = 10_000_000
    with ProgressBarManager_new(steps) as progress_callback:
        agent.agent.learn(total_timesteps=steps, callback=[
                          reward_callback, progress_callback])
        # agent.save()
        env.close()

    end_time = time.time() - start_time
    print(f'\n The Training Took {end_time} seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--agent", type=str, required=True)
    args = parser.parse_args()

    main(args.config, args.agent)

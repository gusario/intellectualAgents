from tqdm import tqdm
import time 
import retro
from stable_baselines3.common.vec_env import DummyVecEnv
from src import callbacks
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import EvalCallback
from src.callbacks import ProgressBarManager_new
import argparse
import json
from stable_baselines3.common.env_util import make_vec_env
from src.config_processing import GameConfig
from src.agent_loader import AgentLoader

def main_vs_5(config: str):

    with open(config) as fp:
        json_data = json.load(fp)

    config = GameConfig.deserialize(json_data)
    config.agents_config["PPO"]["save_path"] += "_vs_5"
    config.agents_config["PPO"]["tensorboard"] += "_vs_5"

    env = DummyVecEnv([lambda: retro.make(config.game_name ,state=config.eval_state[0])])
    agent = AgentLoader.get_agent("PPO", config.agents_config, env)
    env.close()


    for st in tqdm(config.train_states, desc='Main Loop'):
        print(st)
        env = DummyVecEnv([lambda: retro.make(config.game_name ,state=st, scenario='scenario')])
        agent.agent.set_env(env)
        agent.agent.learn(total_timesteps=20000)
        agent.save()
        env.close()


def main_vs_time(config: str):
    with open(config) as fp:
        json_data = json.load(fp)

    config = GameConfig.deserialize(json_data)
    config.agents_config["PPO"]["save_path"] += "_vs_time_pt_check"
    config.agents_config["PPO"]["tensorboard"] += "_vs_time_check"

    env = DummyVecEnv([lambda: (retro.make(config.game_name ,state=config.eval_state[0]))])
    agent = AgentLoader.get_agent("PPO", config.agents_config, env)
    env.close()
    env = DummyVecEnv([lambda: (retro.make(config.game_name ,state=config.eval_state[0]))])
    agent.agent.set_env(env)

    with ProgressBarManager_new(1000) as callback:
        agent.agent.learn(1000, callback=callback)
        agent.save()
        env.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()
    if args.mode == "first":
        main_vs_5(args.config)
    if args.mode == "second":
        main_vs_time(args.config)
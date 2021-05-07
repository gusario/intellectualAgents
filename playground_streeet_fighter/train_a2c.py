from tqdm import tqdm
import time 
import retro
from stable_baselines3.common.vec_env import DummyVecEnv
from src import callbacks
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import EvalCallback
from src.callbacks import ProgressBarManager_old, ProgressBarManager_new
import argparse
import json
import time
from src.config_processing import GameConfig
from src.agent_loader import AgentLoader

def main_vs_5(config: str):

    with open(config) as fp:
        json_data = json.load(fp)

    config = GameConfig.deserialize(json_data)
    config.agents_config["A2C"]["save_path"] += "_vs_5"
    config.agents_config["A2C"]["tensorboard"] += "_vs_5"

    env = DummyVecEnv([lambda: retro.make(config.game_name ,state=config.train_states[0])])
    agent = AgentLoader.get_agent("A2C", config.agents_config, env)
    env.close()

    start_time = time.time()
    for st in tqdm(config.train_states, desc='Main Loop'):
        print(st)
        env = DummyVecEnv([lambda: retro.make(config.game_name ,state=st, scenario='scenario')])
        agent.agent.set_env(env)
        agent.agent.learn(total_timesteps=10000)
        agent.save()
        env.close()
    end_time = time.time() - start_time
    print(f'\n The Training Took {end_time} seconds')


def main_vs_time(config: str):
    with open(config) as fp:
        json_data = json.load(fp)

    config = GameConfig.deserialize(json_data)
    config.agents_config["A2C"]["save_path"] += "_vs_time_pt"
    config.agents_config["A2C"]["tensorboard"] += "_vs_time"

    env = DummyVecEnv([lambda: (retro.make(config.game_name ,state=config.eval_state[0]))])
    agent = AgentLoader.get_agent("A2C", config.agents_config, env)
    start_time = time.time()
    with ProgressBarManager_new(40000) as callback:
        agent.agent.learn(total_timesteps=40000, callback=callback)
        agent.save()
        env.close()

    end_time = time.time() - start_time
    print(f'\n The Training Took {end_time} seconds')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()
    if args.mode == "first":
        main_vs_5(args.config)
    if args.mode == "second":
        main_vs_time(args.config)
    

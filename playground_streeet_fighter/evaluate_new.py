import os
import argparse
import retro
import json
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
from src.config_processing import GameConfig
from src.agent_loader import AgentLoader
import pandas as pd

def eval_100_trials(args):
    with open(args.config) as fp:
        json_data = json.load(fp)

    config = GameConfig.deserialize(json_data)
    config.agents_config[args.agent]["save_path"] += "_vs_time_pt.zip"
    env = DummyVecEnv([lambda: retro.make(config.game_name ,state=config.eval_state[1])])
    agent = AgentLoader.get_agent(args.agent, config.agents_config, env, load=True)

    rew_list = []
    trials = 100
    for i in tqdm(range(trials)):
        obs = env.reset()
        done = False
        reward = 0
        while not done:
            actions, _ = agent.agent.predict(obs)
            obs, rew, done, info = env.step(actions)
            reward += rew

        rew_list.append(reward)

    env.close()
    count = sum(i > 0 for i in rew_list)

    print("win percentage = {}%".format(count / trials * 100))
    # df = pd.DataFrame()
    # df["a2c_vs_new_opponent_easy_mode"] = rew_list
    # df.to_csv("evaluate_results/street_fighter/PPO/same_opponent/ppo_vs_same_opponent_hard_mode.csv", index=False)

def eval_time(args):
    with open(args.config) as fp:
        json_data = json.load(fp)

    video_path = os.path.join("./videos", args.agent)
    config = GameConfig.deserialize(json_data)
    config.agents_config[args.agent]["save_path"] += "_vs_time_pt_check.zip"
    env = DummyVecEnv([lambda: retro.make(config.game_name ,state=config.eval_state[1])])
    agent = AgentLoader.get_agent(args.agent, config.agents_config, env, load=True)
    env.close()
    env = DummyVecEnv([lambda: retro.make(config.game_name ,state=config.eval_state[1], record=video_path)])
    obs = env.reset()
    done = False

    while not done:
      actions, _ = agent.agent.predict(obs)
      obs, rew, done, info = env.step(actions)
    #   env.render()

    env.close() 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()
    if args.mode == "first":
        eval_100_trials(args)
    if args.mode == "second":
        eval_time(args)
     
    


if __name__ == "__main__":
    main()
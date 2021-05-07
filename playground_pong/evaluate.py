import os
import argparse
import retro
import gym
import json
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder, DummyVecEnv
from src.config_processing import GameConfig
from src.agent_loader import AgentLoader
from stable_baselines3 import PPO, A2C

def main(config, agent):
    with open(config) as fp:
        json_data = json.load(fp)

    video_path = os.path.join("./videos", agent, "pong")
    config = GameConfig.deserialize(json_data)
    config.agents_config[args.agent]["save_path"] += "best_model.zip"
    # config.agents_config[args.agent]["save_path"] = "my_models/pong/pong_ppo/best_model.zip"
    print(config.agents_config[args.agent]["save_path"])
    # env = retro.make(config.game_name)
    env = gym.make("PongNoFrameskip-v4")


    agent = AgentLoader.get_agent(args.agent, config.agents_config, env, load=True)
    env.close()
    env = gym.make("PongNoFrameskip-v4")
    env = DummyVecEnv([lambda: env])
    # env = retro.make(config.game_name, record=video_path)
    env = VecVideoRecorder(env, video_path, record_video_trigger=lambda x: x == 0, )

    obs = env.reset()
    done = False
    while not done:
      actions, _ = agent.agent.predict(obs)
      obs, rew, done, info = env.step(actions)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--agent", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.agent)
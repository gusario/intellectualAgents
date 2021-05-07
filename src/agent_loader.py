
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from src.exeptions import UnexpectedAgentException
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from typing import Callable


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class Agent:
    def __init__(self, agent, agent_name, config, env, load=False) -> None:
        if load:
            self.agent = self.load_agent(agent, config, env)
        else:
            self.agent = self.init_agent(agent, agent_name, config, env)
        self.save_path = config["save_path"]
        self.agent_name = agent_name

    def save(self):
        print("saving {} agent to {}".format(self.agent_name, self.save_path))
        self.agent.save(self.save_path)

    def init_agent(self, agent, agent_name, config, env):
        if agent_name == "A2C":
            return agent(
                config["policy"],
                env,
                ent_coef=0.01,
                vf_coef=0.25,
                policy_kwargs=dict(optimizer_class=RMSpropTFLike,
                                   optimizer_kwargs=dict(eps=1e-5)),
                verbose=config["verbose"],
                tensorboard_log=config["tensorboard"]
            )
        elif agent_name == "PPO":
            return agent(
                config["policy"],
                env,
                n_steps=128,
                n_epochs=4,
                batch_size=256,
                learning_rate=linear_schedule(2.5e-4),
                clip_range=linear_schedule(0.1),
                ent_coef=0.01,
                vf_coef=0.5,                
                verbose=config["verbose"],
                tensorboard_log=config["tensorboard"]
            )
        else:
            return agent(
                config["policy"],
                env,
                buffer_size=10000,
                batch_size=32,
                learning_starts=100000,
                target_update_interval=1000,
                exploration_final_eps=0.01,
                optimize_memory_usage=True,
                verbose=config["verbose"],
                tensorboard_log=config["tensorboard"]
            )

    def load_agent(self, agent, config, env):
        return agent.load(config["save_path"], env=env)


class AgentLoader:
    agents = {
        "A2C": A2C,
        "PPO": PPO,
        "DQN": DQN
    }

    @classmethod
    def get_agent(cls, algo: str, agents_config: dict, env, load=False):
        if algo in cls.agents and algo in agents_config:
            agent = Agent(cls.agents[algo], algo,
                          agents_config[algo], env, load=load)

            return agent
        else:
            raise UnexpectedAgentException("Agent not found")

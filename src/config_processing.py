
class GameConfig:
    def __init__(self, game_name: str, algos: list, train_states: list, agents_config: dict, eval_state:str, two_players_mode=None) -> None:
        self.game_name: str = game_name
        self.algos: list = algos
        self.train_states: list = train_states
        self.eval_state: str = eval_state
        self.two_players_mode = two_players_mode
        self.agents_config = agents_config

    @staticmethod
    def deserialize(config: dict) -> "GameConfig":
        if "game" not in config:
            raise KeyError("game is not defined")
        if "algos" not in config or len(config["algos"]) < 1:
            raise KeyError("agent algo is not specified")
        if "train_states" not in config or len(config["train_states"]) < 1:
            raise KeyError("train_states are not specified")
        if "eval_state" not in config:
            raise KeyError("eval_state is not defined")
        if "agents_config" not in config:
            raise KeyError("agents_config is not specified")
        
        if "two_players_mode" not in config:
            return GameConfig(
                config["game"], 
                config["algos"], 
                config["train_states"], 
                config["agents_config"], 
                config["eval_state"]
                )
        else:
            return GameConfig(
                config["game"], 
                config["algos"], 
                config["train_states"], 
                config["agents_config"], 
                config["eval_state"], 
                config["two_players_mode"]
                )

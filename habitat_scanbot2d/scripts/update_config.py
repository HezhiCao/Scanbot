from habitat.config.default import Config
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ddppo.ddp_utils import load_resume_state, save_resume_state

def get_new_config() -> Config:
    config = get_config("configs/scanning_rl.yaml")
    config.defrost()
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = [
        "17DRP5sb8fy",
        "2n8kARJN3HM",
        "1pXnuDYAj8r",
        "759xd9YjKW5",
        "5q7pvUzZiYa",
        "7y3sRwLe3Va",
        "JeFG25nYj2p",
        "i5noydFURQK",
    ]
    config.freeze()
    return config

if __name__ == "__main__":
    config = get_new_config()
    state = load_resume_state(config)
    state['config'] = config # type: ignore
    save_resume_state(state, config)

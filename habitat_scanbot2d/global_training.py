from habitat_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
import habitat_scanbot2d


def train_global_policy():
    config = get_config("configs/scanning_rl.yaml")
    config.defrost()
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = [
        "1LXtFkjw3qL",
        "dhjEzFoUFzH",
        "vyrNrziPKCB",
        "VFuaQ6m2Qom",
        "GdvgFV5R1Z5",
        "aayBHfsNo7d",
        "sKLMLpTHeUy",
        "7y3sRwLe3Va",
        "JmbYfDe2QKZ",
        "b8cTxDM8gDG",
        "jh4fc5c5qoQ",
        "SN83YJsR3w2",
        "ULsKaCPVFJR",
        "VVfe2KiqLaN",
        "JeFG25nYj2p",
        "EDJbREhghzL",
        "5q7pvUzZiYa",
        "gZ6f7yhEvPG",
        "17DRP5sb8fy",
        "r1Q1Z4BcV1o",
        "HxpKQynjfin",
        "YmJkqBEsHnH",
        "rPc6DW4iMge",
        "e9zR4mvMWw7",
        "pRbA3pwrgk9",
        "i5noydFURQK",
        "s8pcmisQ38h",
        "Pm6F8kyY3z2",
        "gTV8FGcVJC9",
        "Vvot9Ly1tCj",
        "Uxmj2M2itWa",
        "82sE5b5pLXE",
        "E9uDoFAP3SH",
        "759xd9YjKW5",
        "p5wJjkQkbXX",
        "qoiz87JEwZ2",
        "mJXqzFtmKg4",
        "S9hNv5qa7GM",
        "JF19kD82Mey",
        "D7G3Y4RVNrH",
        "8WUmhLawc2A",
        "1pXnuDYAj8r",
        "sT4fr6TAbpF",
        "ur6pFq6Qu1A",
        "5LpN3gDmAk7",
        "ZMojNkEp431",
        "D7N2EKCX4Sj",
        "V2XKFyX4ASd",
        "cV4RVeZvu5T",
        "VzqfbhrpDEA",
        "B6ByNegPMKs",
        "kEZ7cmS4wCh",
        "ac26ZMwG7aT",
        "VLzqgDo317F",
        "uNb9QFRL6hY",
        "r47D5H71a5s",
        "2n8kARJN3HM",
        "PuKPg4mmafe",
        "PX4nDJXEHrG",
        "29hnd4uzFmX",
        "XcA2TqTSSAj",

        "Albertvill", #gibson
        "Anaheim",
        "Andover",
        "Arkansaw", 
        "Goffs", 
        "Hainesburg",
        "Hillsdale", 
        "Hominy",
        "Micanopy", 
        "Nuevo", 
        "Oyens",
        "Parole",
        "Rosser",
        "Shelbiana", 
        "Silas",
        "Soldier",
        "Stilwell", 
    ]
    config.freeze()
    trainer_factory = baseline_registry.get_trainer(config.TRAINER_NAME)
    trainer = trainer_factory(config)
    trainer.train()


if __name__ == "__main__":
    train_global_policy()

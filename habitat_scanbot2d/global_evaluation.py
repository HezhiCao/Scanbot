import random
from habitat.config.default import Config
import numpy as np
import torch

from habitat_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
import habitat_scanbot2d


def eval_global_policy():
    config = get_config("configs/scanning_rl.yaml")
    config.defrost()
    config.EVAL.SPLIT = "val_test"
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = [
        # small
        # "pLe4wQe7qrG",
        # "x8F5xyUWy9e",
        # "2t7WUuJeko7",
        # "RPmz2sHmrrY",
        # "WYY7iVyf5p8",
        # "YFuZgdQ5vWj",

        # medium
        # "8194nk5LbLH",
        # "zsNo4HB9uLZ",
        # "YVUC4YcDtcY",
        # "EU6Fwq7SyZv",
        # "oLBMNvg9in8",
        # "yqstnuAEVhm",
        "TbHJrupSAjP",
        # "rqfALeAoiTq",
        # "q9vSo1VnCiC",

        # large
        # "X7HyMhZNoso",
        # "gYvKGZ5eRqb",
        # "2azQ1b91cZZ",
        # "ARNzJeq3xxb",
        # "UwV83HsGsw3",
        # "Vt2qJdWjCF2",
        # "Z6MFQCViBuw",
        # "jtcxE69GiFV",

        # extra large
        # "QUCTc6BB5sX",
        # "5ZKStnWn8Zo",
        # "fzynW3qQPVF",
        # "gxdoqLR6rwA",
        # "pa4otMbVnkk",
        # "wc2JMjhGNzB",
    ]
    config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = 100
    config.EVAL_CKPT_PATH_DIR = "data/pretrained_weights/ckpt.43.pth"
    config.NUM_ENVIRONMENTS = min(10, len(config.TASK_CONFIG.DATASET.CONTENT_SCENES))
    config.VERBOSE = False
    config.VIDEO_OPTION = []
    try:
        config.TASK_CONFIG.TASK.MEASUREMENTS.remove("QUALITY_INCREASE_RATIO")
    except ValueError:
        pass
    config.TASK_CONFIG.TASK.OBJECT_DISCOVERY = Config()
    config.TASK_CONFIG.TASK.OBJECT_DISCOVERY.TYPE = "ObjectDiscovery"
    config.TASK_CONFIG.TASK.OBJECT_COVERAGE = Config()
    config.TASK_CONFIG.TASK.OBJECT_COVERAGE.TYPE = "ObjectCoverage"
    # config.TASK_CONFIG.TASK.MEASUREMENTS.append("OBJECT_DISCOVERY")
    config.TASK_CONFIG.TASK.MEASUREMENTS.append("OBJECT_COVERAGE")
    config.freeze()
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    trainer_factory = baseline_registry.get_trainer(config.TRAINER_NAME)
    trainer = trainer_factory(config)
    trainer.eval()

if __name__ == "__main__":
    eval_global_policy()

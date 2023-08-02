from typing import Type
import torch.nn as nn

from habitat_baselines.common.baseline_registry import baseline_registry

def get_auxiliary_task(name: str) -> Type[nn.Module]:
    r"""Return auxiliary task class based on name

    :param name: name of the auxiliary task

    :return: auxiliary task class
    """
    return baseline_registry._get_impl("auxiliary_task", name)

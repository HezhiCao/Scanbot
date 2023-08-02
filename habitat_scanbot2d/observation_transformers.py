import copy
from typing import Dict

from gym import spaces
import numpy as np
import torch
from torch.nn import functional as F

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_scanbot2d.sensors import MapPoseSensor, SemanticTopDownSensor


@baseline_registry.register_obs_transformer()
class NormalizeMapPose(ObservationTransformer):
    r"""Current MapPoseSensor output agent position in (row, col) unit.
    This transformer convert it to (-1.0, 1.0)
    """

    def transform_observation_space(
        self, observation_space: spaces.Dict
    ) -> spaces.Dict:
        observation_space = copy.deepcopy(observation_space)
        # note that, here map_size is max_index + 1
        self.map_size = observation_space.spaces[MapPoseSensor.uuid].high[0] + 1
        observation_space.spaces[MapPoseSensor.uuid] = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        return observation_space

    @torch.no_grad()
    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert hasattr(self, "map_size"), "Unknown map size for MapPoseSensor"
        observations[MapPoseSensor.uuid][:, :2] = (
            observations[MapPoseSensor.uuid][:, :2] / (self.map_size / 2.0) - 1.0
        )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        return cls()


@baseline_registry.register_obs_transformer()
class NormalizeSemanticMap(ObservationTransformer):
    r"""Downsampling global_semantic_map into the same shape with local_semantic_map
    and normalize and obstacle channel of SemanticMap
    """

    def __init__(self, obstacle_threshold):
        super().__init__()
        self.obstacle_threshold = obstacle_threshold

    def transform_observation_space(
        self, observation_space: spaces.Dict
    ) -> spaces.Dict:
        observation_space = copy.deepcopy(observation_space)
        # When using local representation, the final observation will has the same
        # size as local_semantic_map
        if "local_semantic_map" in observation_space.spaces:
            space_shape = list(observation_space.spaces["local_semantic_map"].shape)
            space_shape[2] *= 2
        else:
            space_shape = observation_space.spaces["global_semantic_map"].shape
        observation_space.spaces["global_semantic_map"] = spaces.Box(
            low=0.0,
            high=1.0,
            shape=space_shape,
            dtype=observation_space.spaces["global_semantic_map"].dtype,
        )
        return observation_space

    @torch.no_grad()
    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        original_global_semantic_map = observations["global_semantic_map"]
        self._normalize_obstacle_channel(original_global_semantic_map)

        if "local_semantic_map" in observations:
            local_semantic_map = observations["local_semantic_map"]
            self._normalize_obstacle_channel(local_semantic_map)
            half_num_channels = local_semantic_map.shape[3]
            downsampled_global_semantic_map = torch.empty(
                *local_semantic_map.shape[:3],
                half_num_channels * 2,
                dtype=local_semantic_map.dtype,
                device=local_semantic_map.device
            )
            downsampled_global_semantic_map[..., :half_num_channels] = F.max_pool2d(
                original_global_semantic_map.permute(0, 3, 1, 2),
                kernel_size=int(
                    original_global_semantic_map.shape[1] / local_semantic_map.shape[1]
                ),
            ).permute(0, 2, 3, 1)
            downsampled_global_semantic_map[..., half_num_channels:] = local_semantic_map
            observations["global_semantic_map"] = downsampled_global_semantic_map

        return observations

    def _normalize_obstacle_channel(self, semantic_map: torch.Tensor) -> None:
        r"""Normalize obstacle channel of a semantic map in place
        Before normalization: every cell in obstacle channel represents the number of 3D points
            that fill in the cell
        After normalization: every cell is normalized to a value between [0, 1]

        :param semantic_map: a semantic map whose obstacle channel will be normalized
        """
        semantic_map[..., SemanticTopDownSensor.obstacle_channel] = (
            semantic_map[..., SemanticTopDownSensor.obstacle_channel]
            / self.obstacle_threshold
        ) ** 2
        torch.clamp(
            semantic_map,
            min=0.0,
            max=1.0,
            out=semantic_map,
        )

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            config.RL.POLICY.OBS_TRANSFORMS.NORMALIZE_SEMANTIC_MAP.OBSTACLE_THRESHOLD
        )

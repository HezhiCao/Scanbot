from typing import Optional, Union, cast
import numpy as np
import torch

from habitat.config import Config
from habitat.core.simulator import Observations
from habitat.utils import profiling_wrapper
from habitat_scanbot2d.sensors import SemanticTopDownSensor
from habitat_scanbot2d.cuda_sensors import SemanticTopDownCudaSensor


class NavigationQualityMapBuilder:
    def __init__(self, config: Config):
        self.use_cuda_tensor = False
        self.global_map_size_in_meters = config.MAP_SIZE_IN_METERS
        self.global_map_cell_size = config.MAP_CELL_SIZE
        self.global_map_size_in_cells = self._get_map_size_in_cells()
        self.num_quality_channels = config.NUM_QUALITY_CHANNELS

    def update(self, observations: Observations):
        if SemanticTopDownCudaSensor.uuid in observations:
            quality_topdown = observations[SemanticTopDownCudaSensor.uuid][
                ...,
                SemanticTopDownCudaSensor.quality_channel : SemanticTopDownCudaSensor.quality_channel
                + self.num_quality_channels,
            ]
            self.use_cuda_tensor = True
        else:
            quality_topdown = observations[SemanticTopDownSensor.uuid][
                ...,
                SemanticTopDownCudaSensor.quality_channel : SemanticTopDownCudaSensor.quality_channel
                + self.num_quality_channels,
            ]
            self.use_cuda_tensor = False

        assert quality_topdown.shape == (
            self.global_map_size_in_cells,
            self.global_map_size_in_cells,
            self.num_quality_channels,
        ), "Unmatched single frame and quality map sizes"
        if self.use_cuda_tensor:
            if hasattr(self, "quality_map"):
                torch.maximum(self.quality_map, quality_topdown, out=self.quality_map)
            else:
                self.quality_map = quality_topdown.float().clone()
        else:
            if hasattr(self, "quality_map"):
                np.maximum(self.quality_map, quality_topdown, out=self.quality_map)
            else:
                self.quality_map = quality_topdown.astype(np.float32).copy()

    def reset(self, observations):
        if hasattr(self, "quality_map"):
            if self.use_cuda_tensor:
                self.quality_map.zero_()
            else:
                self.quality_map.fill(0.0)

        self.initial_quality_map = observations["global_semantic_map"][
            ...,
            SemanticTopDownSensor.quality_channel : SemanticTopDownSensor.quality_channel
            + self.num_quality_channels,
        ].clone()

    def compute_quality_increase_ratio(self, observations) -> float:
        if self.use_cuda_tensor:
            quality_increase = torch.sum(
                observations["global_semantic_map"][
                ...,
                SemanticTopDownSensor.quality_channel : SemanticTopDownSensor.quality_channel
                + self.num_quality_channels,
            ]
            - self.initial_quality_map > 0).item()
            observed_object_points = torch.sum(self.quality_map > 0.0).item()
            return quality_increase / observed_object_points if observed_object_points != 0 else 0.0
        else:
            return np.sum(self.quality_map > 0.0)

    def _get_map_size_in_cells(self) -> int:
        r"""Compute how many cells are there in the 2d topdown map

        :return: map size in cell
        """
        return int(np.ceil(self.global_map_size_in_meters / self.global_map_cell_size))

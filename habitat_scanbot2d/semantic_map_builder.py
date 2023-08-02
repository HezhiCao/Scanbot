#  ____                  _           _
# / ___|  ___ __ _ _ __ | |__   ___ | |_
# \___ \ / __/ _` | '_ \| '_ \ / _ \| __|
#  ___) | (_| (_| | | | | |_) | (_) | |_
# |____/ \___\__,_|_| |_|_.__/ \___/ \__|

"""
    habitat_scanbot2d.semantic_map_builder
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    A class that continuously integrates single frame from SemanticTopDownSensor
    into a global consistant 2D representation

    :copyright: (c) 2021 by XiXia.
    :license: MIT License, see LICENSE for more details.
"""

from typing import Optional, Union, cast
import numpy as np
import torch

from habitat.config import Config
from habitat.core.simulator import Observations
from habitat.utils import profiling_wrapper
from habitat_scanbot2d.sensors import MapPoseSensor, SemanticTopDownSensor
from habitat_scanbot2d.cuda_sensors import SemanticTopDownCudaSensor


class SemanticMapBuilder:
    r"""Integrate the local topdown map from SemanticTopDownSensor into a global topdown map

    Args:
        config: config for the SemanticMapBuilder
    """

    def __init__(self, config: Config):
        self.config = config
        self.use_cuda_tensor = False
        self.reset(config)

    def _get_map_size_in_cells(self) -> int:
        r"""Compute how many cells are there in the 2d topdown map

        :return: map size in cell
        """
        return int(np.ceil(self.global_map_size_in_meters / self.global_map_cell_size))


    @profiling_wrapper.RangeContext("SemanticMapBuilder.update")
    def update(self, observations: Observations):
        r"""Update current observation into a global map

        :param semantic_topdown: observation from SemanticTopDown sensor
        """
        if SemanticTopDownCudaSensor.uuid in observations:
            semantic_topdown = observations[SemanticTopDownCudaSensor.uuid]
            self.use_cuda_tensor = True
        else:
            semantic_topdown = observations[SemanticTopDownSensor.uuid]
            self.use_cuda_tensor = False

        assert semantic_topdown.shape == (
            self.global_map_size_in_cells,
            self.global_map_size_in_cells,
            self.num_total_channels,
        ), "Unmatched single frame and integrated map sizes"
        if self.use_cuda_tensor:
            if hasattr(self, "semantic_map"):
                torch.maximum(self.semantic_map, semantic_topdown, out=self.semantic_map)  # type: ignore
            else:
                self.semantic_map = semantic_topdown.float().clone()
        else:
            if hasattr(self, "semantic_map"):
                np.maximum(self.semantic_map, semantic_topdown, out=self.semantic_map)  # type: ignore
            else:
                self.semantic_map = semantic_topdown.astype(np.float32).copy()

        self._update_trajectory_channel(observations[MapPoseSensor.uuid][:2].astype(np.int32))

        observations["global_semantic_map"] = self.semantic_map
        if self.config.USE_LOCAL_REPRESENTATION:
            observations["local_semantic_map"] = self._crop_local_semantic_map(
                observations
            )

    def reset(self, config: Optional[Config] = None):
        r"""Reset current semantic map with optional updated config

        :param config: config for the SemanticMapBuilder
        """
        if config is not None:
            if config.USE_LOCAL_REPRESENTATION:
                self.local_map_cell_size = config.LOCAL_MAP_CELL_SIZE
                self.global_map_size_in_meters = config.MAP_SIZE_IN_METERS
                self.global_map_cell_size = config.MAP_CELL_SIZE
                # Notice that, this is a little bit confusing
                #   - local map is the map that with agent in it's center and cropped from raw global map
                #   - global map is the integration of the raw outputs from SemanticTopDownSensor
                # However, we will finally max pool the global map into the same size with local map,
                # which will use torch cuda operation in obs_transformer.
                # Hopefully, this can be faster than numpy equivalent without creating cuda context
                # in subprocesses.
                # But now, we are dealing with larger and unpooled global map
                self.local_map_size_in_cells = self._get_map_size_in_cells()
                self.global_map_size_in_cells = int(
                    self.local_map_size_in_cells
                    * self.global_map_cell_size
                    / self.local_map_cell_size
                )
            else:
                self.global_map_size_in_meters = config.MAP_SIZE_IN_METERS
                self.global_map_cell_size = config.MAP_CELL_SIZE
                self.global_map_size_in_cells = self._get_map_size_in_cells()
            self.num_total_channels = config.NUM_TOTAL_CHANNELS

        if hasattr(self, "semantic_map"):
            if self.use_cuda_tensor:
                self.semantic_map.zero_()
            else:
                self.semantic_map.fill(0.0)

    def _update_trajectory_channel(self, current_position: np.ndarray):
        r"""Update agent past map positions (trajectory) using a linear decay.
        The most recent position will be set to 1.0 and gradually decreased to 0.0,
        with the total num of the nonzeros to be TRAJECTORY_LENGTH

        :param current_position: 2d position in map coordinate
        """
        self.semantic_map[..., -1] -= 1 / self.config.TRAJECTORY_LENGTH
        if self.use_cuda_tensor:
            torch.clip(self.semantic_map[..., -1], min=0.0, max=1.0, out=self.semantic_map[..., -1])
        else:
            np.clip(self.semantic_map[..., -1], a_min=0.0, a_max=1.0, out=self.semantic_map[..., -1])
        self.semantic_map[current_position[0], current_position[1], -1] = 1.0

    def _crop_local_semantic_map(
        self, observations: Observations
    ) -> Union[np.ndarray, torch.Tensor]:
        r"""Crop a local map from the larger global map with agent at the center of it

        :param observations: all observations from current step
        :return: cropped local map
        """
        map_position = observations[MapPoseSensor.uuid][:2].astype(np.int32)
        if self.use_cuda_tensor:
            local_semantic_map = torch.zeros(
                self.local_map_size_in_cells,
                self.local_map_size_in_cells,
                self.num_total_channels,
                dtype=torch.float32,
                device=self.semantic_map.device, # type: ignore
            )
        else:
            local_semantic_map = np.zeros(
                (
                    self.local_map_size_in_cells,
                    self.local_map_size_in_cells,
                    self.num_total_channels,
                ),
                dtype=np.float32,
            )
        half_length = self.local_map_size_in_cells // 2
        global_min_indexes = np.maximum(map_position - half_length, 0)
        global_max_indexes = np.minimum(
            map_position + (self.local_map_size_in_cells - half_length),
            self.global_map_size_in_cells,
        )
        local_min_indexes = half_length - (map_position - global_min_indexes)
        local_max_indexes = half_length - (map_position - global_max_indexes)
        local_semantic_map[
            local_min_indexes[0] : local_max_indexes[0],
            local_min_indexes[1] : local_max_indexes[1],
        ] = self.semantic_map[  # type: ignore
            global_min_indexes[0] : global_max_indexes[0],
            global_min_indexes[1] : global_max_indexes[1],
        ]
        return local_semantic_map

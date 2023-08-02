from typing import Union
import numpy as np
import quaternion
import habitat_sim
from habitat.core.simulator import AgentState

MP3D_CATEGORY_NAMES = {
    "void": 0,
    "wall": 1,
    "floor": 2,
    "chair": 3,
    "door": 4,
    "table": 5,
    "picture": 6,
    "cabinet": 7,
    "cushion": 8,
    "window": 9,
    "sofa": 10,
    "bed": 11,
    "curtain": 12,
    "chest_of_drawers": 13,
    "plant": 14,
    "sink": 15,
    "stairs": 16,
    "ceiling": 17,
    "toilet": 18,
    "stool": 19,
    "towel": 20,
    "mirror": 21,
    "tv_monitor": 22,
    "shower": 23,
    "column": 24,
    "bathtub": 25,
    "counter": 26,
    "fireplace": 27,
    "lighting": 28,
    "beam": 29,
    "railing": 30,
    "shelving": 31,
    "gym_equipment": 33,
    "seating": 34,
    "furniture": 36,
    "appliances": 37,
    "clothes": 38,
    "objects": 39,
    "misc": 40,
}

def extract_object_category_name(scene):
    object_category_name = {}
    for obj in scene.objects:
        if obj is not None:
            object_category_name[obj.category.index()] = obj.category.name()
    return object_category_name


def compute_map_size_in_cells(map_size_in_meters: float, map_cell_size: float) -> int:
    r"""Compute how many cells are there in the 2d topdown map

    :param map_size_in_meters: max size of the scene that the map can represent (e.g. 40m x 40m)
    :param map_cell_size: the real size that one cell represents
    :return: map size in cell
    """
    return int(np.ceil(map_size_in_meters / map_cell_size))


def construct_transformation_matrix(
    agent_state: Union[AgentState, habitat_sim.AgentState]
) -> np.ndarray:
    r"""Construct a 4x4 matrix from agent state provided by HabitatSim

    :param agent_state: Agent state contains position and rotation (quaternion)
    :return: 4x4 transformation matrix
    """
    transformation = np.eye(4, dtype=np.float32)
    transformation[:3, :3] = quaternion.as_rotation_matrix(agent_state.sensor_states["depth"].rotation)
    transformation[:3, 3] = agent_state.sensor_states["depth"].position
    return transformation

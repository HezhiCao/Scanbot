import numpy as np
import matplotlib.pyplot as plt
from habitat_scanbot2d.sensors import SemanticTopDownSensor
from habitat_scanbot2d.navigation_action import NavigationAction
from habitat_scanbot2d.astar_path_finder import AStarPathFinder
from habitat_scanbot2d.dstar_lite_path_finder import DStarLitePathFinder

def test_normalized_obstacle_map():
    semantic_map = np.load("semantic_map.npy")
    obstacle_map = semantic_map[..., SemanticTopDownSensor.obstacle_channel]
    plt.imshow(obstacle_map, vmin=0.0, vmax=1.0)
    plt.show()
    obstacle_map = NavigationAction.normalize_obstacle_map(obstacle_map, 10)
    plt.imshow(obstacle_map, vmin=0.0, vmax=1.0)
    plt.show()

def test_obstacle_map_without_inflation():
    obstacle_map = np.load("semantic_map.npy")[..., SemanticTopDownSensor.obstacle_channel]

    path_finder = AStarPathFinder()

    start_point = np.array([525, 169], dtype=np.int32)
    end_point = np.array([380, 350], dtype=np.int32)
    obstacle_map[start_point[0], start_point[1]] = 0.0
    obstacle_map[
        max(end_point[0] - 1, 0) : min(
            end_point[0] + 2, obstacle_map.shape[0] - 1
        ),
        max(end_point[1] - 1, 0) : min(
            end_point[1] + 2, obstacle_map.shape[1] - 1
        ),
    ] = 0.0
    result = path_finder.find(
        obstacle_map=obstacle_map,
        start_point=start_point,
        end_point=end_point,
        obstacle_cost=1000.0,
        iteration_threshold=-1,
    )

    plt.imshow(obstacle_map, vmin=0.0, vmax=1.0)
    path = np.array(result, dtype=np.int32)
    np.clip(path, a_min=0, a_max=719, out=path)
    plt.plot(
        [path_point[1] for path_point in path],
        [path_point[0] for path_point in path],
        "r",
    )
    plt.show()
    print(f"result: {result}")

def test_dstar_lite_path_finder():
    obstacle_map = np.load("semantic_map.npy")[..., SemanticTopDownSensor.obstacle_channel]

    path_finder = DStarLitePathFinder()

    start_point = np.array([525, 169], dtype=np.int32)
    end_point = np.array([380, 350], dtype=np.int32)
    # start_point = np.array([440, 200], dtype=np.int32)
    # end_point = np.array([436, 200], dtype=np.int32)
    obstacle_map[start_point[0], start_point[1]] = 0.0
    obstacle_map[
        max(end_point[0] - 1, 0) : min(
            end_point[0] + 2, obstacle_map.shape[0] - 1
        ),
        max(end_point[1] - 1, 0) : min(
            end_point[1] + 2, obstacle_map.shape[1] - 1
        ),
    ] = 0.0
    result = path_finder.find(
        obstacle_map,
        start_point,
        end_point,
        1000.0,
        True,
    )

    plt.imshow(obstacle_map, vmin=0.0, vmax=1.0)
    path = np.array(result, dtype=np.int32)
    np.clip(path, a_min=0, a_max=719, out=path)
    plt.plot(
        [path_point[1] for path_point in path],
        [path_point[0] for path_point in path],
        "r",
    )
    plt.show()
    print(f"result: {result}")


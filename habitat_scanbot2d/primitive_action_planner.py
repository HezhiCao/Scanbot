from typing import List, Dict, Any, Optional
from habitat_baselines.slambased.reprojection import (
    angle_to_pi_2_minus_pi_2,
)
import numpy as np


class PrimitiveActionPlanner:
    r"""A heuristic planner that convert a navigation path into a series of primitive
    actions (MOVE_FORWARD, TURN_RIGHT, TURN_LEFT)

    Args:
        config: config for PrimitiveActionPlanner
        path: generated 2d path by PathPlanner
    """

    def __init__(self, config, path: List[np.ndarray]):
        self.map_size_in_meters = config.MAP_SIZE_IN_METERS
        self.map_cell_size = config.MAP_CELL_SIZE
        self.waypoint: Optional[np.ndarray] = None
        self.current_pose: np.ndarray
        self.position_threshold = config.POSITION_THRESHOLD / self.map_cell_size
        self.next_waypoint_threshold = config.NEXT_WAYPOINT_THRESHOLD / self.map_cell_size
        self.angle_threshold = config.ANGLE_THRESHOLD
        self.path = path

    def step(self, current_pose: np.ndarray) -> Dict[str, Any]:
        r"""Plan next action to approach a waypoint from the planned path

        :param current_pose: agent's current pose in map coordinate
        :return: planned action
        """
        self.current_pose = current_pose
        if self._is_goal_reached():
            return {"action": "STOP"}
        if self.waypoint is None or self._is_waypoint_reached():
            self._find_valid_waypoint()
        action = self._decide_next_action()
        return {"action": action}

    def _decide_next_action(self) -> str:
        r"""Decide next action through current pose and next waypoint

        :param current_pose: agent's current pose in map coordinate
        :return: string representation of action
        """
        rotate_angle = angle_to_pi_2_minus_pi_2(self._compute_angle())
        if abs(rotate_angle) < self.angle_threshold:
            return "MOVE_FORWARD"
         # note that y_axis decreases down to up in map coordinate
        if 0 <= rotate_angle <= np.pi:
            return "TURN_RIGHT"
        else:
            return "TURN_LEFT"

    def _find_valid_waypoint(self):
        r"""Find next waypoint that is not too close to the agent

        :param current_pose: agent's current position in map coordinate
        """
        self.waypoint = self.path[0]
        while self._is_waypoint_reached():
            if len(self.path) > 1:
                self.path = self.path[1:]
                self.waypoint = self.path[0]
            else:
                self.waypoint = self.goal_position
                break

    def _is_goal_reached(self) -> bool:
        r"""Check if the agent has reached the goal

        :return: return True if reached
        """
        dist_diff = self.get_distance(self.goal_position, self.current_pose[:2])
        return dist_diff <= self.position_threshold

    def _is_waypoint_reached(self) -> bool:
        r"""Check if the agent has reached the next waypoint

        :return: return True if reached
        """
        dist_diff = self.get_distance(self.waypoint, self.current_pose[:2])
        return dist_diff <= self.position_threshold

    def _compute_angle(self) -> float:
        r"""Compute the angle to turn before the agent can face the waypoint

        :return: angle of rotation
        """
        pos_diff = self.waypoint - self.current_pose[:2]
        target_angle = np.arctan2(pos_diff[0], pos_diff[1])
        current_angle = np.arctan2(self.current_pose[2], self.current_pose[3])
        return target_angle - current_angle

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path
        self.goal_position = path[-1]
        self.waypoint = None

    @staticmethod
    def get_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

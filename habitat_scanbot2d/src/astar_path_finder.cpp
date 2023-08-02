#include "astar_path_finder.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace scanbot {

std::vector<Eigen::Vector2i> AStarPathFinder::find(const Eigen::MatrixXf &obstacle_map,
                                                   const Eigen::Vector2i &start_point,
                                                   const Eigen::Vector2i &end_point,
                                                   float obstacle_cost, long iteration_threshold) {
    obstacle_cost_ = obstacle_cost;
    iteration_threshold_ = iteration_threshold;
    reset(obstacle_map);
    expand(obstacle_map, start_point, end_point);
    return extractPath(start_point);
}

void AStarPathFinder::expand(const Eigen::MatrixXf &obstacle_map,
                             const Eigen::Vector2i &start_point, const Eigen::Vector2i &end_point) {
    if (!checkWithinBound(start_point) || !checkWithinBound(end_point)) {
        throw std::runtime_error("Invalid start/end point");
    }
    node_queue_.emplace(start_point, 0.0);
    cost_grid_(start_point[0], start_point[1]) = 0;

    Eigen::Vector2i expanding_point;
    nearest_point_ = start_point;
    for (long iteration_counter = 0;
         iteration_counter != iteration_threshold_ && !node_queue_.empty(); ++iteration_counter) {
        ExpandedNode expanding_node = node_queue_.top();
        node_queue_.pop();

        expanding_point = expanding_node.first;

        if (expanding_point == end_point) {
            nearest_point_ = end_point;
            break;
        }

        for (int dx = -1; dx < 2; ++dx) {
            for (int dy = -1; dy < 2; ++dy) {
                if (0 == dx && 0 == dy) continue;
                bool is_vertical = (abs(dx) + abs(dy) != 2);
                Eigen::Vector2i next_expanding_point = expanding_point + Eigen::Vector2i(dy, dx);
                if (!checkWithinBound(next_expanding_point)) continue;

                addNode(obstacle_map, expanding_point, next_expanding_point, end_point,
                        is_vertical);
            }
        }
    }
}

void AStarPathFinder::addNode(const Eigen::MatrixXf &obstacle_map,
                              const Eigen::Vector2i &previous_point,
                              const Eigen::Vector2i &next_point, const Eigen::Vector2i &end_point,
                              bool is_vertical) {
    // With negative obstacle_cost_, the agent is not allowed to traverse obstacles
    // Otherwise we still can't traverse true obstacles
    // WARNING: The generated path may end up with only start point
    if ((obstacle_cost_ < 0 && obstacle_map(next_point[0], next_point[1]) > 0.01) ||
        obstacle_map(next_point[0], next_point[1]) > 0.95) {
        return;
    }

    double step_cost = (is_vertical ? step_cost_ : diagonal_step_cost_);
    double new_cost = cost_grid_(previous_point[0], previous_point[1]) + step_cost +
                      obstacle_map(next_point[0], next_point[1]) * obstacle_cost_;
    if (cost_grid_(next_point[0], next_point[1]) <= new_cost + 1e-3) {
        return;
    }
    double manhattan_distance = (end_point - next_point).lpNorm<1>();
    cost_grid_(next_point[0], next_point[1]) = new_cost;
    node_queue_.emplace(next_point,
                        cost_grid_(next_point[0], next_point[1]) + manhattan_distance * step_cost_);
    expanded_trace_[next_point] = previous_point;
    if (manhattan_distance < min_distance_) {
        min_distance_ = manhattan_distance;
        nearest_point_ = next_point;
    }
}

std::vector<Eigen::Vector2i> AStarPathFinder::extractPath(const Eigen::Vector2i &start_point) {
    std::vector<Eigen::Vector2i> result;
    Eigen::Vector2i current_point = nearest_point_;
    while (current_point != start_point) {
        result.push_back(current_point);
        current_point = expanded_trace_.find(current_point)->second;
    }
    result.push_back(start_point);
    std::reverse(result.begin(), result.end());
    return result;
}

void AStarPathFinder::reset(const Eigen::MatrixXf &obstacle_map) {
    if (obstacle_map.cols() != cost_grid_.cols() || obstacle_map.rows() != cost_grid_.rows()) {
        cost_grid_ = Eigen::MatrixXf::Constant(obstacle_map.rows(), obstacle_map.cols(),
                                               std::numeric_limits<float>::max());
    } else {
        cost_grid_.fill(std::numeric_limits<float>::max());
    }
    expanded_trace_ = std::unordered_map<Eigen::Vector2i, Eigen::Vector2i>{};
    node_queue_ = std::priority_queue<ExpandedNode, std::vector<ExpandedNode>,
                                      decltype(nodePriorityComparator) *>{nodePriorityComparator};
    min_distance_ = std::numeric_limits<double>::max();
}

bool AStarPathFinder::checkWithinBound(const Eigen::Vector2i &query_point) {
    return !(query_point[0] < 0 || query_point[1] < 0 || query_point[0] >= cost_grid_.rows() ||
             query_point[1] >= cost_grid_.cols());
}

}  // end namespace scanbot

#ifndef SCANBOT_ASTAR_PATH_FINDER_H_
#define SCANBOT_ASTAR_PATH_FINDER_H_

#include <Eigen/Dense>
#include <queue>
#include <unordered_map>
#include <vector>

#include "eigen_hash.h"

namespace scanbot {
class AStarPathFinder {
public:
    std::vector<Eigen::Vector2i> find(const Eigen::MatrixXf &obstacle_map,
                                      const Eigen::Vector2i &start_point,
                                      const Eigen::Vector2i &end_point,
                                      float obstacle_cost = 1000.0f,
                                      long iteration_threshold = 500);

private:
    void expand(const Eigen::MatrixXf &obstacle_map, const Eigen::Vector2i &start_point,
                const Eigen::Vector2i &end_point);

    std::vector<Eigen::Vector2i> extractPath(const Eigen::Vector2i &start_point);

    void addNode(const Eigen::MatrixXf &obstacle_map, const Eigen::Vector2i &prev_point,
                 const Eigen::Vector2i &next_point, const Eigen::Vector2i &end_point,
                 bool is_vertical);

    void reset(const Eigen::MatrixXf &obstacle_map);

    bool checkWithinBound(const Eigen::Vector2i &query_point);

    using ExpandedNode = std::pair<Eigen::Vector2i, float>;
    float obstacle_cost_ = 5.0f;
    const float step_cost_ = 1.0f;
    const float diagonal_step_cost_ = std::sqrt(2.0f);
    long iteration_threshold_ = 500;
    Eigen::MatrixXf cost_grid_;
    std::unordered_map<Eigen::Vector2i, Eigen::Vector2i> expanded_trace_;
    Eigen::Vector2i nearest_point_;
    double min_distance_;

    static bool nodePriorityComparator(const ExpandedNode &lhs, const ExpandedNode &rhs) {
        return lhs.second > rhs.second;
    }

    std::priority_queue<ExpandedNode, std::vector<ExpandedNode>, decltype(nodePriorityComparator) *>
        node_queue_{nodePriorityComparator};
};

}  // end namespace scanbot
#endif  // SCANBOT_ASTAR_PATH_FINDER_H_

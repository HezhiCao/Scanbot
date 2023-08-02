use float_eq::{float_eq, float_ne};
use ndarray::Array2;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections;

mod utils;

fn compute_heuristic(node1: &utils::NodePosition, node2: &utils::NodePosition) -> f64 {
    (node1.0 as i64 - node2.0 as i64)
        .abs()
        .max((node1.1 as i64 - node2.1 as i64).abs()) as f64
}

#[pyclass]
#[derive(Debug, Default)]
pub struct DStarLitePathFinder {
    start: utils::NodePosition,
    goal: utils::NodePosition,
    previous_obstacle_map: Array2<f32>,
    obstacle_map: Array2<f32>,
    obstacle_cost: f64,
    key_modifier: f64,
    node_queue: collections::BinaryHeap<utils::HeapEntry>,
    node_states: collections::HashMap<utils::NodePosition, utils::NodeState>,
}

#[pymethods]
impl DStarLitePathFinder {
    #[new]
    fn new() -> Self {
        Self {
            obstacle_cost: 1.0,
            ..Default::default()
        }
    }

    fn find(
        &mut self,
        obstacle_map: PyReadonlyArray2<f32>,
        start: PyReadonlyArray1<i32>,
        goal: PyReadonlyArray1<i32>,
        obstacle_cost: f64,
        reset: bool,
    ) -> Vec<(usize, usize)> {
        self.obstacle_map = obstacle_map.to_owned_array();
        self.obstacle_cost = obstacle_cost;
        let new_start = utils::NodePosition::from(start.as_array());
        let new_goal = utils::NodePosition::from(goal.as_array());
        if reset || new_goal != self.goal {
            self.start = new_start;
            self.goal = new_goal;
            self.reset();
        } else {
            self.key_modifier += compute_heuristic(&new_start, &self.start);
            self.start = new_start;
        }
        self.solve_impl()
    }
}

impl DStarLitePathFinder {
    fn solve_impl(&mut self) -> Vec<(usize, usize)> {
        self.update_edge_costs();
        self.compute_path();
        self.construct_path()
    }

    fn reset(&mut self) {
        self.key_modifier = 0.0;
        self.node_queue.clear();
        self.node_states.clear();

        self.node_states.insert(
            self.start,
            utils::NodeState {
                g: f64::INFINITY,
                rhs: f64::INFINITY,
            },
        );
        self.node_states.insert(
            self.goal,
            utils::NodeState {
                g: f64::INFINITY,
                rhs: 0.0,
            },
        );

        self.node_queue.push(utils::HeapEntry::new(
            self.goal,
            self.compute_key(&self.goal),
        ));
    }

    fn update_edge_costs(&mut self) {
        if self.previous_obstacle_map.is_empty() {
            self.previous_obstacle_map.clone_from(&self.obstacle_map);
            return;
        }

        let changed_nodes: Vec<_> = self
            .node_states
            .keys()
            .filter(|&p| {
                float_ne!(
                    self.previous_obstacle_map[[p.0, p.1]],
                    self.obstacle_map[[p.0, p.1]],
                    abs <= 0.01
                )
            })
            .copied()
            .collect();

        for changed_node in changed_nodes {
            if self.node_states.contains_key(&changed_node) {
                let changed_node_state = self.node_states[&changed_node];
                let o_old = self.previous_obstacle_map[[changed_node.0, changed_node.1]] as f64
                    * self.obstacle_cost;
                // Here we assume the cost will only increase
                // with c_old < c_new
                for neighbor in self.get_neighbors(&changed_node) {
                    if let Some(&neighbor_state) = self.node_states.get(&neighbor) {
                        let c_old = o_old.max(
                            self.previous_obstacle_map[[neighbor.0, neighbor.1]] as f64
                                * self.obstacle_cost,
                        ) + 1.0;
                        if neighbor != self.goal
                            && float_eq!(
                                neighbor_state.rhs,
                                c_old + changed_node_state.g,
                                rmin <= f64::EPSILON
                            )
                        {
                            self.update_rhs_by_neighbors(&neighbor);
                            self.update_vertex(&neighbor);
                        }

                        if float_eq!(
                            changed_node_state.rhs,
                            c_old + neighbor_state.g,
                            rmin <= f64::EPSILON
                        ) {
                            self.update_rhs_by_neighbors(&changed_node);
                        }
                    }
                }
                self.update_vertex(&changed_node);
            }
        }

        self.previous_obstacle_map.clone_from(&self.obstacle_map);
    }

    fn construct_path(&self) -> Vec<(usize, usize)> {
        let mut path = Vec::new();
        let mut current_node = self.start;
        path.push((current_node.0, current_node.1));
        while current_node != self.goal {
            let mut min_cost = f64::INFINITY;
            let mut min_distance = f64::INFINITY;
            let mut next_node = current_node;
            for neighbor in self.get_neighbors(&current_node) {
                if let Some(neighbor_state) = self.node_states.get(&neighbor) {
                    if (neighbor_state.g + self.get_cost(&current_node, &neighbor) < min_cost)
                        || (float_eq!(
                            neighbor_state.g + self.get_cost(&current_node, &neighbor),
                            min_cost,
                            rmin <= f64::EPSILON
                        ) && self.compute_distance_to_goal(&neighbor) < min_distance)
                    {
                        min_cost =
                            neighbor_state.g + self.get_cost(&current_node, &neighbor);
                        min_distance = self.compute_distance_to_goal(&neighbor);
                        next_node = neighbor;
                    }
                }
            }
            current_node = next_node;
            path.push((current_node.0, current_node.1));
        }
        path
    }

    fn compute_key(&self, node: &utils::NodePosition) -> (f64, f64) {
        let state = &self.node_states[node];
        let key1 = state.g.min(state.rhs);
        let key0 = key1 + compute_heuristic(node, &self.start) + self.key_modifier;
        (key0, key1)
    }

    fn compute_path(&mut self) {
        while (!self.node_queue.is_empty()
            && self.node_queue.peek().unwrap().keys <= self.compute_key(&self.start))
            || self.node_states[&self.start].rhs > self.node_states[&self.start].g
        {
            let mut current_node = self.node_queue.pop().unwrap();
            if float_eq!(
                self.node_states[&current_node.position].g,
                self.node_states[&current_node.position].rhs,
                rmin <= f64::EPSILON
            ) {
                continue;
            }
            // We only need to recompute key second planning onwards
            if self.key_modifier > 0.0 {
                let new_key = self.compute_key(&current_node.position);
                if current_node.keys < new_key {
                    current_node.keys = new_key;
                    self.node_queue.push(current_node);
                    continue;
                }
            }

            let mut current_state = self.node_states[&current_node.position];
            // Overconsistent
            if current_state.g > current_state.rhs {
                current_state.g = current_state.rhs;
                for neighbor in self.get_neighbors(&current_node.position) {
                    let mut neighbor_state =
                        *(self
                            .node_states
                            .entry(neighbor)
                            .or_insert(utils::NodeState {
                                rhs: f64::INFINITY,
                                g: f64::INFINITY,
                            }));

                    if current_state.g + self.get_cost(&current_node.position, &neighbor)
                        < neighbor_state.rhs
                    {
                        neighbor_state.rhs =
                            current_state.g + self.get_cost(&current_node.position, &neighbor);
                        *self.node_states.get_mut(&neighbor).unwrap() = neighbor_state;
                        self.update_vertex(&neighbor)
                    }
                }
                *self.node_states.get_mut(&current_node.position).unwrap() = current_state;
            // Underconsistent
            } else {
                let g_old = current_state.g;
                current_state.g = f64::INFINITY;
                *self.node_states.get_mut(&current_node.position).unwrap() = current_state;
                for neighbor in self.get_neighbors(&current_node.position) {
                    let neighbor_state =
                        *(self
                            .node_states
                            .entry(neighbor)
                            .or_insert(utils::NodeState {
                                rhs: f64::INFINITY,
                                g: f64::INFINITY,
                            }));

                    if float_eq!(
                        neighbor_state.rhs,
                        self.get_cost(&current_node.position, &neighbor) + g_old,
                        rmin <= f64::EPSILON
                    ) && neighbor != self.goal
                    {
                        self.update_rhs_by_neighbors(&neighbor);
                    }
                    self.update_vertex(&neighbor);
                }

                self.update_vertex(&current_node.position)
            }
        }
    }

    fn update_rhs_by_neighbors(&mut self, node: &utils::NodePosition) {
        self.node_states.get_mut(node).unwrap().rhs =
            self.get_neighbors(node)
                .into_iter()
                .fold(f64::INFINITY, |rhs, neighbor| {
                    if self.node_states.contains_key(&neighbor) {
                        rhs.min(self.get_cost(node, &neighbor) + self.node_states[&neighbor].g)
                    } else {
                        rhs
                    }
                });
    }

    fn update_vertex(&mut self, node: &utils::NodePosition) {
        if float_ne!(
            self.node_states[node].g,
            self.node_states[node].rhs,
            rmin <= f64::EPSILON
        ) {
            self.node_queue
                .push(utils::HeapEntry::new(*node, self.compute_key(node)));
        }
    }

    fn get_neighbors(&self, node: &utils::NodePosition) -> Vec<utils::NodePosition> {
        let mut neighbors = Vec::with_capacity(8);
        for i in -1..=1 {
            for j in -1..=1 {
                if i == 0 && j == 0 {
                    continue;
                }
                if ((node.0 as i32 + i) as usize) < self.obstacle_map.shape()[0]
                    && ((node.1 as i32 + j) as usize) < self.obstacle_map.shape()[1]
                {
                    neighbors.push(utils::NodePosition(
                        (node.0 as i32 + i) as usize,
                        (node.1 as i32 + j) as usize,
                    ));
                }
            }
        }
        neighbors
    }

    fn get_cost(&self, node1: &utils::NodePosition, node2: &utils::NodePosition) -> f64 {
        self.obstacle_map[[node1.0, node1.1]].max(self.obstacle_map[[node2.0, node2.1]]) as f64
            * self.obstacle_cost
            + 1.0
    }

    fn compute_distance_to_goal(&self, node: &utils::NodePosition) -> f64 {
        ((self.goal.0 - node.0).pow(2) + (self.goal.1 - node.1).pow(2)) as f64
    }
}

#[pymodule]
fn dstar_lite_path_finder(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DStarLitePathFinder>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;

    #[test]
    fn test_compute_heuristic() {
        let node1 = utils::NodePosition(8, 10);
        let node2 = utils::NodePosition(1, 0);
        assert_float_eq!(
            compute_heuristic(&node1, &node2),
            10.0,
            r2nd <= f64::EPSILON
        );

        let node1 = utils::NodePosition(12, 10);
        let node2 = utils::NodePosition(1, 0);
        assert_float_eq!(
            compute_heuristic(&node1, &node2),
            11.0,
            r2nd <= f64::EPSILON
        );

        let node1 = utils::NodePosition(1, 5);
        let node2 = utils::NodePosition(7, 0);
        assert_float_eq!(compute_heuristic(&node1, &node2), 6.0, r2nd <= f64::EPSILON);
    }

    #[test]
    fn test_compute_key() {
        let mut path_finder = DStarLitePathFinder::new();
        let query_node1 = utils::NodePosition(1, 1);
        let query_node2 = utils::NodePosition(2, 2);

        path_finder
            .node_states
            .insert(query_node1, utils::NodeState { rhs: 10.0, g: 8.0 });

        path_finder.node_states.insert(
            query_node2,
            utils::NodeState {
                rhs: 5.1,
                g: f64::INFINITY,
            },
        );

        let (key1, key2) = path_finder.compute_key(&query_node1);
        assert_float_eq!(key1, 9.0, r2nd <= f64::EPSILON);
        assert_float_eq!(key2, 8.0, r2nd <= f64::EPSILON);

        let (key1, key2) = path_finder.compute_key(&query_node2);
        assert_float_eq!(key1, 7.1, r2nd <= f64::EPSILON);
        assert_float_eq!(key2, 5.1, r2nd <= f64::EPSILON);

        path_finder.key_modifier += 3.0;
        let (key1, key2) = path_finder.compute_key(&query_node1);
        assert_float_eq!(key1, 12.0, r2nd <= f64::EPSILON);
        assert_float_eq!(key2, 8.0, r2nd <= f64::EPSILON);

        let (key1, key2) = path_finder.compute_key(&query_node2);
        assert_float_eq!(key1, 10.1, r2nd <= f64::EPSILON);
        assert_float_eq!(key2, 5.1, r2nd <= f64::EPSILON);
    }

    #[test]
    fn test_get_neighbors() {
        let mut path_finder = DStarLitePathFinder::new();
        path_finder.obstacle_map = Array2::zeros((5, 3));
        let query_node = utils::NodePosition(1, 0);
        assert_eq!(
            path_finder.get_neighbors(&query_node),
            vec![
                utils::NodePosition(0, 0),
                utils::NodePosition(0, 1),
                utils::NodePosition(1, 1),
                utils::NodePosition(2, 0),
                utils::NodePosition(2, 1),
            ]
        );
    }

    #[test]
    fn test_simple_grid_example() {
        let mut path_finder = DStarLitePathFinder::new();
        path_finder.obstacle_map = Array2::zeros((5, 3));
        path_finder.obstacle_map[[1, 1]] = f32::INFINITY;
        path_finder.obstacle_map[[2, 1]] = f32::INFINITY;
        path_finder.start = utils::NodePosition(1, 0);
        path_finder.goal = utils::NodePosition(4, 2);
        path_finder.reset();
        let path = path_finder.solve_impl();
        assert_eq!(path, vec![(1, 0), (2, 0), (3, 1), (4, 2)]);

        // Add obstacle at (3, 1)
        path_finder.obstacle_map[[3, 1]] = f32::INFINITY;
        let new_start = utils::NodePosition(2, 0);
        path_finder.key_modifier += compute_heuristic(&new_start, &path_finder.start);
        path_finder.start = new_start;
        let path = path_finder.solve_impl();
        assert_eq!(path, vec![(2, 0), (3, 0), (4, 1), (4, 2)]);
    }
}

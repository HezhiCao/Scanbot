#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "astar_path_finder.h"

namespace py = pybind11;
using py::literals::operator""_a;

namespace scanbot {
PYBIND11_MODULE(astar_path_finder, m) {
    py::class_<AStarPathFinder>(m, "AStarPathFinder")
        .def(py::init<>())
        .def("find", &AStarPathFinder::find, "Find a navigation path using A* algorithm",
             "obstacle_map"_a, "start_point"_a, "end_point"_a,
             "obstacle_cost"_a = 1000.0, "iteration_threshold"_a = 500);
}
}

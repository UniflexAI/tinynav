#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>


namespace py = pybind11;
//
// Function to perform raycasting on a depth image
//
// depth_image: np.ndarray (H, W) - depth values
// T_cam_to_world: np.ndarray (4, 4) - transformation matrix from camera to world
// grid_shape: list of int - shape of the grid (e.g., [x, y, z])
// fx, fy, cx, cy: float - camera intrinsic parameters
// origin: np.ndarray (3,) - origin of the raycasting
// step: int - step size for raycasting
// resolution: float - resolution of the grid
// 
// source: tinynav/cpp/raycasting.cpp
// test: tests/test_planning_node.py
//
extern Eigen::MatrixXd run_raycasting_cpp(
    py::array_t<float, py::array::c_style | py::array::forcecast> depth_image,
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_cam_to_world,
    std::vector<int> grid_shape,
    double fx, double fy, double cx, double cy,
    Eigen::Vector3d origin,
    int step,
    double resolution);

//
// Function to solve the pose graph optimization problem using Ceres Solver
//
// camera_poses: dict[int, np.ndarray (4x4)]
// constant_pose_index: dict[int, bool]
// relative_pose_constraints: list of tuples
//   Each tuple contains:
//   - cam_idx_i: int
//   - cam_idx_j: int
//   - relative_pose_j_i: np.ndarray (4x4)
//   - translation_weight: np.ndarray (3,)
//   - rotation_weight: np.ndarray (3,)
//
// source: tinynav/cpp/pose_graph_solver.cpp
// test: tests/test_map_node.py
//
extern std::unordered_map<int64_t, py::array_t<double>> pose_graph_solve(
    std::unordered_map<int64_t, py::array_t<double>> camera_poses,
    std::vector<std::tuple<int64_t, int64_t, py::array_t<double>, py::array_t<double>, py::array_t<double>>> relative_pose_constraints,
    std::unordered_map<int64_t, bool> constant_pose_index,
    int64_t max_iteration_num);


extern py::tuple ba_solve(
    std::unordered_map<int64_t, py::array_t<double>> camera_poses,
    std::unordered_map<int64_t, py::array_t<double>> point_3ds,
    std::vector<std::tuple<int64_t, int64_t, py::array_t<double>>> observations,
    py::array_t<double> K,
    std::unordered_map<int64_t, bool> constant_pose_index,
    std::vector<std::tuple<int64_t, int64_t, py::array_t<double>, py::array_t<double>, py::array_t<double>>> relative_pose_constraints
);



PYBIND11_MODULE(tinynav_cpp_bind, m) {
    m.doc() = "tinynav pybind11 binding";
    m.def("run_raycasting_cpp", &run_raycasting_cpp, "A function that performs raycasting on a depth image",
          py::arg("depth_image"),
          py::arg("T_cam_to_world"),
          py::arg("grid_shape"),
          py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
          py::arg("origin"),
          py::arg("step"),
          py::arg("resolution"));

   m.def(
        "pose_graph_solve",
        &pose_graph_solve,
        py::arg("camera_poses"),
        py::arg("relative_pose_constraints"),
        py::arg("constant_pose_index"),
        py::arg("max_iteration_num"),
        R"pbdoc(
            pose graph solve function.

            Parameters
            ----------
            camera_poses : dict[int, np.ndarray (4x4)]
                Dictionary mapping camera index to 4x4 pose matrix.
            relative_pose_constraints : list of tuples
                Each tuple contains:
                - cam_idx_i : int
                - cam_idx_j : int
                - relative_pose_j_i : np.ndarray (4x4)
                - translation_weight : np.ndarray (3,)
                - rotation_weight : np.ndarray (3,)
            constant_pose_index : dict[int, bool]
                Dictionary mapping camera index to whether the pose is constant.
            max_iteration_num : int
                Maximum number of iterations for the pose graph optimization.
            Returns
            -------
                optimized_camera_poses : dict[int, np.ndarray (4x4)]
                    Dictionary mapping camera index to optimized 4x4 pose matrix.
        )pbdoc"
    );

    m.def(
        "ba_solve",
        &ba_solve,
        py::arg("camera_poses"),
        py::arg("point_3ds"),
        py::arg("observations"),
        py::arg("K"),
        py::arg("constant_pose_index"),
        py::arg("relative_pose_constraints"),
        R"pbdoc(
            Bundle adjustment solve function.

            Parameters
            ----------
            camera_poses : dict[int, np.ndarray (4x4)]
                Dictionary mapping camera index to 4x4 pose matrix.
            point_3ds : dict[int, np.ndarray (3,)]
                Dictionary mapping point index to 3D point.
            observations : list[tuple[int, int, np.ndarray (2,)]]
                List of (camera_index, point_index, 2D observation).

            Returns
            -------
            tuple
                (optimized_camera_poses, optimized_point_3ds)
        )pbdoc"
    );
}

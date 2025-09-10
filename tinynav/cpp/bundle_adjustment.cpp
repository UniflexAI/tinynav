#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <tuple>
#include "ceres/ceres.h"  // Uncomment when you add Ceres code
#include "ceres/rotation.h"
#include <Eigen/Dense>
#include <ceres/autodiff_cost_function.h>

namespace py = pybind11;

// Type aliases for clarity
using CameraPoses = std::unordered_map<int64_t, py::array_t<double>>; // 4x4
using Point3Ds = std::unordered_map<int64_t, py::array_t<double>>;    // 3x1
using Observation = std::tuple<int64_t, int64_t, py::array_t<double>>;     // (cam_idx, pt_idx, 2x1)
using Observations = std::vector<Observation>;

using ConstantPoseIndex = std::unordered_map<int64_t, bool>;
// (cam_idx_i, cam_idx_j, relative_pose_j_i, translation_weight, rotation_weight)
using RelativePoseConstraints = std::vector<std::tuple<int64_t, int64_t, py::array_t<double>, py::array_t<double>, py::array_t<double>>>;


class ReprojectionError {
public:
    ReprojectionError(Eigen::Vector2d observed_keypoint, Eigen::Matrix3d K)
        : observed_keypoint_(observed_keypoint), K_(K) {}
    template<typename T>
    bool operator()(const T* camera_parameters, const T* point_parameters, T* residuals) const {
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> camera(camera_parameters);
        Eigen::Map<const Vector3T> point_3d(point_parameters);
        Eigen::Matrix<T, 3, 1> phi = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(camera.data() + 3);
        Eigen::Matrix<T, 3, 1> translation = Eigen::Map<const Vector3T>(camera.data());
        Eigen::Matrix<T, 3, 3> rotation;
        ceres::AngleAxisToRotationMatrix(phi.data(), rotation.data());
        Vector3T point_3d_in_camera = rotation.transpose() * (point_3d - translation);
        Vector3T reprojection = K_ * point_3d_in_camera;
        residuals[0] = reprojection[0] / reprojection[2] - observed_keypoint_[0];
        residuals[1] = reprojection[1] / reprojection[2] - observed_keypoint_[1];
        return true;
    }
    private:
    Eigen::Vector2d observed_keypoint_;
    Eigen::Matrix3d K_;
};

// This is a stub. You will fill in the Ceres logic.
py::tuple ba_solve(
    CameraPoses camera_poses,
    Point3Ds point_3ds,
    Observations observations,
    py::array_t<double> K,
    ConstantPoseIndex constant_pose_index,
    RelativePoseConstraints relative_pose_constraints
) {
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    std::map<int64_t, std::array<double, 6>> camera_parameters;
    int64_t min_cam_indx = std::numeric_limits<int64_t>::max();
    for (const auto& [cam_idx, cam_pose] : camera_poses) {
        std::array<double, 6> camera_parameter;
        auto cam_pose_buf = cam_pose.unchecked<2>();
        Eigen::Matrix4d cam_pose_eigen;
        for (ssize_t i = 0; i < 4; ++i)
            for (ssize_t j = 0; j < 4; ++j)
                cam_pose_eigen(i, j) = cam_pose_buf(i, j);
        
        Eigen::Matrix3d R = cam_pose_eigen.block<3, 3>(0, 0);
        Eigen::Vector3d t = cam_pose_eigen.block<3, 1>(0, 3);
        
        camera_parameter[0] = t[0];
        camera_parameter[1] = t[1];
        camera_parameter[2] = t[2];
        ceres::RotationMatrixToAngleAxis(R.data(), camera_parameter.data() + 3);
        camera_parameters[cam_idx] = camera_parameter;
    }

    std::map<int64_t, std::array<double, 3>> point_parameters;
    for (const auto& [pt_idx, pt_3d] : point_3ds) {
        auto pt_3d_buf = pt_3d.unchecked<1>();
        std::array<double, 3> point_parameter;
        point_parameter[0] = pt_3d_buf(0);
        point_parameter[1] = pt_3d_buf(1);
        point_parameter[2] = pt_3d_buf(2);
        point_parameters[pt_idx] = point_parameter;
    }

    ceres::LossFunction* loss_function = new ceres::HuberLoss(2.0);
    auto K_buf = K.unchecked<2>();
    Eigen::Matrix3d K_eigen;
    for (ssize_t i = 0; i < 3; ++i)
        for (ssize_t j = 0; j < 3; ++j)
            K_eigen(i, j) = K_buf(i, j);
    // observations is a list of tuples (cam_idx, pt_idx, 2x1)
    for (const auto& [cam_idx, pt_idx, obs] : observations) {
        auto obs_buf = obs.unchecked<1>();
        Eigen::Vector2d observed_keypoint(obs_buf(0), obs_buf(1));
        ceres::CostFunction* reprojection_error = new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(new ReprojectionError(observed_keypoint, K_eigen));
        problem.AddResidualBlock(reprojection_error, loss_function, camera_parameters.at(cam_idx).data(), point_parameters.at(pt_idx).data());
    }

    // constant_pose_index maybe None
    for (const auto& [cam_idx, is_constant] : constant_pose_index) {
        if (is_constant) {
            if (!problem.HasParameterBlock(camera_parameters.at(cam_idx).data())) {
                problem.AddParameterBlock(camera_parameters.at(cam_idx).data(), 6);
            }
            problem.SetParameterBlockConstant(camera_parameters.at(cam_idx).data());
        }
    }

    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 1024;
    options.num_threads = 1;

    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    CameraPoses optimized_camera_poses;
    for (const auto& [cam_idx, cam_parameter] : camera_parameters) {
      Eigen::Matrix<double, 4, 4, Eigen::RowMajor> cam_pose_eigen = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();
      Eigen::Matrix<double, 3, 3> R;
      ceres::AngleAxisToRotationMatrix(cam_parameter.data() + 3, R.data());
      cam_pose_eigen.block<3, 3>(0, 0) = R;
      cam_pose_eigen.block<3, 1>(0, 3) =
          Eigen::Map<const Eigen::Vector3d>(cam_parameter.data());
      optimized_camera_poses[cam_idx] =
          py::array_t<double>({4, 4}, cam_pose_eigen.data());
    }

    Point3Ds optimized_point_3ds;
    for (const auto& [pt_idx, pt_parameter] : point_parameters) {
        Eigen::Vector3d pt_3d_eigen = Eigen::Map<const Eigen::Vector3d>(pt_parameter.data());
        optimized_point_3ds[pt_idx] = py::array_t<double>({3}, pt_3d_eigen.data());
    }

    py::dict py_optimized_point_3ds;
    for (const auto& [pt_idx, pt_3d_eigen] : optimized_point_3ds) {
        py_optimized_point_3ds[py::int_(pt_idx)] = py::array_t<double>({3}, pt_3d_eigen.data());
    }
    return py::make_tuple(optimized_camera_poses, py_optimized_point_3ds);
}

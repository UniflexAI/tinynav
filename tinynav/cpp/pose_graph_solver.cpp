#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace py = pybind11;

// T_j_i @ (T_w_i)^-1 @ T_w_j = I
class RelativePoseError {
public:
    RelativePoseError(const Eigen::Matrix4d& relative_pose, Eigen::Vector3d translation_weight, Eigen::Vector3d rotation_weight)
        : relative_j_i_translation_(Eigen::Vector3d::Zero()), relative_j_i_rotation_lie_algebra_(Eigen::Vector3d::Zero()), translation_weight_(translation_weight), rotation_weight_(rotation_weight) {
            relative_j_i_translation_ = relative_pose.block<3, 1>(0, 3);
            Eigen::Matrix<double, 3, 3> relative_j_i_rotation = relative_pose.block<3, 3>(0, 0);
            ceres::RotationMatrixToAngleAxis(relative_j_i_rotation.data(), relative_j_i_rotation_lie_algebra_.data());
        }
    template<typename T>
    bool operator()(const T* camera_i, const T* camera_j, T* residuals) const {
        using TMatrix3 = Eigen::Matrix<T, 3, 3>;
        using TVector3 = Eigen::Matrix<T, 3, 1>;
        TVector3 translation_i = Eigen::Map<const TVector3>(camera_i);
        TVector3 rotation_i = Eigen::Map<const TVector3>(camera_i + 3);
        TVector3 translation_j = Eigen::Map<const TVector3>(camera_j);
        TVector3 rotation_j = Eigen::Map<const TVector3>(camera_j + 3);
        TMatrix3 R_i;
        ceres::AngleAxisToRotationMatrix(rotation_i.data(), R_i.data());
        TMatrix3 R_j;
        ceres::AngleAxisToRotationMatrix(rotation_j.data(), R_j.data());
        TMatrix3 relative_j_i_rotation;
        TVector3 relative_j_i_rotation_lie_algebra_T = relative_j_i_rotation_lie_algebra_.cast<T>();
        ceres::AngleAxisToRotationMatrix(relative_j_i_rotation_lie_algebra_T.data(), relative_j_i_rotation.data());
        TMatrix3 rotation_loss = relative_j_i_rotation * R_i.transpose() * R_j;
        ceres::RotationMatrixToAngleAxis(rotation_loss.data(), residuals + 3);
        Eigen::Map<TVector3> rotation_residuals_map(residuals + 3);
        rotation_residuals_map = rotation_weight_.cast<T>().asDiagonal() * rotation_residuals_map;
        TVector3 translation_residuals = relative_j_i_rotation * R_i.transpose() * (translation_j - translation_i) + relative_j_i_translation_;
        Eigen::Map<TVector3> translation_residuals_map(residuals);
        translation_residuals_map = translation_weight_.cast<T>().asDiagonal() * translation_residuals;
        return true;
    }
    Eigen::Vector3d relative_j_i_translation_;
    Eigen::Vector3d relative_j_i_rotation_lie_algebra_;
    Eigen::Vector3d translation_weight_;
    Eigen::Vector3d rotation_weight_;
};

std::unordered_map<int64_t, py::array_t<double>> pose_graph_solve(
    std::unordered_map<int64_t, py::array_t<double>> camera_poses,
    std::vector<std::tuple<int64_t, int64_t, py::array_t<double>, py::array_t<double>, py::array_t<double>>> relative_pose_constraints,
    std::unordered_map<int64_t, bool> constant_pose_index,
    int64_t max_iteration_num) {
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

    for (const auto& [cam_idx_i, cam_idx_j, relative_pose_j_i, translation_weight, rotation_weight] : relative_pose_constraints) {
        auto relative_pose_j_i_buf = relative_pose_j_i.unchecked<2>();
        Eigen::Matrix4d relative_pose_j_i_eigen;
        for (ssize_t i = 0; i < 4; ++i)
            for (ssize_t j = 0; j < 4; ++j)
                relative_pose_j_i_eigen(i, j) = relative_pose_j_i_buf(i, j);
        auto translation_weight_buf = translation_weight.unchecked<1>();
        Eigen::Vector3d translation_weight_eigen;
        for (ssize_t i = 0; i < 3; ++i)
            translation_weight_eigen[i] = translation_weight_buf(i);
        auto rotation_weight_buf = rotation_weight.unchecked<1>();
        Eigen::Vector3d rotation_weight_eigen;
        for (ssize_t i = 0; i < 3; ++i)
            rotation_weight_eigen[i] = rotation_weight_buf(i);
        ceres::CostFunction* relative_pose_error = new ceres::AutoDiffCostFunction<RelativePoseError, 6, 6, 6>(new RelativePoseError(relative_pose_j_i_eigen, translation_weight_eigen, rotation_weight_eigen));
        problem.AddResidualBlock(relative_pose_error, nullptr, camera_parameters.at(cam_idx_i).data(), camera_parameters.at(cam_idx_j).data());
    }

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
    options.max_num_iterations = max_iteration_num;
    options.num_threads = 1;

    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    std::unordered_map<int64_t, py::array_t<double>> optimized_camera_poses;
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

   
    return optimized_camera_poses;
}

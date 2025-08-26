#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace py = pybind11;

inline Eigen::Matrix3d skew(Eigen::Vector3d v) {
    Eigen::Matrix3d skew_matrix;
    skew_matrix << 0, -v[2], v[1],
                   v[2], 0, -v[0],
                   -v[1], v[0], 0;
    return skew_matrix;
}

Eigen::Matrix3d right_jacobian_rotation(Eigen::Matrix<double, 3, 1> v) {
    double theta = v.norm();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d skew_matrix = skew(v);

    if (theta < 1e-5) {
        return I - 0.5 * skew_matrix + (1.0 / 6.0) * skew_matrix * skew_matrix;
    } else {
        double theta2 = theta * theta;
        double theta3 = theta * theta2;
        double A = (1 - std::cos(theta)) / theta2;
        double B = (theta - std::sin(theta)) / theta3;
        return I - A * skew_matrix + B * skew_matrix * skew_matrix;
    }
}

class RelativePoseError : public ceres::SizedCostFunction<6, 6, 6> {
public:
    RelativePoseError(const Eigen::Matrix4d& relative_pose, Eigen::Vector3d translation_weight, Eigen::Vector3d rotation_weight)
        : relative_j_i_translation_(Eigen::Vector3d::Zero()), relative_j_i_rotation_lie_algebra_(Eigen::Vector3d::Zero()), translation_weight_(translation_weight), rotation_weight_(rotation_weight) {
            relative_j_i_translation_ = relative_pose.block<3, 1>(0, 3);
            Eigen::Matrix<double, 3, 3> relative_j_i_rotation = relative_pose.block<3, 3>(0, 0);
            ceres::RotationMatrixToAngleAxis(relative_j_i_rotation.data(), relative_j_i_rotation_lie_algebra_.data());
        }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> camera_i(parameters[0]);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> camera_j(parameters[1]);

        Eigen::Matrix<double, 3, 1> translation_i = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(camera_i.data());
        Eigen::Matrix<double, 3, 1> rotation_i = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(camera_i.data() + 3);

        Eigen::Matrix<double, 3, 1> translation_j = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(camera_j.data());
        Eigen::Matrix<double, 3, 1> rotation_j = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(camera_j.data() + 3);

        Eigen::Matrix<double, 3, 3> R_i;
        ceres::AngleAxisToRotationMatrix(rotation_i.data(), R_i.data());
        Eigen::Matrix<double, 3, 3> R_j;
        ceres::AngleAxisToRotationMatrix(rotation_j.data(), R_j.data());

        Eigen::Matrix<double, 3, 3> relative_j_i_rotation;
        ceres::AngleAxisToRotationMatrix(relative_j_i_rotation_lie_algebra_.data(), relative_j_i_rotation.data());
        Eigen::Matrix<double, 3, 3> rotation_loss = relative_j_i_rotation * R_i.transpose() * R_j;
        ceres::RotationMatrixToAngleAxis(rotation_loss.data(), residuals + 3);
        Eigen::Map<Eigen::Vector3d> rotation_residuals_map(residuals + 3);
        rotation_residuals_map = rotation_weight_.asDiagonal() * rotation_residuals_map;
        Eigen::Vector3d translation_loss = relative_j_i_rotation * R_i.transpose() * (translation_j - translation_i) + relative_j_i_translation_;
        Eigen::Map<Eigen::Vector3d> residuals_map(residuals);
        residuals_map = translation_weight_.asDiagonal() * translation_loss;

        if (jacobians != nullptr) {
            Eigen::Matrix<double, 6, 6> J_camera_i;
            J_camera_i.setZero();

            J_camera_i.block<3, 3>(0, 0) =
                translation_weight_.asDiagonal() * -relative_j_i_rotation * R_i.transpose();
            J_camera_i.block<3, 3>(3, 3) = 
                translation_weight_.asDiagonal() *
                relative_j_i_rotation *
                R_i.transpose() *
                skew(translation_j - translation_i) *
                right_jacobian_rotation(-rotation_i);
            J_camera_i.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
            J_camera_i.block<3, 3>(3, 3) =
                rotation_weight_.asDiagonal() * -R_j.transpose() * right_jacobian_rotation(-rotation_i);

            Eigen::Matrix<double, 6, 6> J_camera_j;
            J_camera_j.setZero();
            J_camera_j.block<3, 3>(0, 0) = translation_weight_.asDiagonal() * relative_j_i_rotation * R_i.transpose();
            J_camera_j.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
            J_camera_j.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
            J_camera_j.block<3, 3>(3, 3) = rotation_weight_.asDiagonal() * right_jacobian_rotation(rotation_j);
            if (jacobians[0] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(jacobians[0]).noalias() = J_camera_i;
            }
            if (jacobians[1] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(jacobians[1]).noalias() = J_camera_j;
            }
        }
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
    int64_t max_iteration_num
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
        ceres::CostFunction* relative_pose_error = new RelativePoseError(relative_pose_j_i_eigen, translation_weight_eigen, rotation_weight_eigen);
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

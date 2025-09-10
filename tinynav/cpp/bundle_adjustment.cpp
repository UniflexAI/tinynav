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

namespace py = pybind11;

// Type aliases for clarity
using CameraPoses = std::unordered_map<int64_t, py::array_t<double>>; // 4x4
using Point3Ds = std::unordered_map<int64_t, py::array_t<double>>;    // 3x1
using Observation = std::tuple<int64_t, int64_t, py::array_t<double>>;     // (cam_idx, pt_idx, 2x1)
using Observations = std::vector<Observation>;

using ConstantPoseIndex = std::unordered_map<int64_t, bool>;
// (cam_idx_i, cam_idx_j, relative_pose_j_i, translation_weight, rotation_weight)
using RelativePoseConstraints = std::vector<std::tuple<int64_t, int64_t, py::array_t<double>, py::array_t<double>, py::array_t<double>>>;

inline Eigen::Matrix3d skew(Eigen::Vector3d v) {
    Eigen::Matrix3d skew_matrix;
    skew_matrix << 0, -v[2], v[1],
                   v[2], 0, -v[0],
                   -v[1], v[0], 0;
    return skew_matrix;
}

inline Eigen::Matrix3d right_jacobian_rotation(Eigen::Matrix<double, 3, 1> v) {
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

class ReprojectionError : public ceres::SizedCostFunction<2, 6, 3> {
public:
    ReprojectionError(Eigen::Vector2d observed_keypoint, Eigen::Matrix3d K)
        : observed_keypoint_(observed_keypoint), K_(K) {}

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> camera(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> point_3d(parameters[1]);
        Eigen::Matrix<double, 3, 1> phi = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(camera.data() + 3);

        Eigen::Matrix<double, 3, 1> translation = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(camera.data());
        Eigen::Matrix<double, 3, 3> rotation;
        ceres::AngleAxisToRotationMatrix(camera.data() + 3, rotation.data());

        Eigen::Vector3d point_3d_in_camera = rotation.transpose() * (point_3d - translation);

        Eigen::Vector3d reprojection = K_ * point_3d_in_camera;
        // if (reprojection[2] <= 0) {
        //     std::cout << "camera: " << camera.transpose() << std::endl;
        //     std::cout << "point_3d: " << point_3d.transpose() << std::endl;
        //     std::cout << "reprojection: " << reprojection.transpose() << std::endl;
        //     std::cout << "observed_keypoint: " << observed_keypoint_.transpose() << std::endl;
        //     return false;
        // }

        residuals[0] = reprojection[0] / reprojection[2] - observed_keypoint_[0];
        residuals[1] = reprojection[1] / reprojection[2] - observed_keypoint_[1];

        if (jacobians != nullptr) {
            double x = point_3d_in_camera[0];
            double y = point_3d_in_camera[1];
            double z = point_3d_in_camera[2];
            double z_inv = 1.0 / z;
            double fx = K_(0, 0);
            double fy = K_(1, 1);
            double cx = K_(0, 2);
            double cy = K_(1, 2);
            Eigen::Matrix<double, 2, 3> J_camera_point;
            J_camera_point(0, 0) = fx * z_inv;
            J_camera_point(0, 1) = 0;
            J_camera_point(0, 2) = -fx * x * z_inv * z_inv;
            J_camera_point(1, 0) = 0;
            J_camera_point(1, 1) = fy * z_inv;
            J_camera_point(1, 2) = -fy * y * z_inv * z_inv;
            Eigen::Matrix<double, 3, 3> J_camera_point_to_rotation = rotation.transpose() * skew(point_3d - translation) * right_jacobian_rotation(-phi);

            Eigen::Matrix<double, 3, 3> J_camera_point_to_translation = -rotation.transpose();
            Eigen::Matrix<double, 3, 3> J_camera_point_to_point = rotation.transpose();

            Eigen::Matrix<double, 2, 3> J_rotation = J_camera_point * J_camera_point_to_rotation;
            Eigen::Matrix<double, 2, 3> J_translation = J_camera_point * J_camera_point_to_translation;
            Eigen::Matrix<double, 2, 3> J_point = J_camera_point * J_camera_point_to_point;

            Eigen::Matrix<double, 2, 6> J_camera;
            J_camera.block<2, 3>(0, 3) = J_rotation;
            J_camera.block<2, 3>(0, 0) = J_translation;

            if (jacobians[0] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>(jacobians[0]).noalias() = J_camera;
            }

            if (jacobians[1] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>(jacobians[1]).noalias() = J_point;
            }
        }
        return true;
    }

    private:
    Eigen::Vector2d observed_keypoint_;
    Eigen::Matrix3d K_;
};

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
        ceres::CostFunction* reprojection_error = new ReprojectionError(observed_keypoint, K_eigen);
        problem.AddResidualBlock(reprojection_error, loss_function, camera_parameters.at(cam_idx).data(), point_parameters.at(pt_idx).data());
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

    // constant_pose_index maybe None
    for (const auto& [cam_idx, is_constant] : constant_pose_index) {
        if (is_constant) {
            //std::cout << "set camera " << cam_idx << " to constant" << std::endl;
            if (!problem.HasParameterBlock(camera_parameters.at(cam_idx).data())) {
                problem.AddParameterBlock(camera_parameters.at(cam_idx).data(), 6);
            }
            problem.SetParameterBlockConstant(camera_parameters.at(cam_idx).data());
        }
    }

    options.linear_solver_type = ceres::DENSE_SCHUR;
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


py::array_t<double> test_relative_pose_error(py::array_t<double> relative_pose, py::array_t<double> translation_weight, py::array_t<double> rotation_weight) {
    std::cout << "test_relative_pose_error" << std::endl;
    auto relative_pose_buf = relative_pose.unchecked<2>();
    Eigen::Matrix4d relative_pose_eigen;
    for (ssize_t i = 0; i < 4; ++i)
        for (ssize_t j = 0; j < 4; ++j)
            relative_pose_eigen(i, j) = relative_pose_buf(i, j);
    std::cout << "relative_pose_eigen: " << relative_pose_eigen << std::endl;

    auto translation_weight_buf = translation_weight.unchecked<1>();
    auto rotation_weight_buf = rotation_weight.unchecked<1>();
    Eigen::Vector3d translation_weight_eigen;
    Eigen::Vector3d rotation_weight_eigen;
    for (ssize_t i = 0; i < 3; ++i) {
        translation_weight_eigen[i] = translation_weight_buf(i);
        rotation_weight_eigen[i] = rotation_weight_buf(i);
    }

    std::array<double, 6> camera_i = {0, 0, 0, 0, 0, 0};
    std::array<double, 6> camera_j = {0, 0, 0, 0, 0, 0};

    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    ceres::CostFunction* relative_pose_error = new RelativePoseError(relative_pose_eigen, translation_weight_eigen, rotation_weight_eigen);
    problem.AddResidualBlock(relative_pose_error, nullptr, camera_i.data(), camera_j.data());
    problem.SetParameterBlockConstant(camera_i.data());
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;


    Eigen::Matrix<double, 3, 3> rotation_j;
    ceres::AngleAxisToRotationMatrix(camera_j.data() + 3, rotation_j.data());
    Eigen::Vector3d translation_j = Eigen::Map<const Eigen::Vector3d>(camera_j.data());

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> relative_pose_eigen_optimized;
    relative_pose_eigen_optimized.block<3, 3>(0, 0) = rotation_j;
    relative_pose_eigen_optimized.block<3, 1>(0, 3) = translation_j;
    relative_pose_eigen_optimized(3, 3) = 1;
    // std::cout << "relative_pose_eigen_optimized: " << relative_pose_eigen_optimized << std::endl;

    py::array_t<double> relative_pose_eigen_optimized_py = py::array_t<double>({4, 4}, relative_pose_eigen_optimized.data());
    return relative_pose_eigen_optimized_py;
}

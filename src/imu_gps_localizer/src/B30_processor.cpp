#include "imu_gps_localizer/B30_processor.h"

#include "imu_gps_localizer/utils.h"

namespace ImuGpsLocalization
{
    B30Processor::B30Processor(const Eigen::Vector3d& I_p_B30)  : I_p_B30_(I_p_B30) { }
    bool B30Processor::UpdateStateByB30Depth(const B30DepthDataPtr B30_data_ptr, State* state){
        Eigen::Matrix<double, 3, 15> H;
        Eigen::Vector3d residual;
        ComputeJacobianAndResidual_B(B30_data_ptr, *state, &H, residual);
        Eigen::Matrix3d V;
        V.Zero();
        V(2,2) = B30_data_ptr->var;

        // EKF.
        const Eigen::MatrixXd& P = state->cov;
        const Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + V).inverse();
        const Eigen::VectorXd delta_x = K * residual;

        // Add delta_x to state.
        AddDeltaToState_B(delta_x, state);

        // Covarance.
        const Eigen::MatrixXd I_KH = Eigen::Matrix<double, 15, 15>::Identity() - K * H;
        state->cov = I_KH * P * I_KH.transpose() + K * V * K.transpose();
    }
    void B30Processor::ComputeJacobianAndResidual_B(const B30DepthDataPtr B30_data, 
                                                  const State& state,
                                                  Eigen::Matrix<double, 3, 15>* jacobian,
                                                  Eigen::Vector3d& residual) {
        const Eigen::Vector3d& G_p_I   = state.G_p_I;
        const Eigen::Matrix3d& G_R_I   = state.G_R_I;

        Eigen::Vector3d I_p_B30_t;

        I_p_B30_t = G_p_I + G_R_I * I_p_B30_;

        // Convert wgs84 to ENU frame.
        // Eigen::Vector3d G_p_Gps;
        // ConvertLLAToENU(init_lla, gps_data->lla, &G_p_Gps);

        // Compute residual.
        residual.setZero();
        residual(2) = B30_data->dep - I_p_B30_t(2);

        // Compute jacobian.
        Eigen::Vector3d A_h(0,0,1);
        jacobian->setZero();
        jacobian->block<3, 1>(0, 2) = A_h;
        jacobian->block<3, 3>(0, 6)  = - G_R_I * GetSkewMatrix(I_p_B30_);
    }

    void AddDeltaToState_B(const Eigen::Matrix<double, 15, 1>& delta_x, State* state) {
        state->G_p_I     += delta_x.block<3, 1>(0, 0);
        state->G_v_I     += delta_x.block<3, 1>(3, 0);
        state->acc_bias  += delta_x.block<3, 1>(9, 0);
        state->gyro_bias += delta_x.block<3, 1>(12, 0);

        if (delta_x.block<3, 1>(6, 0).norm() > 1e-12) {
            state->G_R_I *= Eigen::AngleAxisd(delta_x.block<3, 1>(6, 0).norm(), delta_x.block<3, 1>(6, 0).normalized()).toRotationMatrix();
        }
    }
} // namespace ImuGps

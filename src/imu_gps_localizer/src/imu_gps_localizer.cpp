#include "imu_gps_localizer/imu_gps_localizer.h"

#include <glog/logging.h>

#include "imu_gps_localizer/utils.h"

namespace ImuGpsLocalization {

ImuGpsLocalizer::ImuGpsLocalizer(const double acc_noise, const double gyro_noise,
                                 const double acc_bias_noise, const double gyro_bias_noise,
                                 const Eigen::Vector3d& I_p_Gps,
                                 const Eigen::Vector3d& I_p_B30) 
    : initialized_(false){
    initializer_ = std::make_unique<Initializer>(I_p_Gps);
    initializer_ = std::make_unique<Initializer>(I_p_B30);
    imu_processor_ = std::make_unique<ImuProcessor>(acc_noise, gyro_noise, 
                                                    acc_bias_noise, gyro_bias_noise,
                                                    Eigen::Vector3d(0., 0., -9.81007));
    gps_processor_ = std::make_unique<GpsProcessor>(I_p_Gps);
    B30_processor_ = std::make_unique<B30Processor>(I_p_B30);
}

bool ImuGpsLocalizer::ProcessImuData(const ImuDataPtr imu_data_ptr, State* fused_state) {
    if (!initialized_) {
        initializer_->AddImuData(imu_data_ptr);
        return false;
    }
    
    // Predict.
    imu_processor_->Predict(state_.imu_data_ptr, imu_data_ptr, &state_);

    // Convert ENU state to lla.
    ConvertENUToLLA(init_lla_, state_.G_p_I, &(state_.lla));
    *fused_state = state_;
    return true;
}

bool ImuGpsLocalizer::ProcessGpsPositionData(const GpsPositionDataPtr gps_data_ptr) {
    if (!initialized_) {
        if (!initializer_->AddGpsPositionData(gps_data_ptr, &state_)) {
            return false;
        }

        // Initialize the initial gps point used to convert lla to ENU.
        init_lla_ = gps_data_ptr->lla;
        
        initialized_ = true;

        LOG(INFO) << "[ProcessGpsPositionData]: System initialized!";
        return true;
    }

    // Update.
    gps_processor_->UpdateStateByGpsPosition(init_lla_, gps_data_ptr, &state_);

    return true;
}

bool ImuGpsLocalizer::ProcessB30DepthData(const B30DepthDataPtr B30_data_ptr) {
    if (!initialized_) {
        if (!initializer_->AddB30DepthData(B30_data_ptr, &state_)) {
            return false;
        }

        // Initialize the initial gps point used to convert lla to ENU.
        init_dep_ = B30_data_ptr->dep;
        
        initialized_ = true;

        LOG(INFO) << "[ProcessB30DepthData]: System initialized!";
        return true;
    }

    // Update.
    B30_processor_->UpdateStateByB30Depth(B30_data_ptr, &state_);

    return true;
}

}  // namespace ImuGpsLocalization
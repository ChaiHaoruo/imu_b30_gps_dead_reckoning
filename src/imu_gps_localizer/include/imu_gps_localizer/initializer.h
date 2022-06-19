#pragma once

#include <deque>

#include "imu_gps_localizer/base_type.h"

namespace ImuGpsLocalization {

constexpr int kImuDataBufferLength = 100;
constexpr int kAccStdLimit         = 3.;

class Initializer {
public:
    Initializer(const Eigen::Vector3d& init_I_p_Gps/*, float& init_B30_Noi*/);

    // Initializer(const float& init_dep);
    
    void AddImuData(const ImuDataPtr imu_data_ptr);

    bool AddGpsPositionData(const GpsPositionDataPtr gps_data_ptr, State* state);

    bool AddB30DepthData(const B30DepthDataPtr B30_data_ptr, State* state);

private:
    bool ComputeG_R_IFromImuData(Eigen::Matrix3d* G_R_I);

    Eigen::Vector3d init_I_p_Gps_;//I_p_Gps_是一开机时计算出来的gps原点与imu的转换关系，是固定的
    // double init_B30_Noi_;
    std::deque<ImuDataPtr> imu_buffer_;
};

}  // namespace ImuGpsLocalization
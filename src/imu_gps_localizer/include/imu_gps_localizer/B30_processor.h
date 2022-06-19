#pragma once 

#include <Eigen/Dense>

#include "imu_gps_localizer/base_type.h"

namespace ImuGpsLocalization {

class B30Processor {
public:
    B30Processor(const Eigen::Vector3d& I_p_B30);

    bool UpdateStateByB30Depth(const B30DepthDataPtr B30_data_ptr, State* state);

private:    
    void ComputeJacobianAndResidual_B(const B30DepthDataPtr B30_data, 
                                    const State& state,
                                    Eigen::Matrix<double, 3, 15>* jacobian,
                                    Eigen::Vector3d& residual);

    const Eigen::Vector3d I_p_B30_;  
};

void AddDeltaToState_B(const Eigen::Matrix<double, 15, 1>& delta_x, State* state);

}  // namespace ImuGpsLocalization
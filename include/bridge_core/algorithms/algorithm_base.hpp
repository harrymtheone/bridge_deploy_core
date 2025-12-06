#ifndef BRIDGE_CORE_ALGORITHM_BASE_HPP
#define BRIDGE_CORE_ALGORITHM_BASE_HPP

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "bridge_core/interfaces/algorithm_interface.hpp"
#include "bridge_core/core/transforms.hpp"

namespace bridge_core {

/**
 * @brief Base class for RL algorithms with common functionality
 *
 * Provides default implementation for observation computation and
 * target position calculation using ONNX Runtime. Derived classes 
 * override forward() method for specific network architectures.
 */
class AlgorithmBase : public AlgorithmInterface {
public:
    AlgorithmBase() = default;
    ~AlgorithmBase() override = default;

    void initialize(
        const rclcpp::Node::SharedPtr& node,
        const AlgorithmConfig& config,
        const RobotConfig& robot_config,
        const ControlConfig& control_config) override;

    void reset() override;

    void computeObservations(
        const RobotState& state,
        const Control& control) override;

    std::vector<float> forward() override = 0; // Pure virtual - must implement

protected:
    /**
     * @brief Compute target joint positions from network output
     * @param actions Raw actions from the network
     * @return Target positions for all DOFs
     */
    std::vector<float> computeTargetDofPos(const std::vector<float>& actions);

    /**
     * @brief Initialize observation tensors
     */
    virtual void initObservations();

    // Configuration
    AlgorithmConfig config_;
    RobotConfig robot_config_;
    ControlConfig control_config_;

    // ONNX Runtime
    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> ort_session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;

    // Observations
    Observations obs_;

    // Constants
    std::array<float, 3> gravity_vec_ = {0.0f, 0.0f, -1.0f};

    // Action clipping bounds
    float clip_actions_lower_;
    float clip_actions_upper_;

    // Number of activated DOFs (controlled by RL)
    int num_dof_activated_;

    // ROS interface
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr action_publisher_;
};

} // namespace bridge_core

#endif // BRIDGE_CORE_ALGORITHM_BASE_HPP


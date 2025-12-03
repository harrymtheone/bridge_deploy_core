#ifndef BRIDGE_CORE_ALGORITHM_INTERFACE_HPP
#define BRIDGE_CORE_ALGORITHM_INTERFACE_HPP

#include <memory>
#include <vector>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include "bridge_core/core/types.hpp"

namespace bridge_core {

/**
 * @brief Abstract interface for RL algorithms
 * 
 * Defines the contract that all RL policy implementations must follow.
 */
class AlgorithmInterface {
public:
    virtual ~AlgorithmInterface() = default;
    
    /**
     * @brief Initialize the algorithm
     * @param node ROS2 node for publishers/subscribers
     * @param config Algorithm configuration including model path
     * @param robot_config Robot configuration
     * @param control_config Control configuration with default positions
     */
    virtual void initialize(
        const rclcpp::Node::SharedPtr& node,
        const AlgorithmConfig& config,
        const RobotConfig& robot_config,
        const ControlConfig& control_config
    ) = 0;
    
    /**
     * @brief Reset observations and internal state
     * Called when transitioning to RL_READY state
     */
    virtual void reset() = 0;
    
    /**
     * @brief Compute observations from current robot state
     * @param state Current robot state
     * @param control User velocity commands
     */
    virtual void computeObservations(
        const RobotState& state,
        const Control& control
    ) = 0;
    
    /**
     * @brief Run forward pass through policy network
     * @return Target joint positions for all DOFs
     */
    virtual std::vector<float> forward() = 0;
    
    /**
     * @brief Get algorithm name/type
     * @return Algorithm identifier string
     */
    virtual std::string getName() const = 0;
};

} // namespace bridge_core

#endif // BRIDGE_CORE_ALGORITHM_INTERFACE_HPP


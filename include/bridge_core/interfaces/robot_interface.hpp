#ifndef BRIDGE_CORE_ROBOT_INTERFACE_HPP
#define BRIDGE_CORE_ROBOT_INTERFACE_HPP

#include <memory>
#include "bridge_core/core/types.hpp"

namespace bridge_core {

/**
 * @brief Abstract interface for robot communication
 * 
 * Defines the contract for getting robot state and sending commands.
 * Implementations can be for simulation (MuJoCo, Gazebo) or real robot SDKs.
 */
class RobotInterface {
public:
    virtual ~RobotInterface() = default;
    
    /**
     * @brief Initialize the robot interface
     * @param config Robot configuration
     */
    virtual void initialize(const RobotConfig& config) = 0;
    
    /**
     * @brief Get current robot state
     * @return Current state including IMU and motor data
     */
    virtual RobotState getState() = 0;
    
    /**
     * @brief Send command to robot
     * @param command Motor commands (position, velocity, torque, gains)
     */
    virtual void sendCommand(const RobotCommand& command) = 0;
    
    /**
     * @brief Check if interface is connected/ready
     * @return true if ready to communicate
     */
    virtual bool isReady() const = 0;
    
    /**
     * @brief Get robot name/type
     * @return Robot identifier string
     */
    virtual std::string getRobotName() const = 0;
};

/**
 * @brief Extended interface for simulation environments
 * 
 * Adds simulation-specific functionality like reset and pause.
 */
class SimInterface : public RobotInterface {
public:
    ~SimInterface() override = default;
    
    /**
     * @brief Reset simulation to initial state
     */
    virtual void resetSimulation() = 0;
    
    /**
     * @brief Pause/unpause simulation
     * @param pause true to pause, false to resume
     */
    virtual void pauseSimulation(bool pause) = 0;
    
    /**
     * @brief Check if simulation is paused
     */
    virtual bool isPaused() const = 0;
};

} // namespace bridge_core

#endif // BRIDGE_CORE_ROBOT_INTERFACE_HPP


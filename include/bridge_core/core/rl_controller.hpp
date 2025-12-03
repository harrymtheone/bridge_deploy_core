#ifndef BRIDGE_CORE_RL_CONTROLLER_HPP
#define BRIDGE_CORE_RL_CONTROLLER_HPP

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "bridge_core/core/types.hpp"
#include "bridge_core/core/state_machine.hpp"
#include "bridge_core/interfaces/robot_interface.hpp"
#include "bridge_core/interfaces/algorithm_interface.hpp"

namespace bridge_core {

/**
 * @brief Main controller orchestrating RL-based robot control
 *
 * Manages state machine, algorithm execution, and robot communication.
 * Uses ROS2 timers for control loops.
 */
class RLController {
public:
    /**
     * @brief Constructor
     * @param node ROS2 node handle
     * @param robot_interface Robot communication interface
     * @param algorithm RL algorithm implementation
     * @param config Complete configuration
     */
    RLController(
        const rclcpp::Node::SharedPtr& node,
        std::shared_ptr<RobotInterface> robot_interface,
        std::shared_ptr<AlgorithmInterface> algorithm,
        const Config& config);

    ~RLController() = default;

    /**
     * @brief Start the controller (creates timers)
     */
    void start();

    /**
     * @brief Stop the controller (cancels timers)
     */
    void stop();

    /**
     * @brief Get current state machine state
     */
    State getState() const { return state_machine_->getState(); }
    
    /**
     * @brief Process a state command manually
     */
    bool processCommand(StateCommand cmd) { return state_machine_->processCommand(cmd); }

private:
    // Timer callbacks
    void controlLoop();
    void rlLoop();

    // Input handling
    void joystickCallback(const sensor_msgs::msg::Joy::SharedPtr msg);

    // State machine callbacks
    void onStandingUp(State state);
    void onStanding(State state);
    void onRLReady(State state);
    void onRLRunning(State state);
    void onSittingDown(State state);

    // Control helpers
    void computeStandUpCommand();
    void computeSitDownCommand();

    // TF broadcasting
    void publishTF();

    // Configuration
    Config config_;
    rclcpp::Node::SharedPtr node_;

    // Core components
    std::shared_ptr<StateMachine> state_machine_;
    std::shared_ptr<RobotInterface> robot_interface_;
    std::shared_ptr<AlgorithmInterface> algorithm_;

    // State
    RobotCommand current_command_;
    Control control_;
    std::vector<float> start_positions_;

    // Joystick button change detection
    std::vector<int> prev_buttons_;

    // ROS interfaces
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::TimerBase::SharedPtr rl_timer_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Timing
    int motion_time_ = 0;
};

} // namespace bridge_core

#endif // BRIDGE_CORE_RL_CONTROLLER_HPP


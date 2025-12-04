#ifndef BRIDGE_CORE_MUJOCO_INTERFACE_HPP
#define BRIDGE_CORE_MUJOCO_INTERFACE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_srvs/srv/empty.hpp>
#include <mujoco_ros_msgs/msg/joint_control_cmd.hpp>

#include "bridge_core/interfaces/robot_interface.hpp"
#include "bridge_core/core/types.hpp"
#include "bridge_core/core/transforms.hpp"

#include <mutex>
#include <unordered_map>
#include <chrono>

namespace bridge_core
{

    /**
     * @brief Simulation interface that communicates with mujoco_ros via ROS topics
     *
     * This class implements the SimInterface and provides communication with a
     * MuJoCo simulation running via mujoco_rospy. It subscribes to sensor data
     * (joint states, IMU, odometry) and publishes joint control commands.
     */
    class MujocoSimInterface : public SimInterface
    {
    public:
        /**
         * @brief Construct a new MujocoSimInterface
         * @param node ROS2 node to use for creating publishers/subscribers
         */
        explicit MujocoSimInterface(rclcpp::Node::SharedPtr node);

        ~MujocoSimInterface() override = default;

        // RobotInterface methods
        void initialize(const RobotConfig &config) override;
        RobotState getState() override;
        void sendCommand(const RobotCommand &command) override;
        bool isReady() const override;
        std::string getRobotName() const override;

        // SimInterface methods
        void resetSimulation() override;
        void pauseSimulation(bool pause) override;
        bool isPaused() const override;

    private:
        void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
        void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
        void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
        void watchdogCallback();

        rclcpp::Node::SharedPtr node_;
        RobotConfig config_;

        // State
        std::mutex state_mutex_;
        RobotState state_;
        std::unordered_map<std::string, int> joint_name_to_idx_;
        bool is_ready_ = false;
        bool is_paused_ = false;

        // Timestamps for watchdog
        std::chrono::steady_clock::time_point last_joint_state_time_;
        std::chrono::steady_clock::time_point last_imu_time_;
        std::chrono::steady_clock::time_point last_odom_time_;

        // Publisher for combined control command
        rclcpp::Publisher<mujoco_ros_msgs::msg::JointControlCmd>::SharedPtr joint_cmd_pub_;

        // Subscribers
        rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

        // Watchdog timer
        rclcpp::TimerBase::SharedPtr watchdog_timer_;

        // Service clients
        rclcpp::Client<std_srvs::srv::Empty>::SharedPtr reset_client_;
    };

} // namespace bridge_core

#endif // BRIDGE_CORE_MUJOCO_INTERFACE_HPP

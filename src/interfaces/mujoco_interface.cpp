#include "bridge_core/interfaces/mujoco_interface.hpp"

namespace bridge_core
{

    MujocoSimInterface::MujocoSimInterface(rclcpp::Node::SharedPtr node)
        : node_(node)
    {
        // Publisher for combined joint control command
        joint_cmd_pub_ = node_->create_publisher<mujoco_ros_msgs::msg::JointControlCmd>(
            "/mujoco/joint_cmd", 10);

        // Subscribers for sensor data
        joint_state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
            "/mujoco/joint_states", 10,
            std::bind(&MujocoSimInterface::jointStateCallback, this, std::placeholders::_1));

        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            "/mujoco/imu", 10,
            std::bind(&MujocoSimInterface::imuCallback, this, std::placeholders::_1));

        odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
            "/mujoco/odom", 10,
            std::bind(&MujocoSimInterface::odomCallback, this, std::placeholders::_1));

        // Service clients
        reset_client_ = node_->create_client<std_srvs::srv::Empty>("/mujoco/reset");

        // Watchdog timer to check for stale data (runs every 1 second)
        watchdog_timer_ = node_->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&MujocoSimInterface::watchdogCallback, this));

        RCLCPP_INFO(node_->get_logger(), "MujocoSimInterface initialized");
    }

    void MujocoSimInterface::initialize(const RobotConfig &config)
    {
        config_ = config;

        std::lock_guard<std::mutex> lock(state_mutex_);
        state_.resize(static_cast<size_t>(config.num_dof));

        // Build name-to-index mapping
        for (size_t i = 0; i < config.joint_names.size(); ++i)
        {
            joint_name_to_idx_[config.joint_names[i]] = static_cast<int>(i);
        }

        is_ready_ = true;
    }

    RobotState MujocoSimInterface::getState()
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }

    void MujocoSimInterface::sendCommand(const RobotCommand &command)
    {
        mujoco_ros_msgs::msg::JointControlCmd cmd_msg;
        cmd_msg.header.stamp = node_->now();

        size_t num_joints = config_.joint_names.size();

        // Joint names (for proper mapping on receiver side)
        cmd_msg.name = config_.joint_names;

        // Position commands
        cmd_msg.position.resize(num_joints);
        for (size_t i = 0; i < num_joints && i < command.motor.q.size(); ++i)
        {
            cmd_msg.position[i] = static_cast<double>(command.motor.q[i]);
        }

        // Velocity commands
        cmd_msg.velocity.resize(num_joints);
        for (size_t i = 0; i < num_joints && i < command.motor.dq.size(); ++i)
        {
            cmd_msg.velocity[i] = static_cast<double>(command.motor.dq[i]);
        }

        // Torque commands
        cmd_msg.torque.resize(num_joints);
        for (size_t i = 0; i < num_joints && i < command.motor.tau.size(); ++i)
        {
            cmd_msg.torque[i] = static_cast<double>(command.motor.tau[i]);
        }

        // Kp gains
        cmd_msg.kp.resize(num_joints);
        for (size_t i = 0; i < num_joints && i < command.motor.kp.size(); ++i)
        {
            cmd_msg.kp[i] = static_cast<double>(command.motor.kp[i]);
        }

        // Kd gains
        cmd_msg.kd.resize(num_joints);
        for (size_t i = 0; i < num_joints && i < command.motor.kd.size(); ++i)
        {
            cmd_msg.kd[i] = static_cast<double>(command.motor.kd[i]);
        }

        joint_cmd_pub_->publish(cmd_msg);
    }

    bool MujocoSimInterface::isReady() const
    {
        return is_ready_;
    }

    std::string MujocoSimInterface::getRobotName() const
    {
        return config_.robot_name;
    }

    void MujocoSimInterface::resetSimulation()
    {
        if (reset_client_->wait_for_service(std::chrono::seconds(1)))
        {
            auto request = std::make_shared<std_srvs::srv::Empty::Request>();
            reset_client_->async_send_request(request);
            RCLCPP_INFO(node_->get_logger(), "Reset request sent");
        }
        else
        {
            RCLCPP_WARN(node_->get_logger(), "Reset service not available");
        }
    }

    void MujocoSimInterface::pauseSimulation(bool pause)
    {
        is_paused_ = pause;
        // Note: Could add pause service client if needed
    }

    bool MujocoSimInterface::isPaused() const
    {
        return is_paused_;
    }

    void MujocoSimInterface::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);

        last_joint_state_time_ = std::chrono::steady_clock::now();

        for (size_t i = 0; i < msg->name.size(); ++i)
        {
            auto it = joint_name_to_idx_.find(msg->name[i]);
            if (it != joint_name_to_idx_.end())
            {
                int idx = it->second;
                if (idx >= 0 && static_cast<size_t>(idx) < state_.motor.q.size())
                {
                    if (i < msg->position.size())
                    {
                        state_.motor.q[idx] = static_cast<float>(msg->position[i]);
                    }
                    if (i < msg->velocity.size())
                    {
                        state_.motor.dq[idx] = static_cast<float>(msg->velocity[i]);
                    }
                }
            }
        }
    }

    void MujocoSimInterface::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);

        last_imu_time_ = std::chrono::steady_clock::now();

        state_.imu.quaternion[0] = static_cast<float>(msg->orientation.w);
        state_.imu.quaternion[1] = static_cast<float>(msg->orientation.x);
        state_.imu.quaternion[2] = static_cast<float>(msg->orientation.y);
        state_.imu.quaternion[3] = static_cast<float>(msg->orientation.z);

        state_.imu.gyroscope[0] = static_cast<float>(msg->angular_velocity.x);
        state_.imu.gyroscope[1] = static_cast<float>(msg->angular_velocity.y);
        state_.imu.gyroscope[2] = static_cast<float>(msg->angular_velocity.z);

        state_.imu.accelerometer[0] = static_cast<float>(msg->linear_acceleration.x);
        state_.imu.accelerometer[1] = static_cast<float>(msg->linear_acceleration.y);
        state_.imu.accelerometer[2] = static_cast<float>(msg->linear_acceleration.z);

        // Compute euler angles from quaternion
        state_.imu.euler = quatToEuler(state_.imu.quaternion);
    }

    void MujocoSimInterface::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);

        last_odom_time_ = std::chrono::steady_clock::now();

        // Linear velocity in body frame
        state_.lin_vel_b[0] = static_cast<float>(msg->twist.twist.linear.x);
        state_.lin_vel_b[1] = static_cast<float>(msg->twist.twist.linear.y);
        state_.lin_vel_b[2] = static_cast<float>(msg->twist.twist.linear.z);
    }

    void MujocoSimInterface::watchdogCallback()
    {
        auto now = std::chrono::steady_clock::now();
        constexpr auto timeout = std::chrono::seconds(1);

        std::lock_guard<std::mutex> lock(state_mutex_);

        // Check joint state timeout
        if (last_joint_state_time_.time_since_epoch().count() > 0)
        {
            auto elapsed = now - last_joint_state_time_;
            if (elapsed > timeout)
            {
                RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                                     "JointState not received for %.1f seconds!",
                                     std::chrono::duration<float>(elapsed).count());
            }
        }
        else
        {
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                                 "JointState never received!");
        }

        // Check IMU timeout
        if (last_imu_time_.time_since_epoch().count() > 0)
        {
            auto elapsed = now - last_imu_time_;
            if (elapsed > timeout)
            {
                RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                                     "IMU not received for %.1f seconds!",
                                     std::chrono::duration<float>(elapsed).count());
            }
        }
        else
        {
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                                 "IMU never received!");
        }

        // Check Odom timeout
        if (last_odom_time_.time_since_epoch().count() > 0)
        {
            auto elapsed = now - last_odom_time_;
            if (elapsed > timeout)
            {
                RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                                     "Odometry not received for %.1f seconds!",
                                     std::chrono::duration<float>(elapsed).count());
            }
        }
        else
        {
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                                 "Odometry never received!");
        }
    }

} // namespace bridge_core

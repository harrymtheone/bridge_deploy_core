#include "bridge_core/algorithms/algorithm_base.hpp"
#include <cmath>
#include <algorithm>

namespace bridge_core
{

    void AlgorithmBase::initialize(
        const rclcpp::Node::SharedPtr &node,
        const AlgorithmConfig &config,
        const RobotConfig &robot_config,
        const ControlConfig &control_config)
    {
        node_ = node;
        config_ = config;
        robot_config_ = robot_config;
        control_config_ = control_config;

        // Initialize ONNX Runtime
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "bridge_core");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(4);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Load ONNX model
        try
        {
            ort_session_ = std::make_unique<Ort::Session>(*ort_env_, config_.model_path.c_str(), *session_options_);
            RCLCPP_INFO(node_->get_logger(), "Successfully loaded ONNX model from: %s",
                        config_.model_path.c_str());
        }
        catch (const Ort::Exception &e)
        {
            RCLCPP_ERROR(node_->get_logger(), "Error loading ONNX model: %s", e.what());
            throw;
        }

        // Initialize observations
        initObservations();

        // Create action publisher for debugging
        action_publisher_ = node_->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/bridge/actions", 10);
    }

    void AlgorithmBase::initObservations()
    {
        num_dof_activated_ = static_cast<int>(robot_config_.dof_activated_ids.size());

        // Initialize observation struct
        obs_.resize(static_cast<size_t>(num_dof_activated_));

        // Initialize action clipping bounds
        clip_actions_lower_ = config_.clip_actions_lower / config_.action_scale;
        clip_actions_upper_ = config_.clip_actions_upper / config_.action_scale;
    }

    void AlgorithmBase::reset()
    {
        std::fill(obs_.last_actions.begin(), obs_.last_actions.end(), 0.0f);
    }

    void AlgorithmBase::computeObservations(
        const RobotState &state,
        const Control &control)
    {
        // IMU data
        obs_.base_quat[0] = state.imu.quaternion[0];
        obs_.base_quat[1] = state.imu.quaternion[1];
        obs_.base_quat[2] = state.imu.quaternion[2];
        obs_.base_quat[3] = state.imu.quaternion[3];

        obs_.base_euler[0] = state.imu.euler[0];
        obs_.base_euler[1] = state.imu.euler[1];
        obs_.base_euler[2] = state.imu.euler[2];

        // Gravity projection in base frame
        std::array<float, 3> gravity_proj = quatRotateInv(state.imu.quaternion, gravity_vec_);
        obs_.gravity_proj[0] = gravity_proj[0];
        obs_.gravity_proj[1] = gravity_proj[1];
        obs_.gravity_proj[2] = gravity_proj[2];

        // Velocities (scaled)
        obs_.lin_vel[0] = state.lin_vel_b[0] * config_.lin_vel_scale;
        obs_.lin_vel[1] = state.lin_vel_b[1] * config_.lin_vel_scale;
        obs_.lin_vel[2] = state.lin_vel_b[2] * config_.lin_vel_scale;

        obs_.ang_vel[0] = state.imu.gyroscope[0] * config_.ang_vel_scale;
        obs_.ang_vel[1] = state.imu.gyroscope[1] * config_.ang_vel_scale;
        obs_.ang_vel[2] = state.imu.gyroscope[2] * config_.ang_vel_scale;

        // Commands (scaled)
        obs_.commands[0] = control.x * config_.commands_scale[0];
        obs_.commands[1] = control.y * config_.commands_scale[1];
        obs_.commands[2] = control.yaw * config_.commands_scale[2];

        // Joint states (only activated DOFs, scaled)
        for (int i = 0; i < num_dof_activated_; ++i)
        {
            int dof_idx = robot_config_.dof_activated_ids[i];
            float dof_pos = state.motor.q[dof_idx];
            float default_pos = control_config_.default_dof_pos[dof_idx];

            obs_.dof_pos[i] = (dof_pos - default_pos) * config_.dof_pos_scale;
            obs_.dof_vel[i] = state.motor.dq[dof_idx] * config_.dof_vel_scale;
        }
    }

    std::vector<float> AlgorithmBase::computeTargetDofPos(const std::vector<float> &actions)
    {
        // Start with default positions for all DOFs
        std::vector<float> target_dof_pos = control_config_.default_dof_pos;

        // Publish actions for debugging
        std_msgs::msg::Float64MultiArray msg;
        msg.data.resize(static_cast<size_t>(num_dof_activated_));

        // Apply actions to activated DOFs (adding to default positions)
        for (int i = 0; i < num_dof_activated_; ++i)
        {
            // Clip raw action
            float clipped_action = std::clamp(actions[i], clip_actions_lower_, clip_actions_upper_);
            
            // Compute scaled action for control
            float scaled_action = clipped_action * config_.action_scale;

            int dof_idx = robot_config_.dof_activated_ids[i];
            target_dof_pos[dof_idx] += scaled_action;
            msg.data[i] = target_dof_pos[dof_idx];
        }

        action_publisher_->publish(msg);

        return target_dof_pos;
    }

} // namespace bridge_core

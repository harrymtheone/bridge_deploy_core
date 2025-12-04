#include "bridge_core/core/rl_controller.hpp"

namespace bridge_core
{

    RLController::RLController(
        const rclcpp::Node::SharedPtr &node,
        std::shared_ptr<RobotInterface> robot_interface,
        std::shared_ptr<AlgorithmInterface> algorithm,
        const Config &config)
        : config_(config), node_(node), robot_interface_(robot_interface), algorithm_(algorithm)
    {
        // Initialize state machine
        state_machine_ = std::make_shared<StateMachine>(node_->get_logger());

        // Set transition durations from config
        state_machine_->setTransitionDuration(State::STANDING_UP, config_.control.stand_up_time);
        state_machine_->setTransitionDuration(State::SITTING_DOWN, config_.control.sit_down_time);

        // Register state callbacks
        state_machine_->onStateEntry(State::STANDING_UP,
                                     std::bind(&RLController::onStandingUp, this, std::placeholders::_1));
        state_machine_->onStateEntry(State::STANDING,
                                     std::bind(&RLController::onStanding, this, std::placeholders::_1));
        state_machine_->onStateEntry(State::RL_READY,
                                     std::bind(&RLController::onRLReady, this, std::placeholders::_1));
        state_machine_->onStateEntry(State::RL_RUNNING,
                                     std::bind(&RLController::onRLRunning, this, std::placeholders::_1));
        state_machine_->onStateEntry(State::SITTING_DOWN,
                                     std::bind(&RLController::onSittingDown, this, std::placeholders::_1));

        // Initialize command structure with config defaults
        current_command_.resize(static_cast<size_t>(config_.robot.num_dof));
        current_command_.motor.q = config_.control.default_dof_pos;
        current_command_.motor.kp = config_.control.fixed_kp;
        current_command_.motor.kd = config_.control.fixed_kd;

        // Initialize start_positions_ to default (safe fallback)
        start_positions_ = config_.control.default_dof_pos;

        // Initialize control
        control_.x = 0.0f;
        control_.y = 0.0f;
        control_.yaw = 0.0f;

        // Setup ROS interfaces
        joy_sub_ = node_->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", 10,
            std::bind(&RLController::joystickCallback, this, std::placeholders::_1));

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);

        RCLCPP_INFO(node_->get_logger(), "RLController initialized with algorithm: %s",
                    algorithm_->getName().c_str());
    }

    void RLController::start()
    {
        // Create timers with appropriate rates
        double control_dt = static_cast<double>(config_.algorithm.dt);
        double rl_dt = static_cast<double>(config_.algorithm.dt * config_.algorithm.decimation);

        control_timer_ = node_->create_wall_timer(
            std::chrono::duration<double>(control_dt),
            std::bind(&RLController::controlLoop, this));

        rl_timer_ = node_->create_wall_timer(
            std::chrono::duration<double>(rl_dt),
            std::bind(&RLController::rlLoop, this));

        RCLCPP_INFO(node_->get_logger(), "RLController started (control: %.1f Hz, RL: %.1f Hz)",
                    1.0 / control_dt, 1.0 / rl_dt);
    }

    void RLController::stop()
    {
        if (control_timer_)
            control_timer_->cancel();
        if (rl_timer_)
            rl_timer_->cancel();

        RCLCPP_INFO(node_->get_logger(), "RLController stopped");
    }

    void RLController::controlLoop()
    {
        motion_time_++;

        // Update state machine
        state_machine_->update(config_.algorithm.dt);

        // Get current robot state (for transitions that need it)
        // RobotState robot_state = robot_interface_->getState();

        // Compute command based on current state
        State current_state = state_machine_->getState();

        switch (current_state)
        {
        case State::IDLE:
            // Do not control the robot - but still publish TF for visualization
            publishTF();
            return;

        case State::STANDING_UP:
            computeStandUpCommand();
            break;

        case State::SITTING_DOWN:
            computeSitDownCommand();
            break;

        case State::STANDING:
        case State::RL_READY:
        case State::RL_RUNNING:
            // Command is set by RL loop or held at default position
            break;

        default:
            break;
        }

        // Send command to robot
        robot_interface_->sendCommand(current_command_);

        // Publish TF
        publishTF();
    }

    void RLController::rlLoop()
    {
        if (state_machine_->getState() != State::RL_RUNNING)
        {
            return;
        }

        // Get current robot state
        RobotState robot_state = robot_interface_->getState();

        // Compute observations
        algorithm_->computeObservations(robot_state, control_);

        // Run forward pass
        std::vector<float> target_positions = algorithm_->forward();

        // Update command with RL output
        // 1. Set all joints to default positions first (safeguard for non-activated joints)
        current_command_.motor.q = config_.control.default_dof_pos;

        // 2. Apply RL targets for activated joints
        // Note: target_positions from algorithm already includes default_pos + action
        if (target_positions.size() == current_command_.motor.q.size())
        {
            for (int dof_idx : config_.robot.dof_activated_indices)
            {
                size_t idx = static_cast<size_t>(dof_idx);
                if (idx < target_positions.size())
                {
                    current_command_.motor.q[idx] = target_positions[idx];
                }
            }
        }

        // Zero velocity and torque for PD control
        std::fill(current_command_.motor.dq.begin(), current_command_.motor.dq.end(), 0.0f);
        std::fill(current_command_.motor.tau.begin(), current_command_.motor.tau.end(), 0.0f);

        // Set gains: fixed for non-active joints, RL gains for active joints
        current_command_.motor.kp = config_.control.fixed_kp;
        current_command_.motor.kd = config_.control.fixed_kd;

        // Override with RL gains for activated joints
        // rl_kp/rl_kd are indexed by position in dof_activated_indices array (0 to n-1)
        for (size_t i = 0; i < config_.robot.dof_activated_indices.size(); ++i)
        {
            int dof_idx = config_.robot.dof_activated_indices[i];
            if (dof_idx >= 0 && static_cast<size_t>(dof_idx) < current_command_.motor.kp.size())
            {
                if (i < config_.control.rl_kp.size())
                {
                    current_command_.motor.kp[dof_idx] = config_.control.rl_kp[i];
                }
                if (i < config_.control.rl_kd.size())
                {
                    current_command_.motor.kd[dof_idx] = config_.control.rl_kd[i];
                }
            }
        }
    }

    void RLController::joystickCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
    {
        // Update control commands from joystick
        if (msg->axes.size() >= 4)
        {
            control_.x = msg->axes[1] * config_.algorithm.joystick_scale[0];
            control_.y = msg->axes[0] * config_.algorithm.joystick_scale[1];
            control_.yaw = msg->axes[3] * config_.algorithm.joystick_scale[2];
        }

        // Helper to detect rising edge
        auto isButtonPressed = [&](size_t idx) -> bool
        {
            if (idx >= msg->buttons.size())
                return false;
            bool current = msg->buttons[idx] == 1;
            bool previous = (idx < prev_buttons_.size()) ? (prev_buttons_[idx] == 1) : false;
            return current && !previous;
        };

        // L1 (4): Stand up from IDLE, or Stop RL from RL states
        if (isButtonPressed(4))
        {
            State current = state_machine_->getState();
            if (current == State::IDLE)
            {
                state_machine_->processCommand(StateCommand::STAND_UP);
            }
            else if (current == State::RL_RUNNING || current == State::RL_READY)
            {
                state_machine_->processCommand(StateCommand::STOP_RL);
            }
        }

        // R1 + A (5 + 0): Start RL
        if (msg->buttons.size() > 5 && msg->buttons[5] == 1 && isButtonPressed(0))
        {
            state_machine_->processCommand(StateCommand::START_RL);
        }

        // B (1): Sit down from STANDING
        if (isButtonPressed(1))
        {
            state_machine_->processCommand(StateCommand::SIT_DOWN);
        }

        // Select/Back (6): Reset simulation
        if (isButtonPressed(6))
        {
            RCLCPP_INFO(node_->get_logger(), "Reset requested");
            auto sim = std::dynamic_pointer_cast<SimInterface>(robot_interface_);
            if (sim)
            {
                sim->resetSimulation();
            }
            state_machine_->processCommand(StateCommand::RESET);
        }

        prev_buttons_ = msg->buttons;
    }

    void RLController::onStandingUp([[maybe_unused]] State state)
    {
        RobotState robot_state = robot_interface_->getState();
        start_positions_ = robot_state.motor.q;

        current_command_.motor.kp = config_.control.fixed_kp;
        current_command_.motor.kd = config_.control.fixed_kd;

        current_command_.motor.q = start_positions_;
        std::fill(current_command_.motor.dq.begin(), current_command_.motor.dq.end(), 0.0f);
        std::fill(current_command_.motor.tau.begin(), current_command_.motor.tau.end(), 0.0f);
    }

    void RLController::onStanding([[maybe_unused]] State state)
    {
        current_command_.motor.kp = config_.control.fixed_kp;
        current_command_.motor.kd = config_.control.fixed_kd;

        current_command_.motor.q = config_.control.default_dof_pos;
        std::fill(current_command_.motor.dq.begin(), current_command_.motor.dq.end(), 0.0f);
        std::fill(current_command_.motor.tau.begin(), current_command_.motor.tau.end(), 0.0f);
    }

    void RLController::onRLReady([[maybe_unused]] State state)
    {
        algorithm_->reset();

        // Start with fixed gains for all joints
        current_command_.motor.kp = config_.control.fixed_kp;
        current_command_.motor.kd = config_.control.fixed_kd;

        current_command_.motor.q = config_.control.default_dof_pos;

        // Override with RL gains for activated joints only
        // rl_kp/rl_kd are indexed by position in dof_activated_indices array (0 to n-1)
        for (size_t i = 0; i < config_.robot.dof_activated_indices.size(); ++i)
        {
            int dof_idx = config_.robot.dof_activated_indices[i];
            if (dof_idx >= 0 && static_cast<size_t>(dof_idx) < current_command_.motor.kp.size())
            {
                if (i < config_.control.rl_kp.size())
                {
                    current_command_.motor.kp[dof_idx] = config_.control.rl_kp[i];
                }
                if (i < config_.control.rl_kd.size())
                {
                    current_command_.motor.kd[dof_idx] = config_.control.rl_kd[i];
                }
            }
        }
    }

    void RLController::onRLRunning([[maybe_unused]] State state)
    {
        // RL loop will handle it
    }

    void RLController::onSittingDown([[maybe_unused]] State state)
    {
        RobotState robot_state = robot_interface_->getState();
        start_positions_ = robot_state.motor.q;

        current_command_.motor.kp = config_.control.fixed_kp;
        current_command_.motor.kd = config_.control.fixed_kd;

        current_command_.motor.q = start_positions_;
        std::fill(current_command_.motor.dq.begin(), current_command_.motor.dq.end(), 0.0f);
        std::fill(current_command_.motor.tau.begin(), current_command_.motor.tau.end(), 0.0f);
    }

    void RLController::computeStandUpCommand()
    {
        float progress = state_machine_->getTransitionProgress();

        // Interpolate ALL joints from start_positions to default_dof_pos
        size_t num_dof = current_command_.motor.q.size();
        for (size_t i = 0; i < num_dof; ++i)
        {
            float start_pos = start_positions_[i];
            float default_pos = config_.control.default_dof_pos[i];

            // Smooth interpolation for all joints
            current_command_.motor.q[i] = (1.0f - progress) * start_pos + progress * default_pos;
        }
    }

    void RLController::computeSitDownCommand()
    {
        float progress = state_machine_->getTransitionProgress();

        // Interpolate ALL joints from start_positions to zero (sit/rest position)
        size_t num_dof = current_command_.motor.q.size();
        for (size_t i = 0; i < num_dof; ++i)
        {
            float start_pos = start_positions_[i];
            float sit_pos = 0.0f; // XML default / rest position

            // Smooth interpolation for all joints
            current_command_.motor.q[i] = (1.0f - progress) * start_pos + progress * sit_pos;
        }
    }

    void RLController::publishTF()
    {
        RobotState robot_state = robot_interface_->getState();

        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = node_->now();
        transform.header.frame_id = "base_link";
        transform.child_frame_id = "hmap";

        transform.transform.translation.x = 0.0;
        transform.transform.translation.y = 0.0;
        transform.transform.translation.z = -0.7;

        // Inverse quaternion to keep hmap horizontal
        transform.transform.rotation.w = robot_state.imu.quaternion[0];
        transform.transform.rotation.x = -robot_state.imu.quaternion[1];
        transform.transform.rotation.y = -robot_state.imu.quaternion[2];
        transform.transform.rotation.z = -robot_state.imu.quaternion[3];

        tf_broadcaster_->sendTransform(transform);
    }

} // namespace bridge_core

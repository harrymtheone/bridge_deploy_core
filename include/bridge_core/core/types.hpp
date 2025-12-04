#ifndef BRIDGE_CORE_TYPES_HPP
#define BRIDGE_CORE_TYPES_HPP

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <yaml-cpp/yaml.h>

namespace bridge_core {

// State machine states
enum class State : uint8_t {
    IDLE,
    STANDING_UP,
    STANDING,
    RL_READY,
    RL_RUNNING,
    SITTING_DOWN,
    ERROR
};

// State machine events/commands
enum class StateCommand : uint8_t {
    NONE,
    STAND_UP,
    START_RL,
    STOP_RL,
    SIT_DOWN,
    EMERGENCY_STOP,
    RESET
};

// Motor command structure
struct MotorCommand {
    std::vector<float> q;      // Position
    std::vector<float> dq;     // Velocity
    std::vector<float> tau;    // Torque (feedforward)
    std::vector<float> kp;     // Position gain
    std::vector<float> kd;     // Velocity gain
    
    void resize(size_t n) {
        q.resize(n, 0.0f);
        dq.resize(n, 0.0f);
        tau.resize(n, 0.0f);
        kp.resize(n, 0.0f);
        kd.resize(n, 0.0f);
    }
};

// Robot command structure
struct RobotCommand {
    MotorCommand motor;
    
    void resize(size_t num_dof) {
        motor.resize(num_dof);
    }
};

// IMU state structure
struct IMUState {
    std::array<float, 4> quaternion = {1.0f, 0.0f, 0.0f, 0.0f}; // w, x, y, z
    std::array<float, 3> euler = {0.0f, 0.0f, 0.0f};            // roll, pitch, yaw
    std::array<float, 3> gyroscope = {0.0f, 0.0f, 0.0f};        // Angular velocity (rad/s)
    std::array<float, 3> accelerometer = {0.0f, 0.0f, 0.0f};    // Linear acceleration (m/s^2)
};

// Motor state structure
struct MotorState {
    std::vector<float> q;        // Position
    std::vector<float> dq;       // Velocity
    std::vector<float> ddq;      // Acceleration
    std::vector<float> tau_est;  // Estimated torque
    std::vector<float> cur;      // Current
    
    void resize(size_t n) {
        q.resize(n, 0.0f);
        dq.resize(n, 0.0f);
        ddq.resize(n, 0.0f);
        tau_est.resize(n, 0.0f);
        cur.resize(n, 0.0f);
    }
};

// Robot state structure
struct RobotState {
    std::array<float, 3> lin_vel_b = {0.0f, 0.0f, 0.0f};  // Linear velocity in base frame
    IMUState imu;
    MotorState motor;
    
    void resize(size_t num_dof) {
        motor.resize(num_dof);
    }
};

// Control command from user/joystick
struct Control {
    float x = 0.0f;    // Forward velocity command
    float y = 0.0f;    // Lateral velocity command
    float yaw = 0.0f;  // Yaw rate command
};

// Robot configuration
struct RobotConfig {
    std::string robot_name;
    int num_dof = 0;
    std::vector<std::string> joint_names;
    std::vector<std::string> dof_activated;  // Names of DOFs controlled by RL
    std::vector<int> dof_activated_indices;  // Resolved indices of DOFs controlled by RL
};

// Algorithm configuration
struct AlgorithmConfig {
    std::string name;
    std::string model_path;
    float dt = 0.0f;
    int decimation = 0;
    
    // Scaling factors
    float action_scale = 0.0f;
    float lin_vel_scale = 0.0f;
    float ang_vel_scale = 0.0f;
    float dof_pos_scale = 0.0f;
    float dof_vel_scale = 0.0f;
    float clip_obs = 0.0f;
    
    // Action limits
    float clip_actions_lower = 0.0f;
    float clip_actions_upper = 0.0f;
    // Command scaling
    std::array<float, 3> commands_scale = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> joystick_scale = {0.0f, 0.0f, 0.0f};
    
    // Raw YAML for algorithm-specific parameters
    YAML::Node yaml;
};

// Control configuration
struct ControlConfig {
    std::vector<float> rl_kp;
    std::vector<float> rl_kd;
    std::vector<float> fixed_kp;
    std::vector<float> fixed_kd;
    std::vector<float> default_dof_pos;
    
    // Transition parameters
    float stand_up_time = 2.5f;    // Time to stand up (seconds)
    float sit_down_time = 2.5f;    // Time to sit down (seconds)
};

// Complete configuration structure
struct Config {
    RobotConfig robot;
    AlgorithmConfig algorithm;
    ControlConfig control;
};

// Observation structure for algorithms
struct Observations {
    std::vector<float> lin_vel;
    std::vector<float> ang_vel;
    std::vector<float> gravity_proj;
    std::vector<float> base_euler;
    std::vector<float> base_quat;
    std::vector<float> commands;
    std::vector<float> dof_pos;
    std::vector<float> dof_vel;
    std::vector<float> actions;
    
    void resize(size_t num_dof_activated) {
        lin_vel.resize(3, 0.0f);
        ang_vel.resize(3, 0.0f);
        gravity_proj.resize(3, 0.0f);
        base_euler.resize(3, 0.0f);
        base_quat.resize(4, 0.0f);
        commands.resize(3, 0.0f);
        dof_pos.resize(num_dof_activated, 0.0f);
        dof_vel.resize(num_dof_activated, 0.0f);
        actions.resize(num_dof_activated, 0.0f);
    }
};

} // namespace bridge_core

#endif // BRIDGE_CORE_TYPES_HPP


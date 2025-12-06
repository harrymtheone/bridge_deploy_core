#include "bridge_core/core/config_manager.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

// Helper to require a parameter from YAML node
template <typename T>
T require(const YAML::Node &node, const std::string &key, const std::string &section) {
    if (!node[key]) {
        throw std::runtime_error("Missing required " + section + " parameter: " + key);
    }
    return node[key].as<T>();
}

// Helper to require a YAML sub-node exists
YAML::Node requireNode(const YAML::Node &node, const std::string &key, const std::string &section) {
    if (!node[key]) {
        throw std::runtime_error("Missing required " + section + " section: " + key);
    }
    return node[key];
}

// Helper to require a vector parameter with expected size
template <typename T>
std::vector<T> requireVector(const YAML::Node &node, const std::string &key,
                             const std::string &section, size_t expected_size) {
    if (!node[key]) {
        throw std::runtime_error("Missing required " + section + " parameter: " + key);
    }
    auto vec = node[key].as<std::vector<T>>();
    if (vec.size() != expected_size) {
        throw std::runtime_error(key + " must have exactly " + std::to_string(expected_size) +
                                 " values");
    }
    return vec;
}

namespace bridge_core {

Config ConfigManager::loadConfig(const std::string &config_path,
                                 const rclcpp::Node::SharedPtr &node) {
    if (!std::filesystem::exists(config_path)) {
        throw std::runtime_error("Config file not found: " + config_path);
    }

    YAML::Node yaml = YAML::LoadFile(config_path);
    Config config;

    // ==================== Robot Configuration ====================
    auto robot_node = requireNode(yaml, "robot", "config");

    // Required fields
    config.robot.robot_name = require<std::string>(robot_node, "type", "robot");
    config.robot.num_dof = require<int>(robot_node, "num_dof", "robot");

    // Validate
    if (config.robot.num_dof <= 0) {
        RCLCPP_ERROR(rclcpp::get_logger("ConfigManager"), "Invalid num_dof: %d",
                     config.robot.num_dof);
        throw std::runtime_error("Invalid num_dof: " + std::to_string(config.robot.num_dof));
    }

    auto joints = requireNode(robot_node, "joints", "robot");
    if (joints.size() != static_cast<size_t>(config.robot.num_dof)) {
        throw std::runtime_error("joints list size (" + std::to_string(joints.size()) +
                                 ") does not match num_dof (" +
                                 std::to_string(config.robot.num_dof) + ")");
    }

    for (const auto &joint : joints) {
        // Parse map entry: { name: "...", activated: bool }
        std::string name = require<std::string>(joint, "name", "joint");
        bool activated = require<bool>(joint, "activated", "joint");

        config.robot.joint_names.push_back(name);
        if (activated) {
            // Index is known immediately: size - 1 since we just pushed
            config.robot.dof_activated_ids.push_back(
                static_cast<int>(config.robot.joint_names.size() - 1));
        }
    }

    // ==================== Algorithm Configuration ====================
    auto alg_node = requireNode(yaml, "algorithm", "config");

    config.algorithm.name = require<std::string>(alg_node, "name", "algorithm");

    // Handle model path - relative to config file
    std::string model_name = require<std::string>(alg_node, "model_name", "algorithm");
    std::filesystem::path config_dir = std::filesystem::path(config_path).parent_path();
    std::filesystem::path model_path = config_dir / model_name;
    config.algorithm.model_path = model_path.string();

    // Required algorithm parameters
    config.algorithm.dt = require<float>(alg_node, "dt", "algorithm");
    config.algorithm.decimation = require<int>(alg_node, "decimation", "algorithm");

    config.algorithm.action_scale = require<float>(alg_node, "action_scale", "algorithm");
    config.algorithm.lin_vel_scale = require<float>(alg_node, "lin_vel_scale", "algorithm");
    config.algorithm.ang_vel_scale = require<float>(alg_node, "ang_vel_scale", "algorithm");
    config.algorithm.dof_pos_scale = require<float>(alg_node, "dof_pos_scale", "algorithm");
    config.algorithm.dof_vel_scale = require<float>(alg_node, "dof_vel_scale", "algorithm");
    config.algorithm.clip_obs = require<float>(alg_node, "clip_obs", "algorithm");

    config.algorithm.clip_actions_lower =
        require<float>(alg_node, "clip_actions_lower", "algorithm");
    config.algorithm.clip_actions_upper =
        require<float>(alg_node, "clip_actions_upper", "algorithm");

    // Load command scales (required)
    auto cmd_scale = requireVector<float>(alg_node, "commands_scale", "algorithm", 3);
    config.algorithm.commands_scale = {cmd_scale[0], cmd_scale[1], cmd_scale[2]};

    auto joy_scale = requireVector<float>(alg_node, "joystick_scale", "algorithm", 3);
    config.algorithm.joystick_scale = {joy_scale[0], joy_scale[1], joy_scale[2]};

    // Store algorithm YAML node for algorithm-specific parameters
    config.algorithm.yaml = alg_node;

    // Validate
    if (config.algorithm.name.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("ConfigManager"), "Algorithm name is empty");
        throw std::runtime_error("Algorithm name is empty");
    }

    if (!std::filesystem::exists(config.algorithm.model_path)) {
        RCLCPP_ERROR(rclcpp::get_logger("ConfigManager"), "Model file not found: %s",
                     config.algorithm.model_path.c_str());
        throw std::runtime_error("Model file not found: " + config.algorithm.model_path);
    }

    // ==================== Control Configuration ====================
    // Resize vectors to match number of joints
    size_t num_joints = config.robot.joint_names.size();
    config.control.rl_kp.resize(num_joints);
    config.control.rl_kd.resize(num_joints);
    config.control.fixed_kp.resize(num_joints);
    config.control.fixed_kd.resize(num_joints);
    config.control.default_dof_pos.resize(num_joints);

    // Populate control vectors (joints are already in order)
    for (size_t i = 0; i < num_joints; ++i) {
        const auto &joint = joints[i];
        config.control.rl_kp[i] = joint["rl_kp"].as<float>();
        config.control.rl_kd[i] = joint["rl_kd"].as<float>();
        config.control.fixed_kp[i] = joint["control_kp"].as<float>();
        config.control.fixed_kd[i] = joint["control_kd"].as<float>();
        config.control.default_dof_pos[i] = joint["default_pos"].as<float>();
    }

    // Transition times
    auto ctrl_node = requireNode(yaml, "control", "config");
    config.control.stand_up_time = require<float>(ctrl_node, "stand_up_time", "control");
    config.control.sit_down_time = require<float>(ctrl_node, "sit_down_time", "control");

    // ==================== Safety Configuration ====================
    auto safety_node = requireNode(yaml, "safety", "config");

    config.safety.max_lean_angle_deg = require<float>(safety_node, "max_lean_angle_deg", "safety");
    config.safety.torque_limit_scale = require<float>(safety_node, "torque_limit_scale", "safety");

    // Resize torque limit vector
    config.safety.torque_limit.resize(num_joints);

    // Populate safety vectors (joints are already in order)
    for (size_t i = 0; i < num_joints; ++i) {
        config.safety.torque_limit[i] = joints[i]["torque_limit"].as<float>();
    }

    // Apply ROS parameter overrides if node provided
    if (node) {
        applyParameterOverrides(config, node);
    }

    return config;
}

void ConfigManager::applyParameterOverrides(Config &config, const rclcpp::Node::SharedPtr &node) {
    // Allow ROS parameters to override configuration
    std::string model_path;
    if (node->get_parameter("model_path", model_path)) {
        config.algorithm.model_path = model_path;
    }
}

} // namespace bridge_core

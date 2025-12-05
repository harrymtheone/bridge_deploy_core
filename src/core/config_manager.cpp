#include "bridge_core/core/config_manager.hpp"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace bridge_core {

namespace {

// Helper to require a float parameter from YAML node
float requireFloat(const YAML::Node& node, const std::string& key, const std::string& section) {
    if (!node[key]) {
        throw std::runtime_error("Missing required " + section + " parameter: " + key);
    }
    return node[key].as<float>();
}

// Helper to require an int parameter from YAML node
int requireInt(const YAML::Node& node, const std::string& key, const std::string& section) {
    if (!node[key]) {
        throw std::runtime_error("Missing required " + section + " parameter: " + key);
    }
    return node[key].as<int>();
}

// Helper to require a string parameter from YAML node
std::string requireString(const YAML::Node& node, const std::string& key, const std::string& section) {
    if (!node[key]) {
        throw std::runtime_error("Missing required " + section + " parameter: " + key);
    }
    return node[key].as<std::string>();
}

} // anonymous namespace

Config ConfigManager::loadConfig(const std::string& config_path,
                                 const rclcpp::Node::SharedPtr& node) {
    if (!std::filesystem::exists(config_path)) {
        throw std::runtime_error("Config file not found: " + config_path);
    }

    Config config;
    config.robot = loadRobotConfig(config_path);
    config.algorithm = loadAlgorithmConfig(config_path);
    config.control = loadControlConfig(config_path, config.robot.joint_names, config.robot.dof_activated);
    config.safety = loadSafetyConfig(config_path, config.robot.joint_names);

    // Apply ROS parameter overrides if node provided
    if (node) {
        applyParameterOverrides(config, node);
    }

    if (!validateConfig(config)) {
        throw std::runtime_error("Invalid configuration in: " + config_path);
    }

    return config;
}

RobotConfig ConfigManager::loadRobotConfig(const std::string& config_path) {
    YAML::Node yaml = YAML::LoadFile(config_path);
    RobotConfig config;

    if (!yaml["robot"]) {
        throw std::runtime_error("Missing 'robot' section in config: " + config_path);
    }

    auto robot_node = yaml["robot"];
    
    // Required fields
    config.robot_name = requireString(robot_node, "type", "robot");
    config.num_dof = requireInt(robot_node, "num_dof", "robot");

    if (!robot_node["joint_names"]) {
        throw std::runtime_error("Missing required robot parameter: joint_names");
    }
    config.joint_names = robot_node["joint_names"].as<std::vector<std::string>>();

    if (!robot_node["dof_activated"]) {
        throw std::runtime_error("Missing required robot parameter: dof_activated");
    }
    config.dof_activated = robot_node["dof_activated"].as<std::vector<std::string>>();
    
    // Resolve joint names to indices
    for (const auto& joint_name : config.dof_activated) {
        auto it = std::find(config.joint_names.begin(), config.joint_names.end(), joint_name);
        if (it != config.joint_names.end()) {
            config.dof_activated_indices.push_back(
                static_cast<int>(std::distance(config.joint_names.begin(), it)));
        } else {
            throw std::runtime_error("dof_activated joint not found in joint_names: " + joint_name);
        }
    }

    return config;
}

AlgorithmConfig ConfigManager::loadAlgorithmConfig(const std::string& config_path) {
    YAML::Node yaml = YAML::LoadFile(config_path);
    AlgorithmConfig config;

    if (!yaml["algorithm"] || !yaml["algorithm"].IsMap()) {
        throw std::runtime_error("Missing or invalid 'algorithm' section in config: " + config_path);
    }

    auto alg_node = yaml["algorithm"];
    
    config.name = requireString(alg_node, "name", "algorithm");

    // Handle model path - can be relative to config file (one of model_name or model_path required)
    std::string model_name = alg_node["model_name"].as<std::string>("");
    if (!model_name.empty()) {
        std::filesystem::path config_dir = std::filesystem::path(config_path).parent_path();
        std::filesystem::path model_path = config_dir / model_name;
        config.model_path = model_path.string();
    } else if (alg_node["model_path"]) {
        config.model_path = alg_node["model_path"].as<std::string>();
    } else {
        throw std::runtime_error("Missing required algorithm parameter: model_name or model_path");
    }

    // Required algorithm parameters
    config.dt = requireFloat(alg_node, "dt", "algorithm");
    config.decimation = requireInt(alg_node, "decimation", "algorithm");

    config.action_scale = requireFloat(alg_node, "action_scale", "algorithm");
    config.lin_vel_scale = requireFloat(alg_node, "lin_vel_scale", "algorithm");
    config.ang_vel_scale = requireFloat(alg_node, "ang_vel_scale", "algorithm");
    config.dof_pos_scale = requireFloat(alg_node, "dof_pos_scale", "algorithm");
    config.dof_vel_scale = requireFloat(alg_node, "dof_vel_scale", "algorithm");
    config.clip_obs = requireFloat(alg_node, "clip_obs", "algorithm");

    config.clip_actions_lower = requireFloat(alg_node, "clip_actions_lower", "algorithm");
    config.clip_actions_upper = requireFloat(alg_node, "clip_actions_upper", "algorithm");

    // Load command scales (required)
    if (!alg_node["commands_scale"]) {
        throw std::runtime_error("Missing required algorithm parameter: commands_scale");
    }
    auto cmd_scale = alg_node["commands_scale"].as<std::vector<float>>();
    if (cmd_scale.size() < 3) {
        throw std::runtime_error("commands_scale must have at least 3 values");
    }
    config.commands_scale = {cmd_scale[0], cmd_scale[1], cmd_scale[2]};

    if (!alg_node["joystick_scale"]) {
        throw std::runtime_error("Missing required algorithm parameter: joystick_scale");
    }
    auto joy_scale = alg_node["joystick_scale"].as<std::vector<float>>();
    if (joy_scale.size() < 3) {
        throw std::runtime_error("joystick_scale must have at least 3 values");
    }
    config.joystick_scale = {joy_scale[0], joy_scale[1], joy_scale[2]};

    // Store algorithm YAML node for algorithm-specific parameters
    config.yaml = alg_node;

    return config;
}

ControlConfig ConfigManager::loadControlConfig(
    const std::string& config_path,
    const std::vector<std::string>& joint_names,
    const std::vector<std::string>& dof_activated) {
    YAML::Node yaml = YAML::LoadFile(config_path);
    ControlConfig config;

    if (!yaml["control"]) {
        throw std::runtime_error("Missing 'control' section in config: " + config_path);
    }

    auto ctrl_node = yaml["control"];

    // Helper to parse a map of joint_name -> value into a vector ordered by joint_names
    auto parseJointMap = [&](const YAML::Node& node, const std::vector<std::string>& names) 
        -> std::vector<float> {
        std::vector<float> result(names.size(), 0.0f);
        if (node && node.IsMap()) {
            for (const auto& pair : node) {
                std::string joint_name = pair.first.as<std::string>();
                float value = pair.second.as<float>();
                auto it = std::find(names.begin(), names.end(), joint_name);
                if (it != names.end()) {
                    size_t idx = static_cast<size_t>(std::distance(names.begin(), it));
                    result[idx] = value;
                } else {
                    throw std::runtime_error("Joint not found in joint list: " + joint_name);
                }
            }
        }
        return result;
    };

    // Helper to require a joint map field
    auto requireJointMap = [&](const std::string& key) -> std::vector<float> {
        if (!ctrl_node[key]) {
            throw std::runtime_error("Missing required control parameter: " + key);
        }
        return parseJointMap(ctrl_node[key], joint_names);
    };

    // Parse RL gains (ordered by joint_names - all joints) - required
    config.rl_kp = requireJointMap("rl_kp");
    config.rl_kd = requireJointMap("rl_kd");
    
    // Parse fixed gains and default positions (ordered by joint_names) - required
    config.fixed_kp = requireJointMap("fixed_kp");
    config.fixed_kd = requireJointMap("fixed_kd");
    config.default_dof_pos = requireJointMap("default_dof_pos");

    // Optional with defaults
    config.stand_up_time = ctrl_node["stand_up_time"].as<float>(2.5f);
    config.sit_down_time = ctrl_node["sit_down_time"].as<float>(2.5f);

    return config;
}

SafetyConfig ConfigManager::loadSafetyConfig(const std::string& config_path,
                                             const std::vector<std::string>& joint_names) {
    YAML::Node yaml = YAML::LoadFile(config_path);
    SafetyConfig config;

    if (!yaml["safety"]) {
        throw std::runtime_error("Missing 'safety' section in config: " + config_path);
    }

    auto safety_node = yaml["safety"];
    
    config.max_lean_angle_deg = requireFloat(safety_node, "max_lean_angle_deg", "safety");
    config.torque_limit_scale = requireFloat(safety_node, "torque_limit_scale", "safety");

    // Parse torque limits (ordered by joint_names)
    if (!safety_node["torque_limit"]) {
        throw std::runtime_error("Missing required safety parameter: torque_limit");
    }
    
    // Helper to parse a map of joint_name -> value into a vector ordered by joint_names
    auto parseJointMap = [&](const YAML::Node& node) -> std::vector<float> {
        std::vector<float> result(joint_names.size(), 0.0f);
        if (node && node.IsMap()) {
            for (const auto& pair : node) {
                std::string joint_name = pair.first.as<std::string>();
                float value = pair.second.as<float>();
                auto it = std::find(joint_names.begin(), joint_names.end(), joint_name);
                if (it != joint_names.end()) {
                    size_t idx = static_cast<size_t>(std::distance(joint_names.begin(), it));
                    result[idx] = value;
                } else {
                    throw std::runtime_error("Joint not found in joint list: " + joint_name);
                }
            }
        }
        return result;
    };
    
    config.torque_limit = parseJointMap(safety_node["torque_limit"]);

    return config;
}

void ConfigManager::applyParameterOverrides(Config& config, const rclcpp::Node::SharedPtr& node) {
    // Allow ROS parameters to override configuration
    std::string model_path;
    if (node->get_parameter("model_path", model_path)) {
        config.algorithm.model_path = model_path;
    }
}

bool ConfigManager::validateConfig(const Config& config) {
    // Validate robot config
    if (config.robot.num_dof <= 0) {
        RCLCPP_ERROR(rclcpp::get_logger("ConfigManager"), 
                     "Invalid num_dof: %d", config.robot.num_dof);
        return false;
    }

    // Validate algorithm config
    if (config.algorithm.name.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("ConfigManager"), 
                     "Algorithm name is empty");
        return false;
    }

    if (config.algorithm.model_path.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("ConfigManager"), 
                     "Model path is empty");
        return false;
    }

    if (!std::filesystem::exists(config.algorithm.model_path)) {
        RCLCPP_ERROR(rclcpp::get_logger("ConfigManager"), 
                     "Model file not found: %s", config.algorithm.model_path.c_str());
        return false;
    }

    // Validate control config - all arrays must match num_dof
    size_t num_dof = static_cast<size_t>(config.robot.num_dof);
    
    if (config.control.default_dof_pos.size() != num_dof) {
        RCLCPP_ERROR(rclcpp::get_logger("ConfigManager"),
                     "default_dof_pos size (%zu) doesn't match num_dof (%d)",
                     config.control.default_dof_pos.size(), config.robot.num_dof);
        return false;
    }
    
    if (config.control.fixed_kp.size() != num_dof) {
        RCLCPP_ERROR(rclcpp::get_logger("ConfigManager"),
                     "fixed_kp size (%zu) doesn't match num_dof (%d)",
                     config.control.fixed_kp.size(), config.robot.num_dof);
        return false;
    }
    
    if (config.control.fixed_kd.size() != num_dof) {
        RCLCPP_ERROR(rclcpp::get_logger("ConfigManager"),
                     "fixed_kd size (%zu) doesn't match num_dof (%d)",
                     config.control.fixed_kd.size(), config.robot.num_dof);
        return false;
    }
    
    // Validate joint_names matches num_dof
    if (!config.robot.joint_names.empty() && 
        config.robot.joint_names.size() != num_dof) {
        RCLCPP_ERROR(rclcpp::get_logger("ConfigManager"),
                     "joint_names size (%zu) doesn't match num_dof (%d)",
                     config.robot.joint_names.size(), config.robot.num_dof);
        return false;
    }

    return true;
}

std::string ConfigManager::findConfigFile(const std::string& model_name,
                                          const std::vector<std::string>& search_paths) {
    std::string filename = model_name + ".yaml";

    for (const auto& search_path : search_paths) {
        std::filesystem::path full_path = std::filesystem::path(search_path) / filename;
        if (std::filesystem::exists(full_path)) {
            return full_path.string();
        }
        
        // Also try config.yaml in a subdirectory named after the model
        std::filesystem::path subdir_path = std::filesystem::path(search_path) / model_name / "config.yaml";
        if (std::filesystem::exists(subdir_path)) {
            return subdir_path.string();
        }
    }

    throw std::runtime_error("Config file not found for model: " + model_name);
}

} // namespace bridge_core


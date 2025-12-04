#include "bridge_core/core/config_manager.hpp"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace bridge_core {

Config ConfigManager::loadConfig(const std::string& config_path,
                                 const rclcpp::Node::SharedPtr& node) {
    if (!std::filesystem::exists(config_path)) {
        throw std::runtime_error("Config file not found: " + config_path);
    }

    Config config;
    config.robot = loadRobotConfig(config_path);
    config.algorithm = loadAlgorithmConfig(config_path);
    config.control = loadControlConfig(config_path, config.robot.joint_names, config.robot.dof_activated);

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
    config.robot_name = robot_node["type"].as<std::string>("unknown");
    config.num_dof = robot_node["num_dof"].as<int>(0);

    if (robot_node["joint_names"]) {
        config.joint_names = robot_node["joint_names"].as<std::vector<std::string>>();
    }

    if (robot_node["dof_activated"]) {
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
    config.name = alg_node["name"].as<std::string>("");

    // Handle model path - can be relative to config file
    std::string model_name = alg_node["model_name"].as<std::string>("");
    if (!model_name.empty()) {
        std::filesystem::path config_dir = std::filesystem::path(config_path).parent_path();
        std::filesystem::path model_path = config_dir / model_name;
        config.model_path = model_path.string();
    } else {
        config.model_path = alg_node["model_path"].as<std::string>("");
    }

    // Required algorithm parameters - throw if missing
    auto requireFloat = [&](const std::string& key) -> float {
        if (!alg_node[key]) {
            throw std::runtime_error("Missing required algorithm parameter: " + key);
        }
        return alg_node[key].as<float>();
    };
    
    auto requireInt = [&](const std::string& key) -> int {
        if (!alg_node[key]) {
            throw std::runtime_error("Missing required algorithm parameter: " + key);
        }
        return alg_node[key].as<int>();
    };

    config.dt = requireFloat("dt");
    config.decimation = requireInt("decimation");

    config.action_scale = requireFloat("action_scale");
    config.lin_vel_scale = requireFloat("lin_vel_scale");
    config.ang_vel_scale = requireFloat("ang_vel_scale");
    config.dof_pos_scale = requireFloat("dof_pos_scale");
    config.dof_vel_scale = requireFloat("dof_vel_scale");
    config.clip_obs = requireFloat("clip_obs");

    config.clip_actions_lower = requireFloat("clip_actions_lower");
    config.clip_actions_upper = requireFloat("clip_actions_upper");

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

    // Parse RL gains (ordered by joint_names - all joints)
    if (ctrl_node["rl_kp"]) {
        config.rl_kp = parseJointMap(ctrl_node["rl_kp"], joint_names);
    }
    if (ctrl_node["rl_kd"]) {
        config.rl_kd = parseJointMap(ctrl_node["rl_kd"], joint_names);
    }
    
    // Parse fixed gains and default positions (ordered by joint_names)
    if (ctrl_node["fixed_kp"]) {
        config.fixed_kp = parseJointMap(ctrl_node["fixed_kp"], joint_names);
    }
    if (ctrl_node["fixed_kd"]) {
        config.fixed_kd = parseJointMap(ctrl_node["fixed_kd"], joint_names);
    }
    if (ctrl_node["default_dof_pos"]) {
        config.default_dof_pos = parseJointMap(ctrl_node["default_dof_pos"], joint_names);
    }

    config.stand_up_time = ctrl_node["stand_up_time"].as<float>(2.5f);
    config.sit_down_time = ctrl_node["sit_down_time"].as<float>(2.5f);

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


#include "bridge_core/core/config_manager.hpp"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <stdexcept>

// Helper to require a parameter from YAML node
template <typename T>
T require(const YAML::Node &node, const std::string &key, const std::string &section)
{
    if (!node[key])
    {
        throw std::runtime_error("Missing required " + section + " parameter: " + key);
    }
    return node[key].as<T>();
}

// Helper to require a YAML sub-node exists
YAML::Node requireNode(const YAML::Node &node, const std::string &key, const std::string &section)
{
    if (!node[key])
    {
        throw std::runtime_error("Missing required " + section + " section: " + key);
    }
    return node[key];
}

// Helper to require a vector parameter with expected size
template <typename T>
std::vector<T> requireVector(const YAML::Node &node, const std::string &key, const std::string &section, size_t expected_size)
{
    if (!node[key])
    {
        throw std::runtime_error("Missing required " + section + " parameter: " + key);
    }
    auto vec = node[key].as<std::vector<T>>();
    if (vec.size() != expected_size)
    {
        throw std::runtime_error(key + " must have exactly " + std::to_string(expected_size) + " values");
    }
    return vec;
}

// Helper to parse a YAML map of joint_name -> value into a vector ordered by joint_names
std::vector<float> parseJointMap(const YAML::Node &node, const std::vector<std::string> &joint_names)
{
    std::vector<float> result(joint_names.size(), 0.0f);
    if (node && node.IsMap())
    {
        for (const auto &pair : node)
        {
            std::string joint_name = pair.first.as<std::string>();
            float value = pair.second.as<float>();
            auto it = std::find(joint_names.begin(), joint_names.end(), joint_name);
            if (it != joint_names.end())
            {
                size_t idx = static_cast<size_t>(std::distance(joint_names.begin(), it));
                result[idx] = value;
            }
            else
            {
                throw std::runtime_error("Joint not found in joint list: " + joint_name);
            }
        }
    }
    return result;
}

// Helper to require a joint map field
std::vector<float> requireJointMap(const YAML::Node &node, const std::string &key, const std::string &section, const std::vector<std::string> &joint_names)
{
    if (!node[key])
    {
        throw std::runtime_error("Missing required " + section + " parameter: " + key);
    }
    return parseJointMap(node[key], joint_names);
}

namespace bridge_core
{

    Config ConfigManager::loadConfig(const std::string &config_path,
                                     const rclcpp::Node::SharedPtr &node)
    {
        if (!std::filesystem::exists(config_path))
        {
            throw std::runtime_error("Config file not found: " + config_path);
        }

        Config config;
        config.robot = loadRobotConfig(config_path);
        config.algorithm = loadAlgorithmConfig(config_path);
        config.control = loadControlConfig(config_path, config.robot.joint_names, config.robot.dof_activated);
        config.safety = loadSafetyConfig(config_path, config.robot.joint_names);

        // Apply ROS parameter overrides if node provided
        if (node)
        {
            applyParameterOverrides(config, node);
        }

        if (!validateConfig(config))
        {
            throw std::runtime_error("Invalid configuration in: " + config_path);
        }

        return config;
    }

    RobotConfig ConfigManager::loadRobotConfig(const std::string &config_path)
    {
        YAML::Node yaml = YAML::LoadFile(config_path);
        RobotConfig config;

        auto robot_node = requireNode(yaml, "robot", "config");

        // Required fields
        config.robot_name = require<std::string>(robot_node, "type", "robot");
        config.num_dof = require<int>(robot_node, "num_dof", "robot");

        config.joint_names = requireVector<std::string>(robot_node, "joint_names", "robot", static_cast<size_t>(config.num_dof));
        config.dof_activated = requireNode(robot_node, "dof_activated", "robot").as<std::vector<std::string>>();

        // Resolve joint names to indices
        for (const auto &joint_name : config.dof_activated)
        {
            auto it = std::find(config.joint_names.begin(), config.joint_names.end(), joint_name);
            if (it != config.joint_names.end())
            {
                config.dof_activated_indices.push_back(
                    static_cast<int>(std::distance(config.joint_names.begin(), it)));
            }
            else
            {
                throw std::runtime_error("dof_activated joint not found in joint_names: " + joint_name);
            }
        }

        return config;
    }

    AlgorithmConfig ConfigManager::loadAlgorithmConfig(const std::string &config_path)
    {
        YAML::Node yaml = YAML::LoadFile(config_path);
        AlgorithmConfig config;

        auto alg_node = requireNode(yaml, "algorithm", "config");

        config.name = require<std::string>(alg_node, "name", "algorithm");

        // Handle model path - can be relative to config file (one of model_name or model_path required)
        std::string model_name = alg_node["model_name"].as<std::string>("");
        if (model_name.empty())
        {
            throw std::runtime_error("Missing required algorithm parameter: model_name or model_path");
        }
        else
        {
            std::filesystem::path config_dir = std::filesystem::path(config_path).parent_path();
            std::filesystem::path model_path = config_dir / model_name;
            config.model_path = model_path.string();
        }

        // Required algorithm parameters
        config.dt = require<float>(alg_node, "dt", "algorithm");
        config.decimation = require<int>(alg_node, "decimation", "algorithm");

        config.action_scale = require<float>(alg_node, "action_scale", "algorithm");
        config.lin_vel_scale = require<float>(alg_node, "lin_vel_scale", "algorithm");
        config.ang_vel_scale = require<float>(alg_node, "ang_vel_scale", "algorithm");
        config.dof_pos_scale = require<float>(alg_node, "dof_pos_scale", "algorithm");
        config.dof_vel_scale = require<float>(alg_node, "dof_vel_scale", "algorithm");
        config.clip_obs = require<float>(alg_node, "clip_obs", "algorithm");

        config.clip_actions_lower = require<float>(alg_node, "clip_actions_lower", "algorithm");
        config.clip_actions_upper = require<float>(alg_node, "clip_actions_upper", "algorithm");

        // Load command scales (required)
        auto cmd_scale = requireVector<float>(alg_node, "commands_scale", "algorithm", 3);
        config.commands_scale = {cmd_scale[0], cmd_scale[1], cmd_scale[2]};

        auto joy_scale = requireVector<float>(alg_node, "joystick_scale", "algorithm", 3);
        config.joystick_scale = {joy_scale[0], joy_scale[1], joy_scale[2]};

        // Store algorithm YAML node for algorithm-specific parameters
        config.yaml = alg_node;

        return config;
    }

    ControlConfig ConfigManager::loadControlConfig(
        const std::string &config_path,
        const std::vector<std::string> &joint_names,
        const std::vector<std::string> &dof_activated)
    {
        YAML::Node yaml = YAML::LoadFile(config_path);
        ControlConfig config;

        auto ctrl_node = requireNode(yaml, "control", "config");

        // Parse RL gains (ordered by joint_names - all joints) - required
        config.rl_kp = requireJointMap(ctrl_node, "rl_kp", "control", joint_names);
        config.rl_kd = requireJointMap(ctrl_node, "rl_kd", "control", joint_names);

        // Parse fixed gains and default positions (ordered by joint_names) - required
        config.fixed_kp = requireJointMap(ctrl_node, "fixed_kp", "control", joint_names);
        config.fixed_kd = requireJointMap(ctrl_node, "fixed_kd", "control", joint_names);
        config.default_dof_pos = requireJointMap(ctrl_node, "default_dof_pos", "control", joint_names);

        // Optional with defaults
        config.stand_up_time = ctrl_node["stand_up_time"].as<float>(2.5f);
        config.sit_down_time = ctrl_node["sit_down_time"].as<float>(2.5f);

        return config;
    }

    SafetyConfig ConfigManager::loadSafetyConfig(const std::string &config_path,
                                                 const std::vector<std::string> &joint_names)
    {
        YAML::Node yaml = YAML::LoadFile(config_path);
        SafetyConfig config;

        auto safety_node = requireNode(yaml, "safety", "config");

        config.max_lean_angle_deg = require<float>(safety_node, "max_lean_angle_deg", "safety");
        config.torque_limit_scale = require<float>(safety_node, "torque_limit_scale", "safety");
        config.torque_limit = requireJointMap(safety_node, "torque_limit", "safety", joint_names);

        return config;
    }

    void ConfigManager::applyParameterOverrides(Config &config, const rclcpp::Node::SharedPtr &node)
    {
        // Allow ROS parameters to override configuration
        std::string model_path;
        if (node->get_parameter("model_path", model_path))
        {
            config.algorithm.model_path = model_path;
        }
    }

    bool ConfigManager::validateConfig(const Config &config)
    {
        // Validate robot config
        if (config.robot.num_dof <= 0)
        {
            RCLCPP_ERROR(rclcpp::get_logger("ConfigManager"),
                         "Invalid num_dof: %d", config.robot.num_dof);
            return false;
        }

        // Validate algorithm config
        if (config.algorithm.name.empty())
        {
            RCLCPP_ERROR(rclcpp::get_logger("ConfigManager"),
                         "Algorithm name is empty");
            return false;
        }

        if (!std::filesystem::exists(config.algorithm.model_path))
        {
            RCLCPP_ERROR(rclcpp::get_logger("ConfigManager"),
                         "Model file not found: %s", config.algorithm.model_path.c_str());
            return false;
        }

        return true;
    }

} // namespace bridge_core

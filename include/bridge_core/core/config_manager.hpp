#ifndef BRIDGE_CORE_CONFIG_MANAGER_HPP
#define BRIDGE_CORE_CONFIG_MANAGER_HPP

#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include "bridge_core/core/types.hpp"

namespace bridge_core {

/**
 * @brief Manages configuration loading and validation
 * 
 * Loads YAML configuration files and validates their contents.
 */
class ConfigManager {
public:
    /**
     * @brief Load configuration from file
     * @param config_path Path to YAML configuration file
     * @param node ROS2 node for parameter overrides (optional)
     * @return Complete configuration structure
     * @throws std::runtime_error if config is invalid or file not found
     */
    static Config loadConfig(const std::string& config_path,
                            const rclcpp::Node::SharedPtr& node = nullptr);
    
    /**
     * @brief Validate configuration completeness
     * @param config Configuration to validate
     * @return true if valid
     */
    static bool validateConfig(const Config& config);
    
    /**
     * @brief Find config file by model name in search paths
     * @param model_name Model name (e.g., "t1_mod")
     * @param search_paths Directories to search in
     * @return Full path to config file
     * @throws std::runtime_error if not found
     */
    static std::string findConfigFile(const std::string& model_name,
                                     const std::vector<std::string>& search_paths);

private:
    static RobotConfig loadRobotConfig(const std::string& config_path);
    static AlgorithmConfig loadAlgorithmConfig(const std::string& config_path);
    static ControlConfig loadControlConfig(const std::string& config_path);
    static void applyParameterOverrides(Config& config, const rclcpp::Node::SharedPtr& node);
};

} // namespace bridge_core

#endif // BRIDGE_CORE_CONFIG_MANAGER_HPP


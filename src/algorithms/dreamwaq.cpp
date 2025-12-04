#include "bridge_core/algorithms/dreamwaq.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace bridge_core {

void DreamWAQ::initObservations() {
    AlgorithmBase::initObservations();
    
    // Load DreamWAQ-specific parameters from algorithm YAML node
    auto requireParam = [&](const std::string& key) {
        if (!config_.yaml[key]) {
            throw std::runtime_error("Missing required DreamWAQ parameter: " + key);
        }
    };
    
    requireParam("gait_cycle_time");
    requireParam("sw_switch");
    requireParam("sw_lin_cmd_thresh");
    requireParam("sw_yaw_cmd_thresh");
    
    gait_cycle_time_ = config_.yaml["gait_cycle_time"].as<float>();
    sw_switch_ = config_.yaml["sw_switch"].as<bool>();
    lin_cmd_thresh_ = config_.yaml["sw_lin_cmd_thresh"].as<float>();
    yaw_cmd_thresh_ = config_.yaml["sw_yaw_cmd_thresh"].as<float>();
    
    RCLCPP_INFO(node_->get_logger(), "DreamWAQ params: cycle_time=%.2f, sw_switch=%s, lin_thresh=%.2f, yaw_thresh=%.2f",
                gait_cycle_time_, sw_switch_ ? "true" : "false", lin_cmd_thresh_, yaw_cmd_thresh_);
    
    // Proprio size: ang_vel(3) + gravity(3) + clock(2) + commands(3) + dof_pos(n) + dof_vel(n) + actions(n)
    // For G1 with 15 activated DOFs: 3 + 3 + 2 + 3 + 15 + 15 + 15 = 56
    size_t n_proprio = 3 + 3 + 2 + 3 + 3 * static_cast<size_t>(num_dof_activated_);
    proprio_.resize(n_proprio, 0.0f);
    
    // Initialize history buffer with zeros
    prop_his_buffer_.clear();
    for (size_t i = 0; i < kHistoryLen; ++i) {
        prop_his_buffer_.push_back(std::vector<float>(n_proprio, 0.0f));
    }
    
    // Initialize flat history buffer for ONNX input
    // Shape: {1, history_len, proprio_size}
    prop_his_flat_.resize(kHistoryLen * n_proprio, 0.0f);
    
    RCLCPP_INFO(node_->get_logger(), "DreamWAQ initialized with %zu proprio dims, %zu history length",
                n_proprio, kHistoryLen);
}

void DreamWAQ::reset() {
    AlgorithmBase::reset();
    
    // Reset history buffer to zeros
    for (auto& hist : prop_his_buffer_) {
        std::fill(hist.begin(), hist.end(), 0.0f);
    }
    std::fill(prop_his_flat_.begin(), prop_his_flat_.end(), 0.0f);
    
    // Reset gait phase
    time_initialized_ = false;
    
    RCLCPP_DEBUG(node_->get_logger(), "DreamWAQ: Reset history buffer and phase");
}

std::pair<float, float> DreamWAQ::computeClockSignals() {
    auto now = std::chrono::steady_clock::now();
    
    // Check for zero command (stand-walk switch)
    // Commands are already scaled, so compare against scaled thresholds
    float lin_cmd_norm = std::sqrt(obs_.commands[0] * obs_.commands[0] + 
                                   obs_.commands[1] * obs_.commands[1]);
    float yaw_cmd = std::abs(obs_.commands[2]);
    bool is_zero_command = (lin_cmd_norm < lin_cmd_thresh_) && (yaw_cmd < yaw_cmd_thresh_);
    
    // If stand-walk switch is enabled and command is zero, keep phase at zero
    if (sw_switch_ && is_zero_command) {
        // Reset start time to now, so phase stays at zero
        start_time_ = now;
        time_initialized_ = true;
        // Return sin(0)=0, cos(0)=1 for standing pose
        return {0.0f, 1.0f};
    }
    
    // Initialize start time on first non-zero command
    if (!time_initialized_) {
        start_time_ = now;
        time_initialized_ = true;
    }
    
    // Compute phase based on absolute elapsed time (immune to inference timing jitter)
    float elapsed_sec = std::chrono::duration<float>(now - start_time_).count();
    float phase = elapsed_sec * 2.0f * static_cast<float>(M_PI) / gait_cycle_time_;
    
    // Return sin and cos of phase
    return {std::sin(phase), std::cos(phase)};
}

void DreamWAQ::updateHistory(const std::vector<float>& proprio) {
    // Add current proprio to the front of the buffer
    prop_his_buffer_.push_front(proprio);
    
    // Remove oldest entry if buffer exceeds history length
    if (prop_his_buffer_.size() > kHistoryLen) {
        prop_his_buffer_.pop_back();
    }
    
    // Flatten history buffer for ONNX input
    // Shape: {history_len, proprio_size} -> flat vector
    size_t idx = 0;
    for (const auto& hist : prop_his_buffer_) {
        for (size_t j = 0; j < hist.size(); ++j) {
            prop_his_flat_[idx++] = hist[j];
        }
    }
    
    // Pad with zeros if buffer is not full yet
    while (idx < prop_his_flat_.size()) {
        prop_his_flat_[idx++] = 0.0f;
    }
}

std::vector<float> DreamWAQ::forward() {
    // Compute clock signals
    auto [clock_sin, clock_cos] = computeClockSignals();
    
    // Build proprio observation vector
    size_t idx = 0;
    
    // Angular velocity (3) - scaled
    for (size_t i = 0; i < obs_.ang_vel.size(); ++i) {
        proprio_[idx++] = obs_.ang_vel[i];
    }
    
    // Gravity projection (3)
    for (size_t i = 0; i < obs_.gravity_proj.size(); ++i) {
        proprio_[idx++] = obs_.gravity_proj[i];
    }
    
    // Clock signals (2) - sin and cos for gait timing
    proprio_[idx++] = clock_sin;
    proprio_[idx++] = clock_cos;
    
    // Commands (3) - scaled
    for (size_t i = 0; i < obs_.commands.size(); ++i) {
        proprio_[idx++] = obs_.commands[i];
    }
    
    // DOF positions (n) - scaled
    for (size_t i = 0; i < obs_.dof_pos.size(); ++i) {
        proprio_[idx++] = obs_.dof_pos[i];
    }
    
    // DOF velocities (n) - scaled
    for (size_t i = 0; i < obs_.dof_vel.size(); ++i) {
        proprio_[idx++] = obs_.dof_vel[i];
    }
    
    // Previous actions (n)
    for (size_t i = 0; i < obs_.actions.size(); ++i) {
        proprio_[idx++] = obs_.actions[i];
    }
    
    // Clip observations
    for (size_t i = 0; i < proprio_.size(); ++i) {
        proprio_[i] = std::clamp(proprio_[i], -config_.clip_obs, config_.clip_obs);
    }
    
    // Update history buffer with current proprio (before clipping for inference)
    updateHistory(proprio_);
    
    // Prepare ONNX input tensors
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Input 1: proprio - shape {1, proprio_size}
    std::vector<int64_t> proprio_shape = {kBatchSize, static_cast<int64_t>(proprio_.size())};
    
    // Input 2: prop_his - shape {1, history_len, proprio_size}
    std::vector<int64_t> prop_his_shape = {
        kBatchSize, 
        static_cast<int64_t>(kHistoryLen), 
        static_cast<int64_t>(proprio_.size())
    };
    
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info,
        proprio_.data(),
        proprio_.size(),
        proprio_shape.data(),
        proprio_shape.size()
    ));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info,
        prop_his_flat_.data(),
        prop_his_flat_.size(),
        prop_his_shape.data(),
        prop_his_shape.size()
    ));
    
    // Run inference
    auto output_tensors = ort_session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names_.data(),
        output_names_.size()
    );
    
    // Extract actions (output 0: "actions")
    float* actions_data = output_tensors[0].GetTensorMutableData<float>();
    auto actions_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = static_cast<size_t>(actions_shape[1]);  // Shape is [1, n]
    
    // Copy output to actions
    obs_.actions.resize(output_size);
    std::copy(actions_data, actions_data + output_size, obs_.actions.begin());
    
    // Output 1: "recon" (scan reconstruction) - optional, for debugging
    // Shape: [1, 32, 16] or [-1, 32, 16]
    if (output_tensors.size() > 1) {
        // Could log or visualize the reconstructed height map here
        // For now, we just skip it
    }
    
    return computeTargetDofPos();
}

} // namespace bridge_core


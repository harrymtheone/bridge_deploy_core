#include "bridge_core/algorithms/mod.hpp"
#include <cmath>
#include <algorithm>

namespace bridge_core {

void Mod::initObservations() {
    AlgorithmBase::initObservations();
    
    // Proprio size: ang_vel(3) + gravity(3) + commands(3) + dof_pos(n) + dof_vel(n) + actions(n)
    size_t n_proprio = 3 + 3 + 3 + 3 * static_cast<size_t>(num_dof_activated_);
    proprio_.resize(n_proprio, 0.0f);
    
    // Initialize GRU hidden states to zeros
    // Shape: {num_layers, batch_size, hidden_size} = {2, 1, 128}
    size_t hidden_size = static_cast<size_t>(kGruNumLayers * kGruBatchSize * kGruHiddenSize);
    actor_hidden_states_.resize(hidden_size, 0.0f);
    
    RCLCPP_INFO(node_->get_logger(), "Mod initialized with %zu proprio dims, %zu hidden dims",
                n_proprio, hidden_size);
}

void Mod::reset() {
    AlgorithmBase::reset();
    
    // Reset GRU hidden states to zeros
    std::fill(actor_hidden_states_.begin(), actor_hidden_states_.end(), 0.0f);
    
    // Reset debug mode timer
    debug_time_initialized_ = false;
    
    RCLCPP_DEBUG(node_->get_logger(), "Mod: Reset hidden states");
}

std::vector<float> Mod::forward() {
    // Concatenate observations into proprio vector
    size_t idx = 0;
    
    // Angular velocity (3)
    for (size_t i = 0; i < obs_.ang_vel.size(); ++i) {
        proprio_[idx++] = obs_.ang_vel[i];
    }
    
    // Gravity projection (3)
    for (size_t i = 0; i < obs_.gravity_proj.size(); ++i) {
        proprio_[idx++] = obs_.gravity_proj[i];
    }
    
    // Commands (3)
    for (size_t i = 0; i < obs_.commands.size(); ++i) {
        proprio_[idx++] = obs_.commands[i];
    }
    
    // DOF positions (n)
    for (size_t i = 0; i < obs_.dof_pos.size(); ++i) {
        proprio_[idx++] = obs_.dof_pos[i];
    }
    
    // DOF velocities (n)
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
    
    // Prepare ONNX input tensors
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Input 1: proprio - shape {1, proprio_size}
    std::vector<int64_t> proprio_shape = {1, static_cast<int64_t>(proprio_.size())};
    
    // Input 2: hidden_states - shape {num_layers, batch_size, hidden_size}
    std::vector<int64_t> hidden_shape = {kGruNumLayers, kGruBatchSize, kGruHiddenSize};
    
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
        actor_hidden_states_.data(),
        actor_hidden_states_.size(),
        hidden_shape.data(),
        hidden_shape.size()
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
    
    // Extract actions (output 0)
    float* actions_data = output_tensors[0].GetTensorMutableData<float>();
    auto actions_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = static_cast<size_t>(actions_shape[1]);  // Assuming shape is [1, n]
    
    // Copy output to actions
    obs_.actions.resize(output_size);
    std::copy(actions_data, actions_data + output_size, obs_.actions.begin());
    
    // Update hidden states (output 1)
    if (output_tensors.size() > 1) {
        float* hidden_data = output_tensors[1].GetTensorMutableData<float>();
        auto hidden_out_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        size_t hidden_total = 1;
        for (auto dim : hidden_out_shape) {
            hidden_total *= static_cast<size_t>(dim);
        }
        
        // Copy updated hidden states
        if (hidden_total == actor_hidden_states_.size()) {
            std::copy(hidden_data, hidden_data + hidden_total, actor_hidden_states_.begin());
        }
    }
    
    ///////////////////////////  Debug  ///////////////////////////
    // Set debug_mode_ = true in mod.hpp to enable reference gait override
    
    if (debug_mode_) {
        // Initialize start time on first call
        if (!debug_time_initialized_) {
            debug_start_time_ = std::chrono::steady_clock::now();
            debug_time_initialized_ = true;
        }
        
        // Compute elapsed time using wall clock (robust to loop timing issues)
        auto now = std::chrono::steady_clock::now();
        float elapsed_sec = std::chrono::duration<float>(now - debug_start_time_).count();
        
        // Compute clock signals for left and right legs (anti-phase)
        float phase = elapsed_sec * gait_frequency_ * 2.0f * static_cast<float>(M_PI);
        float clock_l = std::sin(phase);
        float clock_r = std::sin(phase + static_cast<float>(M_PI));  // 180 degrees out of phase
        
        // Compute reference DOF positions
        float scale1 = 0.3f;
        float scale2 = 2.0f * scale1;
        
        // Initialize reference positions to zero
        std::vector<float> ref_dof_pos(obs_.actions.size(), 0.0f);
        
        // Left swing (only use negative part of sine for swing phase)
        float clock_l_swing = clock_l > 0.0f ? 0.0f : clock_l;
        if (ref_dof_pos.size() > 5) {
            ref_dof_pos[1] = clock_l_swing * scale1;   // Left hip pitch
            ref_dof_pos[4] = -clock_l_swing * scale2;  // Left knee
            ref_dof_pos[5] = clock_l_swing * scale1;   // Left ankle pitch
        }
        
        // Right swing (only use negative part of sine for swing phase)
        float clock_r_swing = clock_r > 0.0f ? 0.0f : clock_r;
        if (ref_dof_pos.size() > 11) {
            ref_dof_pos[7] = clock_r_swing * scale1;   // Right hip pitch
            ref_dof_pos[10] = -clock_r_swing * scale2; // Right knee
            ref_dof_pos[11] = clock_r_swing * scale1;  // Right ankle pitch
        }
        
        // Override neural network actions with reference gait
        for (size_t i = 0; i < obs_.actions.size() && i < ref_dof_pos.size(); ++i) {
            obs_.actions[i] = ref_dof_pos[i];
        }
        
        RCLCPP_DEBUG_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
            "Debug mode: time=%.2fs, phase=%.2f, clock_l=%.2f, clock_r=%.2f", 
            elapsed_sec, phase, clock_l, clock_r);
    }
    
    ///////////////////////////  Debug  ///////////////////////////
    
    return computeTargetDofPos();
}

} // namespace bridge_core


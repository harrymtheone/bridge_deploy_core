#include "bridge_core/algorithms/baseline.hpp"
#include "bridge_core/core/tensor_helpers.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace bridge_core {

void Baseline::initObservations() {
    AlgorithmBase::initObservations();

    // Obs size: ang_vel(3) + gravity(3) + commands(3) + dof_pos(n) + dof_vel(n) + actions(n) +
    // sin_phase(1) + cos_phase(1)
    size_t n_obs = 3 + 3 + 3 + 3 * static_cast<size_t>(num_dof_activated_) + 2;
    obs_vec_.resize(n_obs, 0.0f);

    // Initialize LSTM hidden and cell states to zeros
    // Shape: {num_layers, batch_size, hidden_size} = {1, 1, 64}
    size_t state_size = static_cast<size_t>(kLstmNumLayers * kLstmBatchSize * kLstmHiddenSize);
    h_state_.resize(state_size, 0.0f);
    c_state_.resize(state_size, 0.0f);

    // Initialize phase tracking from config (with default fallback)
    time_initialized_ = false;
    if (!config_.yaml["gait_cycle_time"]) {
        throw std::runtime_error("Baseline: 'gait_cycle_time' is required in algorithm config");
    }
    gait_cycle_time_ = config_.yaml["gait_cycle_time"].as<double>();

    RCLCPP_INFO(node_->get_logger(),
                "Baseline initialized with %zu obs dims, %zu hidden dims, gait_cycle=%.2fs", n_obs,
                state_size, gait_cycle_time_);
}

void Baseline::reset() {
    AlgorithmBase::reset();

    // Reset LSTM hidden and cell states to zeros
    std::fill(h_state_.begin(), h_state_.end(), 0.0f);
    std::fill(c_state_.begin(), c_state_.end(), 0.0f);

    // Reset phase timing
    time_initialized_ = false;

    RCLCPP_DEBUG(node_->get_logger(), "Baseline: Reset LSTM states and phase");
}

std::vector<float> Baseline::forward() {
    // Initialize start time on first call
    auto now = std::chrono::steady_clock::now();
    if (!time_initialized_) {
        start_time_ = now;
        time_initialized_ = true;
    }

    // Compute phase based on absolute elapsed time (immune to inference timing jitter)
    double elapsed_sec = std::chrono::duration<double>(now - start_time_).count();
    double phase = std::fmod(elapsed_sec / gait_cycle_time_, 1.0);

    // Compute sin/cos phase
    float sin_phase = static_cast<float>(std::sin(2.0 * M_PI * phase));
    float cos_phase = static_cast<float>(std::cos(2.0 * M_PI * phase));

    obs_.gravity_proj[0] = 0.0f;
    obs_.gravity_proj[1] = 0.0f;
    obs_.gravity_proj[2] = -1.0f;

    std::cout <<  "obs_.dof_pos: [" << obs_.dof_pos[0] << ", " << obs_.dof_pos[1] << ", " << obs_.dof_pos[2] << "]" << std::endl;
    // std::cout <<  "obs_.dof_vel: [" << obs_.dof_vel[0] << ", " << obs_.dof_vel[1] << ", " << obs_.dof_vel[2] << "]" << std::endl;
    // std::cout <<  "obs_.last_actions: [" << obs_.last_actions[0] << ", " << obs_.last_actions[1] << ", " << obs_.last_actions[2] << "]" << std::endl;
    // std::cout <<  "sin_phase: [" << sin_phase << ", " << cos_phase << "]" << std::endl;
    // std::cout <<  "cos_phase: [" << cos_phase << "]" << std::endl;
    // std::cout <<  "config_.clip_obs: [" << config_.clip_obs << "]" << std::endl;

    // Build observation vector using fluent API
    TensorBuilder(obs_vec_)
        .add(obs_.ang_vel)      // Angular velocity (3)
        .add(obs_.gravity_proj) // Gravity projection (3)
        .add(obs_.commands)     // Commands (3)
        .add(obs_.dof_pos)      // DOF positions (n)
        .add(obs_.dof_vel)      // DOF velocities (n)
        .add(obs_.last_actions) // Previous actions (n)
        .add(sin_phase)         // Sin phase (1)
        .add(cos_phase)         // Cos phase (1)
        .clip(-config_.clip_obs, config_.clip_obs);

    // Prepare ONNX input tensors
    auto input_tensors = InputTensorBuilder()
        .add(obs_vec_, {kLstmBatchSize, static_cast<int64_t>(obs_vec_.size())})
        .add(h_state_, {kLstmNumLayers, kLstmBatchSize, kLstmHiddenSize})
        .add(c_state_, {kLstmNumLayers, kLstmBatchSize, kLstmHiddenSize})
        .build();

    // Run inference
    static constexpr std::array<const char *, 3> kInputNames = {"obs", "h_in", "c_in"};
    static constexpr std::array<const char *, 3> kOutputNames = {"actions", "h_out", "c_out"};

    auto output_tensors =
        ort_session_->Run(
            Ort::RunOptions{nullptr},
            kInputNames.data(),
            input_tensors.data(),
            input_tensors.size(),
            kOutputNames.data(),
            kOutputNames.size()
        );

    // Extract outputs using TensorExtractor
    TensorExtractor extractor(output_tensors);

    // Extract actions (output 0)
    std::vector<float> current_actions = extractor.extractToVector(0);

    // Clip actions before storing
    for (auto& action : current_actions) {
        action = std::clamp(action, clip_actions_lower_, clip_actions_upper_);
    }

    // Update last_actions for next step
    obs_.last_actions = current_actions;

    // Update hidden state (output 1: h_out)
    extractor.extractTo(1, h_state_);

    // Update cell state (output 2: c_out)
    extractor.extractTo(2, c_state_);

    return computeTargetDofPos(current_actions);
}

} // namespace bridge_core

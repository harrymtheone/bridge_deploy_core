#include "bridge_core/algorithms/baseline.hpp"
#include "bridge_core/core/tensor_helpers.hpp"
#include <array>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace bridge_core
{

    void Baseline::initObservations()
    {
        AlgorithmBase::initObservations();

        // Obs size: ang_vel(3) + gravity(3) + commands(3) + dof_pos(n) + dof_vel(n) + actions(n) + sin_phase(1) + cos_phase(1)
        size_t n_obs = 3 + 3 + 3 + 3 * static_cast<size_t>(num_dof_activated_) + 2;
        obs_vec_.resize(n_obs, 0.0f);

        // Initialize LSTM hidden and cell states to zeros
        // Shape: {num_layers, batch_size, hidden_size} = {1, 1, 64}
        size_t state_size = static_cast<size_t>(kLstmNumLayers * kLstmBatchSize * kLstmHiddenSize);
        h_state_.resize(state_size, 0.0f);
        c_state_.resize(state_size, 0.0f);

        // Initialize phase tracking from config (with default fallback)
        phase_ = 0.0;
        if (!config_.yaml["gait_cycle_time"])
        {
            throw std::runtime_error("Baseline: 'gait_cycle_time' is required in algorithm config");
        }
        gait_cycle_time_ = config_.yaml["gait_cycle_time"].as<double>();

        RCLCPP_INFO(node_->get_logger(),
                    "Baseline initialized with %zu obs dims, %zu hidden dims, gait_cycle=%.2fs",
                    n_obs, state_size, gait_cycle_time_);
    }

    void Baseline::reset()
    {
        AlgorithmBase::reset();

        // Reset LSTM hidden and cell states to zeros
        std::fill(h_state_.begin(), h_state_.end(), 0.0f);
        std::fill(c_state_.begin(), c_state_.end(), 0.0f);

        // Reset phase
        phase_ = 0.0;

        RCLCPP_DEBUG(node_->get_logger(), "Baseline: Reset LSTM states and phase");
    }

    std::vector<float> Baseline::forward()
    {
        // Update phase based on control dt (decimation * dt)
        double control_dt = config_.decimation * config_.dt;
        phase_ = std::fmod(phase_ + control_dt / gait_cycle_time_, 1.0);

        // Compute sin/cos phase
        float sin_phase = static_cast<float>(std::sin(2.0 * M_PI * phase_));
        float cos_phase = static_cast<float>(std::cos(2.0 * M_PI * phase_));

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
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Input 1: obs - shape {batch_size, obs_size}
        std::vector<int64_t> obs_shape = {kLstmBatchSize, static_cast<int64_t>(obs_vec_.size())};

        // Input 2 & 3: h_in, c_in - shape {num_layers, batch_size, hidden_size}
        std::vector<int64_t> state_shape = {kLstmNumLayers, kLstmBatchSize, kLstmHiddenSize};

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            obs_vec_.data(),
            obs_vec_.size(),
            obs_shape.data(),
            obs_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            h_state_.data(),
            h_state_.size(),
            state_shape.data(),
            state_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            c_state_.data(),
            c_state_.size(),
            state_shape.data(),
            state_shape.size()));

        // Run inference
        static constexpr std::array<const char *, 3> kInputNames = {"obs", "h_in", "c_in"};
        static constexpr std::array<const char *, 3> kOutputNames = {"actions", "h_out", "c_out"};

        auto output_tensors = ort_session_->Run(
            Ort::RunOptions{nullptr},
            kInputNames.data(),
            input_tensors.data(),
            input_tensors.size(),
            kOutputNames.data(),
            kOutputNames.size());

        // Extract outputs using TensorExtractor
        TensorExtractor extractor(output_tensors);
        
        // Extract actions (output 0)
        std::vector<float> current_actions = extractor.extractToVector(0);
        
        // Update last_actions for next step
        obs_.last_actions = current_actions;

        // Update hidden state (output 1: h_out)
        extractor.extractTo(1, h_state_);

        // Update cell state (output 2: c_out)
        extractor.extractTo(2, c_state_);

        return computeTargetDofPos(current_actions);
    }

} // namespace bridge_core

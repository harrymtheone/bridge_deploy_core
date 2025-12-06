#ifndef BRIDGE_CORE_BASELINE_HPP
#define BRIDGE_CORE_BASELINE_HPP

#include "bridge_core/algorithms/algorithm_base.hpp"
#include <chrono>

namespace bridge_core {

/**
 * @brief Baseline algorithm implementation with LSTM hidden states
 * 
 * This algorithm uses an LSTM network that maintains hidden (h) and cell (c) 
 * states across forward passes. Based on unitree_rl_gym ActorCriticRecurrent.
 * 
 * Input format:
 *   - obs: [ang_vel(3), gravity(3), commands(3), dof_pos(n), dof_vel(n), actions(n), sin_phase(1), cos_phase(1)]
 *   - h_in: LSTM hidden state, shape {num_layers, batch_size, hidden_size}
 *   - c_in: LSTM cell state, shape {num_layers, batch_size, hidden_size}
 * 
 * Output format:
 *   - actions: [n] target joint positions delta
 *   - h_out: updated LSTM hidden state
 *   - c_out: updated LSTM cell state
 */
class Baseline : public AlgorithmBase {
public:
    Baseline() = default;
    ~Baseline() override = default;
    
    std::string getName() const override { return "Baseline"; }
    
    void reset() override;
    std::vector<float> forward() override;

protected:
    void initObservations() override;
    
private:
    // Proprioceptive observation vector
    std::vector<float> obs_vec_;
    
    // LSTM states: shape {num_layers, batch_size, hidden_size}
    // For Baseline: {1, 1, 64} -> flattened to 64 floats each
    std::vector<float> h_state_;  // Hidden state
    std::vector<float> c_state_;  // Cell state
    
    // Phase tracking for gait using absolute time
    std::chrono::steady_clock::time_point start_time_;
    bool time_initialized_ = false;
    double gait_cycle_time_ = 0.8; // Gait cycle period in seconds
    
    // LSTM configuration
    static constexpr int64_t kLstmNumLayers = 1;
    static constexpr int64_t kLstmHiddenSize = 64;
    static constexpr int64_t kLstmBatchSize = 1;
};

} // namespace bridge_core

#endif // BRIDGE_CORE_BASELINE_HPP


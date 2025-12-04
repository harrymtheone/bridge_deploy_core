#ifndef BRIDGE_CORE_MOD_HPP
#define BRIDGE_CORE_MOD_HPP

#include "bridge_core/algorithms/algorithm_base.hpp"
#include <chrono>

namespace bridge_core {

/**
 * @brief Mod algorithm implementation with GRU hidden states
 * 
 * This algorithm uses a recurrent network (GRU) that maintains hidden states
 * across forward passes. The model takes proprioceptive observations and
 * hidden states as input, and outputs actions and updated hidden states.
 * 
 * Input format:
 *   - proprio: [ang_vel(3), gravity(3), commands(3), dof_pos(n), dof_vel(n), actions(n)]
 *   - hidden_states: shape {num_layers, batch_size, hidden_size}
 * 
 * Output format:
 *   - actions: [n] target joint positions delta
 *   - hidden_states: updated GRU hidden states
 */
class Mod : public AlgorithmBase {
public:
    Mod() = default;
    ~Mod() override = default;
    
    std::string getName() const override { return "Mod"; }
    
    void reset() override;
    std::vector<float> forward() override;

protected:
    void initObservations() override;
    
private:
    // Proprioceptive observation vector
    std::vector<float> proprio_;
    
    // GRU hidden states: shape {num_layers * num_directions, batch_size, hidden_size}
    // For Mod: {2, 1, 128} -> flattened to 256 floats
    std::vector<float> actor_hidden_states_;
    
    // GRU configuration (defaults, can be overridden by model inspection)
    static constexpr int64_t kGruNumLayers = 2;
    static constexpr int64_t kGruHiddenSize = 128;
    static constexpr int64_t kGruBatchSize = 1;
    
    // Debug mode: set to true to override neural network with reference gait
    bool debug_mode_ = false;
    
    // Gait phase tracking for debug mode (using wall clock)
    std::chrono::steady_clock::time_point debug_start_time_;
    bool debug_time_initialized_ = false;
    float gait_frequency_ = 1.0f;  // Hz
};

} // namespace bridge_core

#endif // BRIDGE_CORE_MOD_HPP


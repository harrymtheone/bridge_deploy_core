#ifndef BRIDGE_CORE_DREAMWAQ_HPP
#define BRIDGE_CORE_DREAMWAQ_HPP

#include "bridge_core/algorithms/algorithm_base.hpp"
#include <deque>
#include <chrono>

namespace bridge_core {

/**
 * @brief DreamWAQ algorithm implementation with proprioceptive history
 * 
 * This algorithm uses an estimator network that processes proprioceptive history
 * to estimate velocity, latent variables, and height map. The actor then uses
 * these estimates along with current proprioception to output actions.
 * 
 * Input format:
 *   - proprio: [ang_vel(3), gravity(3), clock(2), commands(3), dof_pos(n), dof_vel(n), actions(n)]
 *   - prop_his: history buffer of proprio observations, shape {1, history_len, proprio_size}
 * 
 * Output format:
 *   - actions: [n] target joint positions delta
 *   - scan_recon: [32, 16] reconstructed height map (for debugging)
 * 
 * Reference: DreamWaQ: Learning Robust Quadrupedal Locomotion with Implicit Terrain Imagination
 */
class DreamWAQ : public AlgorithmBase {
public:
    DreamWAQ() = default;
    ~DreamWAQ() override = default;
    
    std::string getName() const override { return "DreamWAQ"; }
    
    void reset() override;
    std::vector<float> forward() override;

protected:
    void initObservations() override;
    
private:
    /**
     * @brief Compute clock signals for gait timing
     * @return pair of (sin, cos) clock signals
     */
    std::pair<float, float> computeClockSignals();
    
    /**
     * @brief Update the proprioceptive history buffer
     * @param proprio Current proprioceptive observation
     */
    void updateHistory(const std::vector<float>& proprio);
    
    /**
     * @brief Compute reference gait pattern for debugging
     * Uses sinusoidal motion for hip, knee, and ankle joints
     * @return Reference joint position deltas
     */
    std::vector<float> computeDebugRefGait();
    
    // Proprioceptive observation vector (56D for G1)
    // [ang_vel(3), gravity(3), clock(2), commands(3), dof_pos(15), dof_vel(15), actions(15)]
    std::vector<float> proprio_;
    
    // Proprioceptive history buffer
    // Shape: {history_len, proprio_size} stored as flat vector for ONNX
    std::vector<float> prop_his_flat_;
    std::deque<std::vector<float>> prop_his_buffer_;
    
    // Configuration
    static constexpr size_t kProprioSize = 56;       // Proprioceptive observation size
    static constexpr size_t kHistoryLen = 100;       // Number of history timesteps
    static constexpr int64_t kBatchSize = 1;         // Batch size for inference
    
    // Gait phase tracking (uses absolute time for accuracy)
    std::chrono::steady_clock::time_point start_time_;
    bool time_initialized_ = false;
    
    // DreamWAQ-specific parameters (loaded from "dreamwaq" section in config YAML)
    float gait_cycle_time_ = 0.8f;   // seconds (one full gait cycle)
    bool sw_switch_ = true;          // Enable stand-walk switch
    float lin_cmd_thresh_ = 0.2f;    // Linear command threshold for standing
    float yaw_cmd_thresh_ = 0.2f;    // Yaw command threshold for standing
    
    // Debug mode: use reference gait instead of neural network
    bool debug_mode_ = true;
    float debug_scale_1_ = 0.3f;     // Scale for hip and ankle motion
    float debug_swing_ratio_ = 0.5f; // Air ratio for swing phase
    float debug_delta_t_ = 0.2f;     // Phase offset
};

} // namespace bridge_core

#endif // BRIDGE_CORE_DREAMWAQ_HPP


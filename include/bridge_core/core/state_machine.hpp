#ifndef BRIDGE_CORE_STATE_MACHINE_HPP
#define BRIDGE_CORE_STATE_MACHINE_HPP

#include <functional>
#include <string>
#include <unordered_map>
#include <rclcpp/rclcpp.hpp>
#include "bridge_core/core/types.hpp"

namespace bridge_core {

/**
 * @brief State machine for managing robot control states
 * 
 * Implements a finite state machine with clear transition rules and callbacks.
 */
class StateMachine {
public:
    using StateCallback = std::function<void(State)>;
    using TransitionCallback = std::function<void(State, State)>;
    
    explicit StateMachine(const rclcpp::Logger& logger);
    
    /**
     * @brief Get current state
     */
    State getState() const { return current_state_; }
    
    /**
     * @brief Process a state command
     * @return true if command resulted in state change
     */
    bool processCommand(StateCommand command);
    
    /**
     * @brief Update state machine (called periodically)
     * @param dt Time since last update (seconds)
     */
    void update(float dt);
    
    /**
     * @brief Set callback for state entry
     */
    void onStateEntry(State state, StateCallback callback);
    
    /**
     * @brief Set callback for state exit
     */
    void onStateExit(State state, StateCallback callback);
    
    /**
     * @brief Set callback for any state transition
     */
    void onTransition(TransitionCallback callback);
    
    /**
     * @brief Get progress through current transition (0.0 to 1.0)
     */
    float getTransitionProgress() const;
    
    /**
     * @brief Check if currently in a transition state
     */
    bool isInTransition() const;
    
    /**
     * @brief Set transition duration
     */
    void setTransitionDuration(State state, float duration);
    
    /**
     * @brief Force state change (for error recovery)
     */
    void forceState(State new_state);
    
    /**
     * @brief Convert state to string
     */
    static std::string stateToString(State state);
    
    /**
     * @brief Convert command to string
     */
    static std::string commandToString(StateCommand cmd);

private:
    bool transition(State new_state);
    bool canTransition(State from, State to) const;
    
    State current_state_;
    State pending_command_state_;
    float transition_progress_;
    float transition_duration_;
    
    std::unordered_map<State, float> transition_durations_;
    std::unordered_map<State, StateCallback> entry_callbacks_;
    std::unordered_map<State, StateCallback> exit_callbacks_;
    TransitionCallback transition_callback_;
    
    rclcpp::Logger logger_;
};

} // namespace bridge_core

#endif // BRIDGE_CORE_STATE_MACHINE_HPP


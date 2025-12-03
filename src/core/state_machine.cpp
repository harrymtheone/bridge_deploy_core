#include "bridge_core/core/state_machine.hpp"

namespace bridge_core {

StateMachine::StateMachine(const rclcpp::Logger& logger)
    : current_state_(State::IDLE)
    , pending_command_state_(State::IDLE)
    , transition_progress_(0.0f)
    , transition_duration_(2.5f)
    , logger_(logger)
{
    transition_durations_[State::STANDING_UP] = 2.5f;
    transition_durations_[State::SITTING_DOWN] = 2.5f;
}

bool StateMachine::processCommand(StateCommand command) {
    if (command == StateCommand::NONE) {
        return false;
    }
    
    State target_state = current_state_;
    
    switch (command) {
        case StateCommand::STAND_UP:
            if (current_state_ == State::IDLE) {
                target_state = State::STANDING_UP;
                pending_command_state_ = State::STANDING;
            }
            break;
            
        case StateCommand::START_RL:
            if (current_state_ == State::STANDING) {
                target_state = State::RL_READY;
            }
            break;
            
        case StateCommand::STOP_RL:
            if (current_state_ == State::RL_RUNNING || current_state_ == State::RL_READY) {
                target_state = State::STANDING_UP;
                pending_command_state_ = State::STANDING;
            }
            break;
            
        case StateCommand::SIT_DOWN:
            if (current_state_ == State::STANDING) {
                target_state = State::SITTING_DOWN;
                pending_command_state_ = State::IDLE;
            }
            break;
            
        case StateCommand::EMERGENCY_STOP:
            if (current_state_ != State::IDLE && current_state_ != State::SITTING_DOWN) {
                target_state = State::SITTING_DOWN;
                pending_command_state_ = State::IDLE;
            }
            break;
            
        case StateCommand::RESET:
            target_state = State::IDLE;
            break;
            
        default:
            break;
    }
    
    if (target_state != current_state_) {
        return transition(target_state);
    }
    
    return false;
}

void StateMachine::update(float dt) {
    // Handle transition states
    if (current_state_ == State::STANDING_UP || current_state_ == State::SITTING_DOWN) {
        transition_progress_ += dt / transition_duration_;
        
        if (transition_progress_ >= 1.0f) {
            transition_progress_ = 1.0f;
            
            // Transition complete, move to next state
            State next_state = pending_command_state_;
            pending_command_state_ = State::IDLE;
            
            if (next_state != State::IDLE || current_state_ == State::SITTING_DOWN) {
                transition(next_state);
            }
        }
    }
    
    // Handle RL_READY -> RL_RUNNING transition (automatic)
    if (current_state_ == State::RL_READY) {
        transition(State::RL_RUNNING);
    }
}

bool StateMachine::transition(State new_state) {
    if (!canTransition(current_state_, new_state)) {
        RCLCPP_WARN(logger_, "Invalid transition from %s to %s",
                    stateToString(current_state_).c_str(),
                    stateToString(new_state).c_str());
        return false;
    }
    
    State old_state = current_state_;
    
    // Call exit callback
    auto exit_it = exit_callbacks_.find(old_state);
    if (exit_it != exit_callbacks_.end()) {
        exit_it->second(old_state);
    }
    
    // Perform transition
    current_state_ = new_state;
    
    // Initialize transition progress for transition states
    if (new_state == State::STANDING_UP || new_state == State::SITTING_DOWN) {
        transition_progress_ = 0.0f;
        auto dur_it = transition_durations_.find(new_state);
        transition_duration_ = (dur_it != transition_durations_.end()) ? dur_it->second : 2.5f;
    }
    
    // Call entry callback
    auto entry_it = entry_callbacks_.find(new_state);
    if (entry_it != entry_callbacks_.end()) {
        entry_it->second(new_state);
    }
    
    // Call transition callback
    if (transition_callback_) {
        transition_callback_(old_state, new_state);
    }
    
    RCLCPP_INFO(logger_, "State: %s -> %s",
                stateToString(old_state).c_str(),
                stateToString(new_state).c_str());
    
    return true;
}

bool StateMachine::canTransition(State from, State to) const {
    if (from == to) return false;
    
    switch (from) {
        case State::IDLE:
            return to == State::STANDING_UP;
            
        case State::STANDING_UP:
            return to == State::STANDING;
            
        case State::STANDING:
            return to == State::RL_READY || to == State::SITTING_DOWN;
            
        case State::RL_READY:
            return to == State::RL_RUNNING || to == State::STANDING || to == State::STANDING_UP;
            
        case State::RL_RUNNING:
            return to == State::STANDING || to == State::STANDING_UP;
            
        case State::SITTING_DOWN:
            return to == State::IDLE;
            
        case State::ERROR:
            return to == State::IDLE;
            
        default:
            return false;
    }
}

void StateMachine::onStateEntry(State state, StateCallback callback) {
    entry_callbacks_[state] = callback;
}

void StateMachine::onStateExit(State state, StateCallback callback) {
    exit_callbacks_[state] = callback;
}

void StateMachine::onTransition(TransitionCallback callback) {
    transition_callback_ = callback;
}

float StateMachine::getTransitionProgress() const {
    if (isInTransition()) {
        return transition_progress_;
    }
    return 1.0f;
}

bool StateMachine::isInTransition() const {
    return current_state_ == State::STANDING_UP || current_state_ == State::SITTING_DOWN;
}

void StateMachine::setTransitionDuration(State state, float duration) {
    if (state == State::STANDING_UP || state == State::SITTING_DOWN) {
        transition_durations_[state] = duration;
    }
}

void StateMachine::forceState(State new_state) {
    RCLCPP_WARN(logger_, "Forcing state change to %s", stateToString(new_state).c_str());
    current_state_ = new_state;
    transition_progress_ = 0.0f;
}

std::string StateMachine::stateToString(State state) {
    switch (state) {
        case State::IDLE: return "IDLE";
        case State::STANDING_UP: return "STANDING_UP";
        case State::STANDING: return "STANDING";
        case State::RL_READY: return "RL_READY";
        case State::RL_RUNNING: return "RL_RUNNING";
        case State::SITTING_DOWN: return "SITTING_DOWN";
        case State::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

std::string StateMachine::commandToString(StateCommand cmd) {
    switch (cmd) {
        case StateCommand::NONE: return "NONE";
        case StateCommand::STAND_UP: return "STAND_UP";
        case StateCommand::START_RL: return "START_RL";
        case StateCommand::STOP_RL: return "STOP_RL";
        case StateCommand::SIT_DOWN: return "SIT_DOWN";
        case StateCommand::EMERGENCY_STOP: return "EMERGENCY_STOP";
        case StateCommand::RESET: return "RESET";
        default: return "UNKNOWN";
    }
}

} // namespace bridge_core


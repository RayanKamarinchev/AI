import numpy as np

# --- ENVIRONMENT CONFIGURATION ---
NUM_STATES = 1000
NUM_ACTIONS = 4
GAMMA = 0.99

class Sector7Env:
    """
    Interface for the Sector 7 Navigation Task.
    
    This class defines the state space, action space, and interaction protocol.
    """
    
    def __init__(self):
        self.n_states = NUM_STATES
        self.n_actions = NUM_ACTIONS
        self.state = 0
        
        # Grid dimensions estimated for Sector 7
        self.width = 32
        self.height = 32 # 32*32 approx 1024 states
        
        self.terminals = {NUM_STATES - 1} # Goal state

    def reset(self):
        """
        Resets the environment to the starting state.
        """
        self.state = 0
        return self.state

    def step(self, action):
        """
        Simulates one time-step within the environment.
        
        Args:
            action (int): 
                0: Move Up (-width)
                1: Move Right (+1)
                2: Move Down (+width)
                3: Move Left (-1)
            
        Returns:
            next_state (int)
            reward (float)
            done (bool)
            info (dict)
        """
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Invalid action: {action}")
        
        if self.state in self.terminals:
            return self.state, 0.0, True, {}

        # 1. Resolve Dynamics
        # Standard navigational impulse with boundary checks
        row = self.state // self.width
        col = self.state % self.width
        
        next_s = self.state
        
        if action == 0:   # Up
            if row > 0: next_s -= self.width
        elif action == 1: # Right
            if col < self.width - 1: next_s += 1
        elif action == 2: # Down
            if row < self.height - 1: next_s += self.width
        elif action == 3: # Left
            if col > 0: next_s -= 1
            
        # Ensure state stays within bounds
        next_s = max(0, min(next_s, self.n_states - 1))
        
        # 2. Resolve Reward
        # Standard friction penalty for movement
        reward = -1.0
        
        # Goal Check
        if next_s in self.terminals:
            reward = 100.0
        
        # 3. Update State
        self.state = next_s
        done = (self.state in self.terminals)
        
        return next_s, reward, done, {}

    def get_action_space(self):
        return list(range(self.n_actions))

    def get_state_space(self):
        return list(range(self.n_states))
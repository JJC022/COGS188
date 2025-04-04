import numpy as np
from src.racetrack import RaceTrack

class DynamicProgramming:
    def __init__(self, env: RaceTrack, gamma=0.9, theta=1e-6, max_iterations=1000):
        """
        Initialize Dynamic Programming solver for RaceTrack environment.
        
        Args:
            env (RaceTrack): Instance of RaceTrack environment.
            gamma (float): Discount factor.
            theta (float): Convergence threshold.
            max_iterations (int): Maximum iterations for policy evaluation.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.all_states = self._generate_all_states()
        self.value_function = {}  # Dictionary to store state values
        self.policy = {}  # Dictionary to store policy (action per state)
        self._initialize_value_function()
        self._initialize_policy()
    
    def _generate_all_states(self):
        """
        Generate all possible states in the environment.
        
        Returns:
            list: A list of tuples representing all valid (x, y, vx, vy) states.
        """
        states = []
        for x in range(self.env.course.shape[0]):
            for y in range(self.env.course.shape[1]):
                # Only include valid states (not walls)
                if self.env.course[x, y] != -1:  
                    for vx in range(-self.env.MAX_VELOCITY, self.env.MAX_VELOCITY + 1):
                        for vy in range(-self.env.MAX_VELOCITY, self.env.MAX_VELOCITY + 1):
                            states.append((x, y, vx, vy))
        return states

    def _initialize_policy(self):
        """
        Initialize a random policy for each state.
        """
        for state in self.all_states:
            self.policy[state] = np.random.choice(self.env.n_actions)

    def _initialize_value_function(self):
        """
        Initialize the value function to zero for all states.
        """
        for state in self.all_states:
            self.value_function[state] = 0.0
    
    def policy_evaluation(self):
        """
        Evaluate the current policy by iteratively updating the value function.
        
        This function updates self.value_function in place until convergence
        or max_iterations is reached.
        """
        for iteration in range(self.max_iterations):
            delta = 0
            
            # Create a copy of the current value function
            old_value_function = self.value_function.copy()
            
            # Update values for all states
            for state in self.all_states:
                if self.env.is_terminal_state():
                    self.value_function[state] = 0
                    continue
                    
                action = self.policy[state]
                
                # Calculate new value based on Bellman equation
                new_value = 0
                reward, next_state = self._simulate_action(state, action)
                
                # Handle terminal states or next_state not in value_function
                next_value = self.value_function.get(next_state, 0)
                new_value = reward + self.gamma * next_value
                
                # Update value function
                self.value_function[state] = new_value
                
                # Track maximum change
                delta = max(delta, abs(old_value_function.get(state, 0) - new_value))
            
            # Check for convergence
            if delta < self.theta:
                print(f"Policy evaluation converged after {iteration+1} iterations")
                break
                
            if iteration == self.max_iterations - 1:
                print(f"Warning: Policy evaluation did not converge after {self.max_iterations} iterations")
    
    def policy_improvement(self):
        """
        Improve the policy using the current value function.
        
        Returns:
            bool: True if the policy is stable (no changes), False otherwise.
        """
        policy_stable = True
        
        for state in self.all_states:
            if self.env.is_terminal_state():
                continue
                
            old_action = self.policy[state]
            
            # Find the best action based on current value function
            best_action = None
            best_value = float('-inf')
            
            for action in range(self.env.n_actions):
                reward, next_state = self._simulate_action(state, action)
                value = reward + self.gamma * self.value_function.get(next_state, 0)
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            # Update policy
            self.policy[state] = best_action
            
            # Check if policy changed
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def policy_iteration(self):
        """
        Perform Policy Iteration algorithm.
        
        This function alternates between policy evaluation and policy improvement
        until the policy stabilizes.
        """
        iteration = 0
        max_iterations = 100  # Prevent infinite loop
        
        while iteration < max_iterations:
            iteration += 1
            print(f"Policy Iteration: {iteration}")
            
            # Step 1: Policy Evaluation
            self.policy_evaluation()
            
            # Step 2: Policy Improvement
            policy_stable = self.policy_improvement()
            
            # Step 3: Check for Convergence
            if policy_stable:
                print(f"Policy converged after {iteration} iterations")
                break
                
        if iteration == max_iterations:
            print(f"Warning: Policy did not converge after {max_iterations} iterations")
        
    def value_iteration(self):
        """
        Perform Value Iteration algorithm.
        
        This function directly computes the optimal value function and extracts
        the optimal policy.
        """
        for iteration in range(self.max_iterations):
            delta = 0
            
            # Update values for all states
            for state in self.all_states:
                if self.env.is_terminal_state():
                    self.value_function[state] = 0
                    continue
                
                old_value = self.value_function[state]
                
                # Find maximum value over all actions
                action_values = []
                for action in range(self.env.n_actions):
                    reward, next_state = self._simulate_action(state, action)
                    value = reward + self.gamma * self.value_function.get(next_state, 0)
                    action_values.append(value)
                
                # Update to the maximum value
                self.value_function[state] = max(action_values) if action_values else 0
                
                # Track maximum change
                delta = max(delta, abs(old_value - self.value_function[state]))
            
            # Check for convergence
            if delta < self.theta:
                print(f"Value iteration converged after {iteration+1} iterations")
                break
                
            if iteration == self.max_iterations - 1:
                print(f"Warning: Value iteration did not converge after {self.max_iterations} iterations")
        
        # Extract policy from value function
        self._extract_policy_from_value_function()
    
    def _extract_policy_from_value_function(self):
        """
        Extract the optimal policy from the value function.
        """
        for state in self.all_states:
            if self.env.is_terminal_state():
                self.policy[state] = 0  # Any action for terminal state
                continue
                
            # Find the best action based on current value function
            best_action = None
            best_value = float('-inf')
            
            for action in range(self.env.n_actions):
                reward, next_state = self._simulate_action(state, action)
                value = reward + self.gamma * self.value_function.get(next_state, 0)
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            self.policy[state] = best_action
    
    def _simulate_action(self, state, action):
        """
        Simulate taking an action from a given state.
        
        Args:
            state (tuple): The current state (x, y, vx, vy).
            action (int): The action to take.
        
        Returns:
            tuple: (reward, new_state) where new_state is the next state tuple.
        """
        x, y, vx, vy = state
        self.env.position = np.array([x, y])
        self.env.velocity = np.array([vx, vy])
        reward = self.env.take_action(int(action))
        new_state = tuple(self.env.get_state())
        return reward, new_state
    
    def solve(self, method='policy_iteration'):
        """
        Solve the environment using the specified DP method.
        
        Args:
            method (str): 'policy_iteration' or 'value_iteration'.
        
        Returns:
            None
        """
        if method == 'policy_iteration':
            self.policy_iteration()
        elif method == 'value_iteration':
            self.value_iteration()
        else:
            raise ValueError("Invalid method. Choose 'policy_iteration' or 'value_iteration'")
    
    def print_policy(self):
        """
        Print the policy in a readable format.
        """
        for y in range(self.env.course.shape[1] - 1, -1, -1):
            row = ''
            for x in range(self.env.course.shape[0]):
                state = (x, y, 0, 0)  # Default velocity
                if state in self.policy:
                    row += str(self.policy[state]) + ' '
                else:
                    row += 'W ' if self.env.course[x, y] == -1 else '. '
            print(row)

# Test the implementation
if __name__ == "__main__":
    tiny_course = [
        "WWWWWW",
        "Woooo+",
        "Woooo+",
        "WooWWW",
        "WooWWW",
        "WooWWW",
        "WooWWW",
        "W--WWW",
    ]
    env = RaceTrack(tiny_course)  # Use the tiny race track for testing
    dp_solver = DynamicProgramming(env)
    dp_solver.solve(method='policy_iteration')  # Change to 'value_iteration' if needed
    dp_solver.print_policy()

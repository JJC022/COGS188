import numpy as np
from tqdm import tqdm
from .discretize import quantize_state, quantize_action
from dm_control.rl.control import Environment

# Initialize the Q-table with dimensions corresponding to each discretized state variable.
def initialize_q_table(state_bins: dict, action_bins: list) -> np.ndarray:
    """
    Initialize the Q-table with dimensions corresponding to each discretized state variable.

    Args:
        state_bins (dict): The discretized bins for each state variable.
        action_bins (list): The discrete actions available.

    Returns:
        np.ndarray: A Q-table initialized to zeros with dimensions matching the state and action space.
    """
    q_table_shape = tuple(len(bins) for bins in state_bins.values()) + (len(action_bins),)
    q_table = np.zeros(q_table_shape)
    return q_table


# TD Learning algorithm
def td_learning(env: Environment, num_episodes: int, alpha: float, gamma: float, epsilon: float, state_bins: dict, action_bins: list, q_table:np.ndarray=None) -> tuple:
    """
    TD Learning algorithm for the given environment.

    Args:
        env (Environment): The environment to train on.
        num_episodes (int): The number of episodes to train.
        alpha (float): The learning rate. 
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        state_bins (dict): The discretized bins for each state variable.
        action_bins (list): The discrete actions available.
        q_table (np.ndarray): The Q-table to start with. If None, initialize a new Q-table.

    Returns:
        tuple: The trained Q-table and the list of total rewards per episode.
    """
    # Calculate state dimensions based on bins
    state_dims = []
    for key, bins_list in state_bins.items():
        for bins in bins_list:
            state_dims.append(len(bins) + 1)  # +1 for the bins edges
    
    # Initialize Q-table if not provided
    if q_table is None:
        q_table_shape = tuple(state_dims + [len(action_bins)])
        q_table = np.zeros(q_table_shape)
        
    rewards = []

    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        # Reset the environment
        time_step = env.reset()
        state = quantize_state(time_step.observation, state_bins)
        total_reward = 0
        done = False

        while not done:
            # Select action using epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.choice(len(action_bins))
            else:
                action = np.argmax(q_table[state])

            # Take action
            time_step = env.step(action_bins[action])
            next_state = quantize_state(time_step.observation, state_bins)
            reward = time_step.reward
            total_reward += reward

            # Update Q-table using TD learning
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            # Update state
            state = next_state

            # Check if the episode is done
            if time_step.last():
                done = True

        rewards.append(total_reward)

    return q_table, rewards



def greedy_policy(q_table: np.ndarray) -> callable:
    """
    Define a greedy policy based on the Q-table.    

    Args:
        q_table (np.ndarray): The Q-table from which to derive the policy.

    Returns:
        callable: A function that takes a state and returns the best action. 
    """
    def policy(state):
        return np.argmax(q_table[state])
    return policy
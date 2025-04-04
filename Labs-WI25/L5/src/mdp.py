import numpy as np

import matplotlib.pyplot as plt
from src.maze_env import MazeEnv
from typing import List, Callable, Tuple
import matplotlib.animation as animation

# Define the states and possible actions
states = np.arange(1, 6)  # States 1 through 5
actions = ['left', 'stay', 'right']  # Available actions in each state


def transition(state, action):
    """
    Transition function that determines the next state based on the current state and action.

    Parameters:
    state (int): The current state.
    action (str): The action chosen.

    Returns:
    int: The next state.
    """
    if action == "right" and state == 5: 
        return 5
    elif action == "left" and state == 1:
        state = 1
    elif action == "left":
        state += -1
    elif action == "right": 
        state += 1
    return state





def reward(state, action):
    """
    Calculate the reward for a given state and action.

    Parameters:
    state (int): The current state.
    action (str): The action taken.

    Returns:
    int: The reward.
    """
    if state == 4 and action == "right": 
        reward = 10
    else: 
        reward = -1
    return reward




def always_right_policy(state):
    """
    Policy that always returns 'right' for any given state.

    Parameters:
    state (int): The current state.

    Returns:
    str: The chosen action ('right').
    """
    return 'right'


def my_policy(state):
    """
    This function implements a custom policy.

    Parameters:
    state (int): The current state of the system.

    Returns:
    str: The action chosen by the policy.
    """
    num = np.random.random()
    if state == 1 or 5:
        if num >= 0.5: 
            action = "right"
        else: 
            action = "left"
    else: 
        if num >= 0.3:
            action = "left"
        else: 
            action = "right"

    return action
    
        
        


        
def simulate_mdp(policy: Callable, initial_state=1, simulation_depth=20):
    """
    Simulates the Markov Decision Process (MDP) based on the given policy. 
    If we reach the terminal state, the simulation ends.
    Keeps track of the number of visits to each state, the cumulative reward, and the history of visited states.

    Parameters:
    - policy: A function that takes the current state as input and returns an action.
    - initial_state: The initial state of the MDP. Default is 1.
    - simulation_depth: The maximum number of steps to simulate. Default is 20.

    Returns:
    - state_visits: An array that tracks the number of visits to each state.
    - cumulative_reward: The cumulative reward obtained during the simulation.
    - visited_history: A list that tracks the history of visited states.
    - reward_history: A list that tracks the history of rewards obtained.
    """
    current_state = initial_state
    cumulative_reward = 0
    state_visits = np.zeros(len(states)) # Track the number of visits to each state
    visited_history = [current_state] # Track the history of visited states
    reward_history = [0] # Track the history of rewards
    
    for _ in range(simulation_depth):
        action = policy(current_state)
        current_reward = reward(current_state, action)
        reward_history.append(current_reward)
        current_state = transition(current_state, action)
        visited_history.append(current_state)
        state_visits[current_state-1] +=1
        cumulative_reward += current_reward
        if current_state == 5: 
            break


    
    return state_visits, cumulative_reward, visited_history, reward_history


def new_policy(state: List[int]) -> int:
    chance = np.random.random()
    if state[1] >= 3:
        if chance <= 0.7:
            return 1
        else:
            return 0
    else:
        if chance <= 0.7:
            return 0
        else:
            return 1



        
def simulate_maze_env(env: MazeEnv, policy: Callable, num_steps=20):
    """
    Simulates the environment using the given policy for a specified number of steps.

    Parameters:
    - env: The environment to simulate.
    - policy: The policy to use for selecting actions (this is a function that takes a state as input and returns an action)
    - num_steps: The number of steps to simulate (default: 20).

    Returns:
    - path: The sequence of states visited during the simulation.
    - total_reward: The total reward accumulated during the simulation.
    """
    state = env.reset()
    total_reward = 0
    path = [state]
    done = False

    for _ in range(num_steps):
        if done: 
            break
        action = policy(state)
        next_state, reward, done = env.step(action) # Use env.step
        total_reward += reward
        path.append(next_state)
        state = next_state #Update current state

    return path, total_reward


def q_learning(env: MazeEnv, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1) -> np.ndarray:
    """
    Perform Q-learning to learn the optimal policy for the given environment.

    Args:
        env (MazeEnv): The environment to learn the policy for.
        episodes (int, optional): Number of episodes for training. Defaults to 500.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.99.
        epsilon (float, optional): Exploration rate. Defaults to 0.1.

    Returns:
        np.ndarray: The learned Q-table.
    """
    num_states = env.size**2
    num_actions = env.action_space.n +1
    q_table = np.zeros((3, 3, 3))

    for episode in range(episodes):
        state = env.reset()
        done = False


        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action) # Use env.step() and unpack 4 values
            row, col = state  
            state_index = row * env.size + col 

            row_next, col_next = next_state  
            next_state_index = row_next * env.size + col_next 
            # Q-learning update rule
            if not done:
                # Only update Q-value if not in terminal state
                print(q_table, next_state)
                best_next_action = np.argmax(q_table[next_state]) 
                td_target = reward + gamma * np.max(q_table[next_state])
            else:
                # If in terminal state, only use the reward
                td_target = reward

            
                
            td_error = td_target - q_table[state_index, action]
            q_table[state_index, action] += alpha * td_error

            state = next_state #Update the current state.


    return q_table



def simulate_maze_env_q_learning(env: MazeEnv, q_table: np.ndarray) -> Tuple[List[Tuple[int, int]], bool]:
    """
    Simulate the maze environment using the Q-table to determine the actions to take.
    Also creates an animation of the agent moving through the environment.
    
    Args:
        env (MazeEnv): The maze environment instance.
        q_table (np.ndarray): The Q-table containing action values.

    Returns:
        Tuple[List[Tuple[int, int]], bool]: A tuple containing a list of states and a boolean indicating if the episode is done.
    """

    state = env.reset()
    done = False

    starting_frame = env.render(mode="rgb_array").T
    frames = [starting_frame]  # List to store frames for animation
    states = [state]  # List to store states

    while not done:
        action = ... # TODO: Your code here
        state, _, done, _ = env.step(action)
        frames.append(
            env.render(mode="rgb_array").T
        )  # Render the environment as an RGB array
        states.append(state)

    def update_frame(i):
        ax.clear()
        ax.imshow(frames[i], cmap="viridis", origin="lower")
        ax.set_title(f"Step {i+1}")
        ax.grid("on")

    # Create animation from frames
    fig, ax = plt.subplots()
    anim = animation.FuncAnimation(fig, update_frame, frames=len(frames), interval=500)
    anim.save("mdp_q_learning.gif", writer="pillow")
    return states, done
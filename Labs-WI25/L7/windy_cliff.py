import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import imageio
import os

np.random.seed(0)

class WindyCliffWorld(gym.Env):
    def __init__(self):
        super(WindyCliffWorld, self).__init__()
        
        self.grid_size = (7, 10)
        self.start_state = (3, 0)
        self.goal_state = (3, 9)
        self.cliff = [(3, i) for i in range(1, 9)]
        self.obstacles = [(2, 4), (4, 4), (2, 7), (4, 7)]
        
        self.wind_strength = {
            (i, j): np.random.choice([-1, 0, 1]) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])
        }

        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Discrete(self.grid_size[0] * self.grid_size[1])
        
        self.state = self.start_state
        
        self.action_effects = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

    def reset(self):
        self.state = self.start_state
        return self.state_to_index(self.state)
    
    def step(self, action):
        #print(f"action effects: {self.action_effects[action][0], self.action_effects[action][1]}")
        new_state = (self.state[0] + self.action_effects[action][0], self.state[1] + self.action_effects[action][1])
        new_state = (max(0, min(new_state[0], self.grid_size[0] - 1)), max(0, min(new_state[1], self.grid_size[1] - 1)))

        #print(f"new state: {new_state}")

        # Apply wind effect
        wind = self.wind_strength[new_state]
        new_state = (new_state[0] + wind, new_state[1])
        new_state = (max(0, min(new_state[0], self.grid_size[0] - 1)), max(0, min(new_state[1], self.grid_size[1] - 1)))

        if new_state in self.cliff:
            reward = -100
            done = True
            new_state = self.start_state
        elif new_state == self.goal_state:
            reward = 10
            done = True
        elif new_state in self.obstacles:
            reward = -10
            done = False
        else:
            reward = -1
            done = False

        self.state = new_state
        return self.state_to_index(new_state), reward, done, {}
    
    def state_to_index(self, state):
        return state[0] * self.grid_size[1] + state[1]
    
    def index_to_state(self, index):
        return (index // self.grid_size[1], index % self.grid_size[1])
    
    def render(self):
        grid = np.zeros(self.grid_size)
        grid[self.state] = 1  # Current position
        for c in self.cliff:
            grid[c] = -1  # Cliff positions
        for o in self.obstacles:
            grid[o] = -0.5  # Obstacle positions
        grid[self.goal_state] = 2  # Goal position
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='viridis')
        plt.axis('off')
        fig.canvas.draw()
        plt.close(fig)
        image = np.array(fig.canvas.renderer.buffer_rgba())
        return image

# Create and register the environment
env = WindyCliffWorld()

def q_learning(env: WindyCliffWorld, num_episodes: int, alpha: float, gamma: float, epsilon: float):
    # Initialize q_table
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros([n_states, n_actions])
    
    # For tracking progress
    rewards_per_episode = []
    
    for episode in range(num_episodes):
        # Reset environment for new episode
        state_idx = env.reset()
        done = False
        total_reward = 0
        
        # Run episode
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() >= epsilon:
                # Choose greedy action
                action = np.argmax(q_table[state_idx])
            else:
                # Choose random action
                action = np.random.choice(n_actions)
            
            # Take action
            next_state_idx, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Update Q-value using Q-learning update rule
            best_next_action = np.argmax(q_table[next_state_idx])
            q_table[state_idx, action] = (1 - alpha) * q_table[state_idx, action] + \
                                        alpha * (reward + gamma * q_table[next_state_idx, best_next_action] * (not done))
            
            # Update state
            state_idx = next_state_idx
        
        rewards_per_episode.append(total_reward)
        
    return q_table, rewards_per_episode

def sarsa(env: WindyCliffWorld, num_episodes: int, alpha: float, gamma: float, epsilon: float):
    # Initialize q_table
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros([n_states, n_actions])
    
    # For tracking progress
    rewards_per_episode = []
    
    for episode in range(num_episodes):
        # Reset environment
        state_idx = env.reset()
        done = False
        total_reward = 0
        
        # Choose initial action with epsilon-greedy
        if np.random.random() >= epsilon:
            action = np.argmax(q_table[state_idx])
        else:
            action = np.random.choice(n_actions)
        
        # Run episode
        while not done:
            # Take action
            next_state_idx, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Choose next action with epsilon-greedy
            if np.random.random() >= epsilon:
                next_action = np.argmax(q_table[next_state_idx])
            else:
                next_action = np.random.choice(n_actions)
            
            # Update Q-value using SARSA update rule
            q_table[state_idx, action] = (1 - alpha) * q_table[state_idx, action] + \
                                       alpha * (reward + gamma * q_table[next_state_idx, next_action] * (not done))
            
            # Update state and action
            state_idx = next_state_idx
            action = next_action
        
        rewards_per_episode.append(total_reward)
        
    return q_table, rewards_per_episode

def save_gif(frames, path='./', filename='gym_animation.gif'):
    imageio.mimsave(os.path.join(path, filename), frames, duration=0.5)

def visualize_policy(env, q_table, filename='q_learning.gif'):
    state = env.reset()
    frames = []
    done = False
    i=0
    while not done:
        print(f"Step-{i}")
        i+=1
        action = np.argmax(q_table[state])
        state, _, done, _ = env.step(action)
        frames.append(env.render())
    
    print("epsiode done")
    save_gif(frames, filename=filename)

def plot_results(results, title, filename):
    plt.figure(figsize=(10, 6))
    
    for label, rewards in results.items():
        # Smooth rewards for better visualization
        smoothed_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(smoothed_rewards, label=label)
    
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


# Example usage:
def run_experiments():
    matplotlib.use("Agg")
    env = WindyCliffWorld()
    
    # Define hyperparameters to test
    alphas = [0.1, 0.3]
    epsilons = [0.1, 0.3]
    
    # Store results
    q_learning_results = {}
    sarsa_results = {}
    
    # Run experiments for Q-learning
    for alpha in alphas:
        for epsilon in epsilons:
            print(f"Running Q-learning with alpha={alpha}, epsilon={epsilon}")
            q_table, rewards = q_learning(env, num_episodes=500, alpha=alpha, gamma=0.99, epsilon=epsilon)
            key = f"α={alpha}, ε={epsilon}"
            q_learning_results[key] = rewards
            
            # Visualize the learned policy
            filename = f"q_learning_a{alpha}_e{epsilon}.gif"
            success = visualize_policy(env, q_table, filename=filename)
            print(f"Policy visualization {'succeeded' if success else 'failed'}")
    
    # Run experiments for SARSA
    for alpha in alphas:
        for epsilon in epsilons:
            print(f"Running SARSA with alpha={alpha}, epsilon={epsilon}")
            q_table, rewards = sarsa(env, num_episodes=500, alpha=alpha, gamma=0.99, epsilon=epsilon)
            key = f"α={alpha}, ε={epsilon}"
            sarsa_results[key] = rewards
            
            # Visualize the learned policy
            filename = f"sarsa_a{alpha}_e{epsilon}.gif"
            success = visualize_policy(env, q_table, filename=filename)
            print(f"Policy visualization {'succeeded' if success else 'failed'}")
    
    # Plot results
    plot_results(q_learning_results, "Q-Learning Performance", "q_learning_results.png")
    plot_results(sarsa_results, "SARSA Performance", "sarsa_results.png")


if __name__ == "__main__":
    run_experiments()

# TODO: Run experiments with different hyperparameters and visualize the results
# You should generate two plots:
# 1. Total reward over episodes for different α and ε values for Q-learning
# 2. Total reward over episodes for different α and ε values for SARSA
# For each plot, use at least 2 different values for α and 2 different values for ε

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

# The TileCoder class is used to discretize the continuous state space
# into a set of tiles. Each tile is represented by a tuple of indices.
# The number of tilings and the number of bins per tiling are specified
# when creating an instance of the TileCoder class.

# Read the code and understand how the TileCoder class works.

class TileCoder:
    def __init__(self, n_tilings, n_bins, low, high):
        self.n_tilings = n_tilings
        self.n_bins = n_bins
        self.low = low
        self.high = high
        self.tile_width = (high - low) / n_bins
        self.offsets = [(i / self.n_tilings) * self.tile_width for i in range(self.n_tilings)]

    def get_tiles(self, state):
        scaled_state = (state - self.low) / self.tile_width
        tiles = []
        for offset in self.offsets:
            tile_indices = tuple(((scaled_state + offset) // 1).astype(int))
            tiles.append(tile_indices)
        return tiles

    def num_tiles(self):
        return self.n_tilings * np.prod(self.n_bins)

# The create_q_table function is used to create a Q-table with the
# specified number of actions and tile coder.
def create_q_table(n_actions, tile_coder):
    return np.zeros((tile_coder.num_tiles(), n_actions))

# The get_tile_indices function is used to get the indices of the tiles
def get_tile_indices(tile_coder, tiles):
    indices = []
    for i, tile in enumerate(tiles):
        index = i * np.prod(tile_coder.n_bins) + np.ravel_multi_index(tile, tile_coder.n_bins)
        indices.append(index)
    return indices

# The get_q_values function is used to get the Q-values for the given
# tile indices
def get_q_values(q_table, tile_indices):
    print(f"tile indices in get_q_values: {tile_indices}")
    print(f"q_table[tile_indices]: {q_table[tile_indices]}")
    return np.mean([q_table[idx] for idx in tile_indices], axis=0)

# The discretize function is used to discretize the continuous state
# into a set of tiles using the tile coder
def discretize(state, tile_coder: TileCoder):
    tiles = tile_coder.get_tiles(state)
    tile_indices = get_tile_indices(tile_coder, tiles)
    return tile_indices

def q_learning(env, num_episodes, alpha, gamma, epsilon, tile_coder: TileCoder):
    """Q-learning with tracking of rewards for plotting."""
    q_table = create_q_table(env.action_space.n, tile_coder.num_tiles)
    rewards_per_episode = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            # Discretize state
            tile_indices = discretize(state, tile_coder)
            
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                print(f"q_table shape: {q_table.shape}, tile_indices max: {max(tile_indices)}")
                q_values = get_q_values(q_table, tile_indices)
                action = np.argmax(q_values)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            # Discretize next state
            next_tile_indices = discretize(next_state, tile_coder)

            # Update q_table
            current_q = get_q_values(q_table, tile_indices)[action]
            next_max_q = np.max(get_q_values(q_table, next_tile_indices))
            new_q = current_q + alpha * (reward + gamma * next_max_q * (not done) - current_q)
            for idx in tile_indices:
                q_table[idx, action] = q_table[idx, action] + alpha * (new_q - q_table[idx, action])
            state = next_state
        
        rewards_per_episode.append(total_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
    
    return {'q_table': q_table, 'rewards': rewards_per_episode}

def sarsa(env, num_episodes, alpha, gamma, epsilon, tile_coder:TileCoder):
    """SARSA with tracking of rewards for plotting."""
    q_table = create_q_table(env.action_space.n, tile_coder)
    rewards_per_episode = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        # Discretize initial state
        tile_indices = discretize(state, tile_coder)
        
        # Select initial action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            print(f"q_table shape: {q_table.shape}, tile_indices max: {max(tile_indices)}")
            q_values = get_q_values(q_table, tile_indices)
            action = np.argmax(q_values)
        
        while not (done or truncated):
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            # Discretize next state
            next_tile_indices = discretize(next_state, tile_coder)
            
            # Select next action
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_q_values = get_q_values(q_table, next_tile_indices)
                next_action = np.argmax(next_q_values)
            
            # update q_table
            current_q = get_q_values(q_table, tile_indices)[action]
            next_q = get_q_values(q_table, next_tile_indices)[next_action]
            new_q = current_q + alpha * (reward + gamma * next_q * (not done) - current_q)
            for idx in tile_indices:
                q_table[idx, action] = q_table[idx, action] + alpha * (new_q - q_table[idx, action])
            
            # Move to next state and action
            state = next_state
            tile_indices = next_tile_indices
            action = next_action
        
        rewards_per_episode.append(total_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
    
    return {'q_table': q_table, 'rewards': rewards_per_episode}

def visualize_policy(env, q_table, tile_coder, video_dir, filename='q_learning'):
    env = RecordVideo(env, video_folder=video_dir, name_prefix=filename)
    state, _ = env.reset()
    done = False

    while not done:
        tile_indices = discretize(state, tile_coder)
        q_values = get_q_values(q_table, tile_indices)
        action = np.argmax(q_values)
        state, _, done, _, _ = env.step(action)
    
    env.close()


if __name__ == "__main__":
#env = gym.make('MountainCar-v0', render_mode='rgb_array')
#tile_coder = TileCoder(n_tilings=8, n_bins=(10, 10), low=env.observation_space.low, high=env.observation_space.high)
#q_table = q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1, tile_coder=tile_coder)
#visualize_policy(env, q_table, tile_coder, video_dir='./videos', filename='q_learning_mountain_car')

    # Running SARSA
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    tile_coder = TileCoder(n_tilings=8, n_bins=(10, 10), low=env.observation_space.low, high=env.observation_space.high)
    q_table = sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1, tile_coder=tile_coder)
    visualize_policy(env, q_table, tile_coder, video_dir='./videos', filename='sarsa_mountain_car')
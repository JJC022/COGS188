import numpy as np
import pandas as pd
import random
from typing import List


def update(q: float, r: float, k: int) -> float:
    """
    Update the Q-value using the given reward and number of times the action has been taken.

    Parameters:
    q (float): The current Q-value.
    r (float): The reward received for the action.
    k (int): The number of times the action has been taken before.

    Returns:
    float: The updated Q-value.
    """
    # Note: since k is the number of times the action has been taken before this update, we need to add 1 to k before using it in the formula.
    k = k+ 1

    q += (1 / k) * (r - q)

    return q



def greedy(q_estimate: np.ndarray) -> int:
    """
    Selects the action with the highest Q-value.

    Parameters:
    q_estimate (numpy.ndarray): 1-D Array of Q-values for each action.

    Returns:
    int: The index of the action with the highest Q-value.
    """
    return np.argmax(q_estimate)


def egreedy(q_estimate: np.ndarray, epsilon: float) -> int:
    """
    Implements the epsilon-greedy exploration strategy for multi-armed bandits.

    Parameters:
    q_estimate (numpy.ndarray): 1-D Array of estimated action values.
    epsilon (float): Exploration rate, determines the probability of selecting a random action.
    n_arms (int): Number of arms in the bandit. default is 10.

    Returns:
    int: The index of the selected action.
    """
    if random.random() < epsilon: 
        return random.randint(0, len(q_estimate) -1)
    else: 
        return np.argmax(q_estimate)



def empirical_egreedy(epsilon: float, n_arms: int, n_plays: int, n_trials: int) -> List[List[float]]:
    rewards = [] 
    reward_distributions = [np.random.normal(loc=i, scale=1) for i in range(n_arms)]
    
    for trial in range(n_trials):
        q_values = np.zeros(n_arms)  
        n_actions = np.zeros(n_arms)  
        trial_rewards = []
        
        for play in range(n_plays):
            if random.random() < epsilon:
                action = random.randint(0, n_arms - 1)
            else:
                action = np.argmax(q_values)
            
            reward = np.random.normal(loc=reward_distributions[action], scale=1)
            trial_rewards.append(reward)
            
            n_actions[action] += 1
            q_values[action] += (reward - q_values[action]) / n_actions[action]
        
        rewards.append(trial_rewards)
    
    return rewards
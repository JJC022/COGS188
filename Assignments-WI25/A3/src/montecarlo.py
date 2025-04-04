import numpy as np
import random
from collections import defaultdict
from src.racetrack import RaceTrack

class MonteCarloControl:
    """
    Monte Carlo Control with Weighted Importance Sampling for off-policy learning.
    
    This class implements the off-policy every-visit Monte Carlo Control algorithm
    using weighted importance sampling to estimate the optimal policy for a given
    environment.
    """
    def __init__(self, env: RaceTrack, gamma: float = 1.0, epsilon: float = 0.1, Q0: float = 0.0, max_episode_size: int = 1000):
        """
        Initialize the Monte Carlo Control object. 

        Q, C, and policies are defaultdicts that have keys representing environment states.  
        Defaultdicts allow you to set a sensible default value 
        for the case of Q[new state never visited before] (and likewise with C/policies).  
        
        Args:
            env (RaceTrack): The environment in which the agent operates.
            gamma (float): The discount factor.
            epsilon (float): Exploration parameter for ε-greedy policies.
            Q0 (float): The initial Q values for all states (optimistic initialization).
            max_episode_size (int): Cutoff to prevent running forever during MC.
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_episode_size = max_episode_size
        
        # Initialize defaultdicts with proper defaults
        self.Q = defaultdict(lambda: np.full(self.env.n_actions, Q0))
        self.C = defaultdict(lambda: np.zeros(self.env.n_actions))
        
        # Initialize policies
        self.target_policy = defaultdict(lambda: np.zeros(self.env.n_actions))
        self.behavior_policy = defaultdict(lambda: np.ones(self.env.n_actions) / self.env.n_actions)

    def create_target_greedy_policy(self):
        """
        Loop through all states in the self.Q dictionary. 
        1. determine the greedy policy for that state
        2. create a probability vector that is all 0s except for the greedy action where it is 1
        3. store that probability vector in self.target_policy[state]

        Args: none
        Returns: none, stores new policy in self.target_policy
        """
        for state_str in self.Q.keys():
            # Find the greedy action (action with highest Q-value)
            greedy_action = np.argmax(self.Q[state_str])
            
            # Set probability 1.0 for the greedy action, 0.0 for others
            prob_vector = np.zeros(self.env.n_actions)
            prob_vector[greedy_action] = 1.0
            
            self.target_policy[state_str] = prob_vector

    def create_behavior_egreedy_policy(self):
        """
        Loop through all states in the self.target_policy dictionary. 
        Using that greedy probability vector, and self.epsilon, 
        calculate the epsilon greedy behavior probability vector and store it in self.behavior_policy[state]
        
        Args: none
        Returns: none, stores new policy in self.target_policy
        """
        # First make sure we have an updated target policy
        self.create_target_greedy_policy()
        
        for state_str in self.Q.keys():
            # Start with equal probabilities for all actions
            prob_vector = np.ones(self.env.n_actions) * (self.epsilon / self.env.n_actions)
            
            # Find the greedy action
            greedy_action = np.argmax(self.Q[state_str])
            
            # Increase probability for the greedy action
            prob_vector[greedy_action] += (1.0 - self.epsilon)
            
            self.behavior_policy[state_str] = prob_vector

    def egreedy_selection(self, state):
        """
        Select an action proportional to the probabilities of epsilon-greedy encoded in self.behavior_policy
        HINT: 
        - check out https://www.w3schools.com/python/ref_random_choices.asp
        - note that random_choices returns a numpy array, you want a single int
        - make sure you are using the probabilities encoded in self.behavior_policy 

        Args: state (string): the current state in which to choose an action
        Returns: action (int): an action index between 0 and self.env.n_actions
        """
        # Convert state to string for dictionary lookup
        state_str = str(state)
        
        # Get probability vector from behavior policy
        probability_vector = self.behavior_policy[state_str]
        
        # Sample action according to the probability distribution
        action = random.choices(
            range(self.env.n_actions), 
            weights=probability_vector, 
            k=1
        )[0]
        
        return action

    def generate_egreedy_episode(self):
        """
        Generate an episode using the epsilon-greedy behavior policy. Will not go longer than self.max_episode_size
        
        Hints: 
        - need to setup and use self.env methods and attributes
        - use self.egreedy_selection() above as a helper function
        - use the behavior e-greedy policy attribute aleady calculated (do not update policy here!)
        
        Returns:
            list: The generated episode, which is a list of (state, action, reward) tuples.
        """
        # Reset the environment to start a new episode
        self.env.reset()
        
        episode = []
        episode_length = 0
        
        while episode_length < self.max_episode_size:
            # Get current state
            state = self.env.get_state()
            state_str = str(state)
            
            # Select action using ε-greedy policy
            action = self.egreedy_selection(state)
            
            # Take action and observe reward
            reward = self.env.take_action(action)
            
            # Record state, action, reward
            episode.append((state_str, action, reward))
            episode_length += 1
            
            # Check if terminal state reached
            if self.env.is_terminal_state():
                break
        
        return episode

    def generate_greedy_episode(self):
        """
        Generate an episode using the greedy target policy. Will not go longer than self.max_episode_size
        Note: this function is not used during learning, its only for evaluating the target policy
        
        Hints: 
        - need to setup and use self.env methods and attributes
        - use the greedy policy attribute aleady calculated (do not update policy here!)

        Returns:
            list: The generated episode, which is a list of (state, action, reward) tuples.
        """
        # Reset the environment to start a new episode
        self.env.reset()
        
        episode = []
        episode_length = 0
        
        while episode_length < self.max_episode_size:
            # Get current state
            state = self.env.get_state()
            state_str = str(state)
            
            # Select action using greedy policy
            action = np.argmax(self.target_policy[state_str])
            
            # Take action and observe reward
            reward = self.env.take_action(int(action))
            
            # Record state, action, reward
            episode.append((state_str, action, reward))
            episode_length += 1
            
            # Check if terminal state reached
            if self.env.is_terminal_state():
                break
        
        return episode

    def update_offpolicy(self, episode):
        """
        Update the Q-values using every visit weighted importance sampling. 
        See Figure 5.9, p. 134 of Sutton and Barto 2nd ed.
        
        Args: episode (list): An episode generated by the behavior policy; a list of (state, action, reward) tuples.
        Returns: none
        """
        # Initialize variables
        G = 0.0  # Return
        W = 1.0  # Importance sampling weight
        
        # Process the episode in reverse order
        for t in range(len(episode) - 1, -1, -1):
            state_str, action, reward = episode[t]
            
            # Update return
            G = self.gamma * G + reward
            
            # Update counts
            self.C[state_str][action] += W
            
            # Update Q-value using weighted importance sampling
            if self.C[state_str][action] > 0:
                self.Q[state_str][action] += (W / self.C[state_str][action]) * (G - self.Q[state_str][action])
            
            # If the action taken is not greedy according to target policy, break
            if action != np.argmax(self.target_policy[state_str]):
                break
            
            # Update importance sampling weight
            # W *= (self.target_policy[state_str][action] / self.behavior_policy[state_str][action])
            # Using the ratio of probabilities from target and behavior policies
            target_prob = self.target_policy[state_str][action]
            behavior_prob = self.behavior_policy[state_str][action]
            
            # Avoid division by zero
            if behavior_prob == 0:
                break
                
            W *= (target_prob / behavior_prob)
            
            # If weight becomes too small, break to reduce variance
            if W == 0:
                break
        
        # Update policies after updating Q-values
        self.create_target_greedy_policy()
        self.create_behavior_egreedy_policy()

    def update_onpolicy(self, episode):
        """
        Update the Q-values using first visit epsilon-greedy. 
        See Figure 5.6, p. 127 of Sutton and Barto 2nd ed.
        
        Args: episode (list): An episode generated by the behavior policy; a list of (state, action, reward) tuples.
        Returns: none
        """
        # Dictionary to track first visit
        first_visit = {}
        
        # Initialize return
        G = 0.0
        
        # Process the episode in reverse order
        for t in range(len(episode) - 1, -1, -1):
            state_str, action, reward = episode[t]
            
            # Create a state-action pair key
            sa_pair = (state_str, action)
            
            # Update return
            G = self.gamma * G + reward
            
            # Check if this is the first visit to this state-action pair
            if sa_pair not in first_visit:
                first_visit[sa_pair] = True
                
                # Update Q-value
                self.Q[state_str][action] += (1 / (self.C[state_str][action] + 1)) * (G - self.Q[state_str][action])
                
                # Update counts
                self.C[state_str][action] += 1
        
        # Update policy to be ε-greedy with respect to current Q
        for state_str in self.Q.keys():
            greedy_action = np.argmax(self.Q[state_str])
            
            # Set probabilities for epsilon-greedy policy
            for a in range(self.env.n_actions):
                if a == greedy_action:
                    self.behavior_policy[state_str][a] = 1 - self.epsilon + (self.epsilon / self.env.n_actions)
                else:
                    self.behavior_policy[state_str][a] = self.epsilon / self.env.n_actions

    def train_offpolicy(self, num_episodes):
        """
        Train the agent over a specified number of episodes.
        
        Args:
            num_episodes (int): The number of episodes to train the agent.
        """
        # Initialize policies
        self.create_target_greedy_policy()
        self.create_behavior_egreedy_policy()
        
        for episode_num in range(num_episodes):
            # Generate an episode using the behavior policy
            episode = self.generate_egreedy_episode()
            
            # Update Q-values and policies
            self.update_offpolicy(episode)
            
            # Optionally print progress
            #if (episode_num + 1) % 100 == 0:
                #print(f"Completed {episode_num + 1} episodes")

    def train_onpolicy(self, num_episodes):
        """
        Train the agent over a specified number of episodes using on-policy learning.
        
        Args:
            num_episodes (int): The number of episodes to train the agent.
        """
        for episode_num in range(num_episodes):
            # Generate an episode using the behavior policy
            episode = self.generate_egreedy_episode()
            
            # Update Q-values and policy
            self.update_onpolicy(episode)
            
            # Optionally print progress
            #if (episode_num + 1) % 100 == 0:
                #print(f"Completed {episode_num + 1} episodes")

    def get_greedy_policy(self):
        """
        Retrieve the learned target policy in the form of an action index per state.
        
        Returns:
            dict: The learned target policy with best action for each state.
        """
        policy = {}
        for state_str, q_values in self.Q.items():
            policy[state_str] = np.argmax(q_values)
        return policy
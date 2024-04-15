import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

random.seed(666)
np.random.seed(666)
class Node:
  def __init__(self, state, parent=None):
    self.state = state
    self.parent = parent
    self.children = {}  # Dictionary to store child nodes (action -> Node)
    self.num_visits = 0
    self.total_reward = 0

class MCTSAgent:
  def __init__(self, env, simulations=100):
    self.env = env
    self.simulations = simulations

  def get_optimal_action(self, state):
    # Create a root node for the current state
    root = Node(state)
    # Perform MCTS simulations
    for _ in range(self.simulations):
      self.simulate(root)

    # Select the child node with the highest action value (UCT)
    best_child = max(root.children.values(), key=lambda node: self.uct(node))
    return best_child.state[1]  # Return action (hit/stand)

  def simulate(self, node):
    # Check for terminal state (player bust)
    if node.state[0][0] > 21:
      return -1  # Reward (loss)

    # Check for terminal state (dealer's turn)
    if node.state[0][0] == 21:
      return 1  # Reward (win)

    # Select or expand the node
    if not node.children:
      self.expand_node(node)

    # Select the most promising child node using UCT
    next_node = self.select_child(node)

    # Simulate the player's action and dealer's turn
    reward = self.simulate_game(next_node.state)

    # Backpropagate reward
    self.backpropagate(next_node, reward)

  def expand_node(self, node):
    # Get all possible actions (hit/stand)
    actions = [0, 1]
    for action in actions:
      next_state, reward, done, _,_ = self.env.step(action)
      # Create a child node for each action
      node.children[action] = Node(next_state, node)
      # Reset environment for next simulation
      self.env.reset()

  def select_child(self, node):
    # Use Upper Confidence Bound Applied to Trees (UCT) formula
    c = 1  # Exploration parameter
    return max(node.children.values(), key=lambda child: self.uct(child, c))

  def uct(self, node, c):
    # Balance exploration and exploitation
    if node.num_visits == 0:
      return float('inf')
    return (node.total_reward / node.num_visits) + c * np.sqrt(np.log(node.parent.num_visits) / node.num_visits)

  def simulate_game(self, state):
    # Simulate player's remaining actions (random hit until not bust)
    while True:
      action = random.choice([0, 1])
      state, reward, done, _,_ = self.env.step(action)
      if done or state[0] > 21:
        break

    # Simulate dealer's turn (hit until reaching or exceeding 17)
    dealer_sum = state[0]
    while dealer_sum < 17:
      _, _, done, _ = self.env.step(1)  # Dealer hits
      dealer_sum = self.env.state[0]

    # Calculate reward based on final outcome
    if dealer_sum > 21:
      return 1  # Player wins
    elif dealer_sum > state[0]:
      return -1  # Player loses
    else:
      return 0  # Tie

  def backpropagate(self, node, reward):
    while node:
      node.num_visits += 1
      node.total_reward += reward
      node = node.parent

n_episodes = 10000
blackjack_env = gym.make('Blackjack-v1', natural=True, sab=False, render_mode='rgb_array')
blackjack_env = gym.wrappers.RecordEpisodeStatistics(blackjack_env, buffer_length=n_episodes)
agent = MCTSAgent(blackjack_env, simulations=500)  # Adjust simulations for desired exploration/exploitation balance

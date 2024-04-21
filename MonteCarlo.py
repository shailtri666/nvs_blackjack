import numpy as np
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm
import random
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

random.seed(666)
np.random.seed(666)


class MonteCarloTreeSearchAgent:
    def __init__(self, env, num_iterations):
        self.env = env
        self.num_iterations = num_iterations

    def train_agent(self):
        root_state,_ = self.env.reset()
        root_node = Node(root_state)

        for _ in tqdm(range(self.num_iterations)):
            # Selection and Expansion
            action = self.env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = self.env.step(action)
            node = root_node
            while terminated==False and not node.is_leaf():
                node = self.tree_policy(node)
                if not terminated==False and node.is_leaf():
                    action = self.env.action_space.sample()  # Randomly choose an action
                    new_state, _, _, _,_ = self.env.step(action)
                    node.children.append(Node(new_state, action, node))
                    node = node.children[-1]

            # Simulation
            reward = self.simulate_episode(node.state)
            print(reward)

            # Backpropagation
            self.backpropagate(node, reward)
        if root_node.children:
            return max(root_node.children, key=lambda child: child.visit_count).action
        else:
            # If root_node has no children, select a random action
            return self.env.action_space.sample()

        #return max(root_node.children, key=lambda child: child.visit_count).action

    def tree_policy(self, node, exploration_constant=1.41):
        if node.is_leaf():
            return node

        best_child = None
        best_score = -np.inf
        for child in node.children:
            exploit_term = child.total_reward / child.visit_count
            explore_term = exploration_constant * np.sqrt(np.log(node.visit_count) / child.visit_count)
            score = exploit_term + explore_term
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def set_state(self, player, dealer, usable_ace):
        """Sets the state to the provided values."""
        self.player = [player]
        self.dealer = [dealer]
        self.usable_ace = usable_ace
    
    def simulate_episode(self, state):
        player,dealer,usable_ace = state  # Unpack the state tuple
        self.set_state(player, dealer, usable_ace)
        done = False
        total_reward = 0
        while not done:
            action = self.env.action_space.sample()
            _, reward, done, _,_ = self.env.step(action)
            total_reward += reward
        return total_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

class Node:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_reward = 0

    def is_leaf(self):
        return len(self.children) == 0

# Example usage
env = gym.make('Blackjack-v1', natural=True, sab=False, render_mode='rgb_array')
initial_state,_ = env.reset()
print(initial_state)
mcts_agent = MonteCarloTreeSearchAgent(env, num_iterations=1000)
best_action = mcts_agent.train_agent()
print("Best action:", best_action)


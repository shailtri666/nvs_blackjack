import numpy as np
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

random.seed(666)
np.random.seed(666)

class MonteCarloTreeSearchAgent:
    def __init__(self, env, num_iterations,epsilon):
        self.env = env
        self.num_iterations = num_iterations
        self.player = None
        self.dealer = None
        self.usable_ace = None
        self.num_wins = 0 
        self.episode_rewards = []
        self.epsilon = epsilon
        self.policy = defaultdict(lambda: np.zeros(env.action_space.n))

    def train_agent(self):
        root_state,_ = self.env.reset()
        root_node = Node(root_state)


        for _ in tqdm(range(self.num_iterations)):

            print('root_node',root_state)

            action = self.env.action_space.sample() 
            observation, reward, terminated, truncated, info = self.env.step(action)
            print("next move:",observation)
            node = root_node
            while terminated==False and not node.is_leaf():
                node = self.tree_policy(node)
                if not terminated==False and node.is_leaf():
                    action = self.env.action_space.sample()  # Randomly choose an action
                    new_state, _, _, _,_ = self.env.step(action)
                    node.children.append(Node(new_state, action, node))
                    node = node.children[-1]
            if terminated==True:
                root_state,_ = self.env.reset()
                root_node = Node(root_state)
                
        
            # Simulation
            reward,is_win = self.simulate_episode(node.state)
            self.episode_rewards.append(reward)
            # Backpropagation
            self.backpropagate(node, reward)
            if is_win:
                self.num_wins += 1
                                
        if root_node.children:
            #print(max(root_node.children, key=lambda child: child.visit_count).action)
            return max(root_node.children, key=lambda child: child.visit_count).action
        else:
            # If root_node has no children, select a random action
            #print(self.env.action_space.sample())
            return self.env.action_space.sample()

        #return max(root_node.children, key=lambda child: child.visit_count).action


            
    def tree_policy(self, node, exploration_constant=1.41):
        if node.is_leaf():
            return node

        best_child = None
        best_score = -np.inf
        for child in node.children:
            if child.visit_count == 0:
                exploit_term = np.inf 
                explore_term = 0 # Assign infinity if visit count is zero
            else:
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
        self.player, self.dealer, self.usable_ace = state
        self.set_state(self.player,self.dealer, self.usable_ace)
        done = False
        total_reward = 0
        while not done:
            action = self.env.action_space.sample()
            _, reward, done, _, _ = self.env.step(action)
            total_reward += reward
        if total_reward>=1:
            is_win=True
        else:
            is_win=False
        return total_reward, is_win

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
mcts_agent = MonteCarloTreeSearchAgent(env, num_iterations=1000,epsilon=0.1)
mcts_agent.train_agent()
#r, i =mcts_agent.simulate_episode()
win_rate = mcts_agent.num_wins / mcts_agent.num_iterations
print("Win rate:", win_rate)

plt.plot(mcts_agent.episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards')
plt.grid(True)
plt.show()



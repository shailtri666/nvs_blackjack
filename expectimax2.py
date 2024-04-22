import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

np.random.seed(666)


def probability_of_sum_between_given_range(starting_sum=None, lower_bound=17, upper_bound=21):
    # (31 because 21 + 10 = 31 is the maximum sum we might need to consider)
    P = [0] * 32

    # Set the base cases
    for s in range(lower_bound, upper_bound + 1):
        P[s] = 1

    # Fill the table backwards from 16 down to 0
    for s in range(lower_bound - 1, starting_sum - 1, -1):
        P[s] = sum(P[s + 1: s + 11]) / 10

    if starting_sum is not None:
        return P[starting_sum]
    else:
        return P


def generate_policy_grid(agent, usable_ace=False):
    policy = defaultdict(int)
    for obs, action_values in agent.expected_values.items():
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(8, 22),
        np.arange(1, 11),
    )

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return policy_grid


def create_plots(policy_grid, title, filename):
    fig, ax = plt.subplots()
    im = ax.imshow(policy_grid, cmap='Pastel1')  # Set3 with black, Pastel1 with black

    ax.set_xticks(range(0, 14), labels=list(map(str, range(8, 22))))
    ax.set_yticks(range(0, 10), ["A"] + list(map(str, range(2, 11))))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(policy_grid.shape[0]):
        for j in range(policy_grid.shape[1]):
            ax.text(j, i, policy_grid[i, j],
                    ha="center", va="center", color="black")

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.title(title)
    plt.savefig(filename)
    plt.show()


def usable_ace(state):  # Does this hand have a usable ace?
    return int(state[2] and state[0] + 10 <= 21)


def sum_hand(state):  # Return current hand total
    if usable_ace(state):
        return state[0] + 10
    return state[0]


class ExpectimaxAgent:
    def __init__(self):
        self.env = gym.make("Blackjack-v1")
        self.expected_values = defaultdict(lambda: np.array([-2.0, -2.0]))

    def observe(self, observation):
        self.player_sum = observation[0]
        self.dealer_showing = observation[1]
        self.usable_ace = observation[2]

    def policy(self):
        # Expectimax function to evaluate state and return action
        def expectimax(state, episode_complete, return_both_values=False):

            hit_value = -1
            stand_value = -1
            flag_calculate_stand_value = True
            flag_calculate_hit_value = True

            if state in self.expected_values and self.expected_values[state][0] > -2:
                stand_value = self.expected_values[state][0]
                flag_calculate_stand_value = False
            if state in self.expected_values and self.expected_values[state][1] > -2:
                hit_value = self.expected_values[state][1]
                flag_calculate_hit_value = False

            # Player's turn (maximize expected reward)
            if not episode_complete:
                if flag_calculate_hit_value:
                    hit_value = [expectimax((sum_hand(state) + card, state[1], state[2]), (sum_hand(state) + card > 21))
                                 for card in range(2, 11)]
                    ace_state = (state[0] + 1, state[1], 1)
                    ace_hit_value = expectimax(ace_state, (sum_hand(ace_state) > 21))
                    hit_value.append(ace_hit_value)
                    hit_value = np.average(hit_value)
                    self.expected_values[state][1] = hit_value
                if flag_calculate_stand_value:
                    # For stand value calculation
                    stand_value = -1
                    if state[0] < 17:
                        # Probability dealer lands a sum between 17 and 21 given his shown card
                        loss_probability = probability_of_sum_between_given_range(state[1])
                        stand_value = -1 * loss_probability + 1 * (1 - loss_probability)
                    elif 17 <= state[0] <= 21:
                        win_range = range(17, state[0])
                        loss_range = range(state[0] + 1, 22)
                        if state[0] > 17:
                            win_probability = probability_of_sum_between_given_range(state[1], win_range[0],
                                                                                     win_range[-1])
                        else:
                            win_probability = 0
                        if state[0] < 21:
                            loss_probability = probability_of_sum_between_given_range(state[1], loss_range[0],
                                                                                      loss_range[-1])
                        else:
                            loss_probability = 0
                        stand_value = -1 * loss_probability + 1 * win_probability  # Excluding draw since reward = 0
                    self.expected_values[state][0] = stand_value
            if return_both_values:
                return stand_value, hit_value
            else:
                return max(hit_value, stand_value)

        # Call expectimax for initial state (considering all dealer possibilities)
        value = expectimax((self.player_sum, self.dealer_showing, self.usable_ace), False, True)
        # Choose action based on maximizing expected reward
        return 0 if value[0] > value[1] else 1  # Stand if expected reward is better, Hit otherwise

    def run(self, num_episodes=100):
        for ep_num in tqdm(range(num_episodes)):
            observation, _ = self.env.reset()
            while True:
                self.observe(observation)
                action = self.policy()
                observation, reward, done, _, _ = self.env.step(action)
                # Optional: Implement learning or update strategy based on reward
                if done:
                    break


def get_win_rate(policy, num_games, env):
    state, _ = env.reset()
    num_wins = 0

    for _ in tqdm(range(num_games)):
        episode_complete_flag = False
        reward = 0

        while not episode_complete_flag:
            action = np.argmax(policy[state])
            state, reward, episode_complete_flag, truncated, info = env.step(action)

        if reward > 0:
            num_wins += 1

        state, info = env.reset()

    return num_wins * 100 / num_games


if __name__ == "__main__":
    agent = ExpectimaxAgent()
    agent.run(num_episodes=100)
    regular_dict_object = {str(key): value.tolist() for key, value in agent.expected_values.items()}
    policy_grid = generate_policy_grid(agent, usable_ace=True)
    create_plots(policy_grid, title="With usable ace", filename="expectimax_usable_ace_policy_heatmap.png")
    policy_grid = generate_policy_grid(agent, usable_ace=False)
    create_plots(policy_grid, title="Without usable ace", filename="expectimax_unusable_ace_policy_heatmap.png")
    win_rate = get_win_rate(agent.expected_values, 100000, agent.env)
    print("Expectimax's win rate: ", win_rate)
    agent.env.close()

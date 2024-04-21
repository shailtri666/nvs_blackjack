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
        np.arange(12, 22),
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

    ax.set_xticks(range(0, 10), labels=list(map(str, range(12, 22))))
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
                    hit_value = np.average(
                        [expectimax((state[0] + card, state[1], state[2]), (state[0] + card > 21)) for card in
                         range(2, 10)])
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
                            win_probability = probability_of_sum_between_given_range(state[1], win_range[0], win_range[-1])
                        else:
                            win_probability = 0
                        if state[0] < 21:
                            loss_probability = probability_of_sum_between_given_range(state[1], loss_range[0],
                                                                                      loss_range[-1])
                        else:
                            loss_probability = 0
                        stand_value = -1 * loss_probability + 1 * win_probability  # Excluding draw since reward = 0
                    self.expected_values[state][0] = stand_value
                    # for action in [0, 1]:  # 0: Stand, 1: Hit
                    #     next_state, reward, done, _, _ = self.env.step(action)
                    #     # Expected value considering dealer's cards (average reward across all dealer possibilities)
                    #     if not done:
                    #         expected_reward = sum(
                    #             expectimax(next_state, episode_complete) / 10)
                    #     else:
                    #         expected_reward = reward
                    #     value = max(value, expected_reward)
                    #     self.env.step(1)  # Step back after simulating hit action
                    # return max(hit_value, stand_value)

            # Dealer's turn (minimize expected reward for player)
            # else:
            #     # value = float("inf")
            #     # while self.is_dealer_playing(state):
            #     #     next_state, reward, done, _, _ = self.env.step(1)  # Dealer hits
            #     #     value = min(value, expectimax(next_state, episode_complete))
            #     # return value
            #     return -1
            if return_both_values:
                return stand_value, hit_value
            else:
                return max(hit_value, stand_value)

        # Call expectimax for initial state (considering all dealer possibilities)
        value = expectimax((self.player_sum, self.dealer_showing, self.usable_ace), False, True)
        # Choose action based on maximizing expected reward
        return 0 if value[0] > value[1] else 1  # Stand if expected reward is better, Hit otherwise

    def is_dealer_playing(self, state):
        return state[1] < 17

    def probability(self, card):
        # Assume uniform probability for simplicity (can be improved with card counting)
        return 1 / 10

    def run(self, num_episodes=100):
        for _ in tqdm(range(num_episodes)):
            observation, _ = self.env.reset()
            while True:
                self.observe(observation)
                action = self.policy()
                observation, reward, done, _, _ = self.env.step(action)
                # Optional: Implement learning or update strategy based on reward
                if done:
                    break


if __name__ == "__main__":
    agent = ExpectimaxAgent()
    agent.run(num_episodes=100000)
    regular_dict_object = {str(key): value.tolist() for key, value in agent.expected_values.items()}
    with open('data.json', 'w') as json_file:
        json.dump(regular_dict_object, json_file, indent=4)
    policy_grid = generate_policy_grid(agent, usable_ace=True)
    create_plots(policy_grid, title="With usable ace", filename="expectimax_usable_ace_policy_heatmap.png")
    policy_grid = generate_policy_grid(agent, usable_ace=False)
    create_plots(policy_grid, title="Without usable ace", filename="expectimax_unusable_ace_policy_heatmap.png")
    agent.env.close()

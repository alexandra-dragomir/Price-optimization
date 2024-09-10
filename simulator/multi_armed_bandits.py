import numpy as np
import math
import torch.distributions as D
import torch as th

NR_AGENTS = 17000
NR_AGENTS_MEAN = 17000
NR_AGENTS_STD = 5000
PROUDCT_PRICE = 300


class EpsilonGreedy:
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)  # Number of times each arm was pulled
        self.values = np.zeros(n_arms)  # Estimated value (mean reward) for each arm

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        else:
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

class RandomStrategy:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        return np.random.randint(0, self.n_arms)

    def update(self, chosen_arm, reward):
        pass  # Random strategy does not learn, so no update is needed

class UCB:
    def __init__(self, n_arms, confidence):
        self.confidence = confidence
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)  # Number of times each arm was pulled
        self.values = np.zeros(n_arms)  # Estimated value (mean reward) for each arm

    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm
    
        ucb_values = [0.0 for _ in range(n_arms)]
        total_counts = sum(self.counts)
        
        for arm in range(n_arms):
            bonus = math.sqrt((math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + self.confidence * bonus
        return ucb_values.index(max(ucb_values))
    

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value


def orders_and_rate_per_day(pop_dist, discount_sample, same_agents=False, agents=[], fixed_nr_agents=True, nr_agents=NR_AGENTS, mean_agents=NR_AGENTS_MEAN, std_agents=NR_AGENTS_STD):
    if same_agents:
        # print(agents, type(agents))
        if agents.shape[0] > 0:
            agents_with_discount = pop_dist.sample_discount(agents, discount_sample)
            result = pop_dist.act(agents_with_discount)
            return float(th.sum(result)), float(result.mean())
        else:
            raise ValueError("No agents in the input. Generate agents and pass them in the 'agents' parameter.")
    
    if fixed_nr_agents:
        agents = pop_dist.sample(discount_sample, nr_agents)

    else:
        nr_agents_distribution = D.Normal(mean_agents, std_agents)

        nr_agents = int(nr_agents_distribution.sample())
        agents = pop_dist.sample(discount_sample, nr_agents)

    result = pop_dist.act(agents)

    return float(th.sum(result)), float(result.mean())

def simulate(population_model, same_agents=False, agents=[], fixed_nr_agents=True, nr_agents=NR_AGENTS, mean_agents=NR_AGENTS_MEAN, std_agents=NR_AGENTS_STD, algorithm='epsilon_greedy', discounts=[0.3, 0.5, 0.8],  n_days=10, epsilon=0.1, reward='orders', product_price=PROUDCT_PRICE):
    n_arms = len(discounts)

    if algorithm == 'epsilon_greedy':
        mab = EpsilonGreedy(n_arms, epsilon)
    elif algorithm == 'random':
        mab = RandomStrategy(n_arms)
    elif algorithm == 'UCB':
        mab = UCB(n_arms, epsilon)

    rewards_per_day = np.zeros(n_days)

    for day in range(n_days):
        chosen_arm = mab.select_arm()
        discount = discounts[chosen_arm]

        # simulate the buying process
        if same_agents:
            orders, rate = orders_and_rate_per_day(population_model, discount, same_agents=True, agents=agents)
        else:
            orders, rate = orders_and_rate_per_day(population_model, discount, same_agents=False, fixed_nr_agents=fixed_nr_agents, nr_agents=nr_agents, mean_agents=mean_agents, std_agents=std_agents)

        if reward == 'orders':
            mab.update(chosen_arm, orders)
            rewards_per_day[day] = orders
        elif reward == 'rate':
            mab.update(chosen_arm, rate)
            rewards_per_day[day] = rate
        elif reward == 'profit':
            profit = orders * (1 - discount) * product_price
            mab.update(chosen_arm, profit)
            rewards_per_day[day] = profit
        # print(f"Rewards day {day}: {reward} for discount {discount}")

    return rewards_per_day, mab.values, mab.counts

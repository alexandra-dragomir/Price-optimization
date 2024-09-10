import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from simulator.multi_armed_bandits import EpsilonGreedy, UCB, simulate, orders_and_rate_per_day

NR_AGENTS = 17000
NR_AGENTS_MEAN = 17000
NR_AGENTS_STD = 5000
PROUDCT_PRICE = 300
PROUDCT_PRICE = 300


def plot_average_rewards(population_model, same_agents=False, agents=[], fixed_nr_agents=True, nr_agents=NR_AGENTS, \
                        mean_agents=NR_AGENTS_MEAN, std_agents=NR_AGENTS_STD, n_days=30, epsilons=[], confidences=[], \
                        num_runs=30, discounts=[0.3,0.5,0.8], reward='orders', product_price=PROUDCT_PRICE, plot_path_file="plot.png"):

    features_returned = []

    # simulations with random discount
    all_rewards_random = np.zeros((num_runs, n_days))
    all_values_random = []
    all_counts_random = []

    for run in tqdm(range(num_runs)):
        rewards, values, counts = simulate(population_model, same_agents=same_agents, agents=agents, fixed_nr_agents=fixed_nr_agents, nr_agents=nr_agents, \
                                            mean_agents=mean_agents, std_agents=std_agents, algorithm='epsilon_greedy', discounts=discounts, n_days=n_days, \
                                            epsilon=1, reward=reward, product_price=product_price)
        all_rewards_random[run] = rewards
        all_values_random.append(list(values))
        all_counts_random.append(list(counts))

    all_values_random = np.array(all_values_random)
    average_rewards_random = np.mean(all_rewards_random, axis=0)
    average_values_random = np.mean(all_values_random, axis=0)
    average_counts_random = np.mean(all_counts_random, axis=0)

    features_returned.append({'rewards_random':average_rewards_random, 'values_random':average_values_random, 'counts_random':average_counts_random})
    
    # simulations Epsilon Greedy 
    if len(epsilons) > 0:
        average_rewards_eps = np.zeros((len(epsilons), n_days))
        average_values_eps = []
        average_counts_eps = []

        for i, epsilon in enumerate(epsilons):
            all_rewards_eps = np.zeros((num_runs, n_days))
            all_values_eps = []
            all_counts_eps = []

            for run in tqdm(range(num_runs)):
                rewards, values, counts = simulate(population_model, same_agents=same_agents, agents=agents, fixed_nr_agents=fixed_nr_agents, nr_agents=nr_agents, \
                                            mean_agents=mean_agents, std_agents=std_agents, algorithm='epsilon_greedy', discounts=discounts, n_days=n_days, \
                                            epsilon=epsilon, reward=reward, product_price=product_price)
                all_rewards_eps[run] = rewards
                all_values_eps.append(list(values))
                all_counts_eps.append(list(counts))
            
            all_values_eps = np.array(all_values_eps)
            average_rewards_eps[i] = np.mean(all_rewards_eps, axis=0)
            average_values_eps.append(np.mean(all_values_eps, axis=0))
            average_counts_eps.append(np.mean(all_counts_eps, axis=0))

        features_returned.append({'rewards_eps':average_rewards_eps, 'values_eps':average_values_eps, 'counts_eps':average_counts_eps})


    
    # simulations UCB
    if len(confidences) > 0:
        average_rewards_ucb = np.zeros((len(confidences), n_days))
        average_values_ucb = []
        average_counts_ucb = []

        for i, confidence in enumerate(confidences):
            all_rewards_ucb = np.zeros((num_runs, n_days))
            all_values_ucb = []
            all_counts_ucb = []

            for run in tqdm(range(num_runs)):
                rewards, values, counts = simulate(population_model, same_agents=same_agents, agents=agents, fixed_nr_agents=fixed_nr_agents, nr_agents=nr_agents, \
                                            mean_agents=mean_agents, std_agents=std_agents, algorithm='UCB', discounts=discounts, n_days=n_days, \
                                            epsilon=confidence, reward=reward, product_price=product_price)
                all_rewards_ucb[run] = rewards
                all_values_ucb.append(list(values))
                all_counts_ucb.append(list(counts))
            
            all_values_ucb = np.array(all_values_ucb)
            average_rewards_ucb[i] = np.mean(all_rewards_ucb, axis=0)
            average_values_ucb.append(np.mean(all_values_ucb, axis=0))
            average_counts_ucb.append(np.mean(all_counts_ucb, axis=0))

        features_returned.append({'rewards_ucb':average_rewards_ucb, 'values_ucb':average_values_ucb, 'counts_ucb':average_counts_ucb})
    
    # Plotting
    fig = plt.figure(figsize=(14, 5))

    if len(epsilons) > 0:
        for i, epsilon in enumerate(epsilons):
            plt.plot(average_rewards_eps[i], label=f'epsilon = {epsilon}')
    if len(confidences) > 0:
        for i, confidence in enumerate(confidences):
            plt.plot(average_rewards_ucb[i], label=f'UCB, confidence = {confidence}')

    plt.plot(average_rewards_random, linestyle='dashed', label=f'random')

    plt.xlabel('Steps')
    plt.ylabel(f'Average Reward ({reward})')
    plt.title('Average Rewards Over Time for Different Epsilon Values')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path_file)
    plt.show()

    return features_returned
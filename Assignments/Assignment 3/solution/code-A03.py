import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt

# Create the FrozenLake environment 
env = gym.make("FrozenLake-v1", render_mode=None, is_slippery=False)

# Hyperparameters
GAMMA = 0.8   
ALPHA = 0.1     
EPSILON = 0.2     
NUM_EPISODES = 10000  

def monte_carlo_exploring_starts(env, num_episodes):
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize policy 
    policy = np.ones((num_states, num_actions)) / num_actions
    Q = np.zeros((num_states, num_actions))
    returns = {s: {a: [] for a in range(num_actions)} for s in range(num_states)}

    success_count = 0
    total_return = 0
    episode_returns = []

    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"MC Exploring Starts - Episode {episode}/{num_episodes}")

        # Exploring start
        state, _ = env.reset()
        action = random.randint(0, num_actions - 1)

        # episode trajectory
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_reward = 0

        done = False

        # First step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # generate episode
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)
        episode_reward += reward
        
        state = next_state

        # Continue episode
        while not done:
            action = np.random.choice(num_actions, p=policy[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_reward += reward
            
            state = next_state

      
        if episode_reward > 0:
            success_count += 1

        # Calculate returns and update policy
        G = 0
        visited_state_actions = set()
        
        for t in range(len(episode_states) - 1, -1, -1):
            state = episode_states[t]
            action = episode_actions[t]
            G = GAMMA * G + episode_rewards[t]

            if (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))
                returns[state][action].append(G)
                Q[state, action] = np.mean(returns[state][action])

                # greedy action
                best_action = np.argmax(Q[state])
                policy[state] = np.eye(num_actions)[best_action]

        total_return += episode_reward
        episode_returns.append(episode_reward)

    avg_return = total_return / num_episodes
    success_rate = (success_count / num_episodes) * 100

    return Q, policy, avg_return, success_rate, episode_returns

def monte_carlo_epsilon_soft(env, num_episodes, epsilon):
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize 
    policy = np.ones((num_states, num_actions)) / num_actions
    Q = np.zeros((num_states, num_actions))
    returns = {s: {a: [] for a in range(num_actions)} for s in range(num_states)}

    success_count = 0
    total_return = 0
    episode_returns = []

    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"MC Epsilon-Soft - Episode {episode}/{num_episodes}")

        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_reward = 0

        done = False

        while not done:
            #  epsilon-soft policy
            action = np.random.choice(num_actions, p=policy[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_reward += reward
            
            state = next_state

      
        if episode_reward > 0:
            success_count += 1

        # return G
        G = 0
        visited_state_actions = set()

        for t in range(len(episode_states) - 1, -1, -1):
            state = episode_states[t]
            action = episode_actions[t]
            G = GAMMA * G + episode_rewards[t]

            if (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))
                returns[state][action].append(G)
                Q[state, action] = np.mean(returns[state][action])

                #  epsilon-soft greedy
                best_action = np.argmax(Q[state])
                policy[state] = np.ones(num_actions) * (epsilon / num_actions)
                policy[state][best_action] += 1.0 - epsilon

        total_return += episode_reward
        episode_returns.append(episode_reward)

    avg_return = total_return / num_episodes
    success_rate = (success_count / num_episodes) * 100

    return Q, policy, avg_return, success_rate, episode_returns

def q_learning(env, num_episodes, alpha, epsilon):
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    success_count = 0
    total_return = 0
    episode_returns = []

    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"Q-Learning - Episode {episode}/{num_episodes}")

        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Epsilon-greedy
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Q-learning 
            best_next_action = np.argmax(Q[next_state])
            Q[state, action] += alpha * (reward + GAMMA * Q[next_state, best_next_action] - Q[state, action])

            state = next_state

        
        if episode_reward > 0:
            success_count += 1

        total_return += episode_reward
        episode_returns.append(episode_reward)

    avg_return = total_return / num_episodes
    success_rate = (success_count / num_episodes) * 100
    
    return Q, avg_return, success_rate, episode_returns

def sarsa(env, num_episodes, alpha, gamma, epsilon):
   
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    success_count = 0
    total_return = 0
    episode_returns = []

    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"SARSA - Episode {episode}/{num_episodes}")

        state, _ = env.reset()
        episode_reward = 0
        
        # Epsilon-greedy 
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Choose next action
            if random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            # SARSA 
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])


            state = next_state
            action = next_action

        # Record successful episodes
        if episode_reward > 0:
            success_count += 1

        total_return += episode_reward
        episode_returns.append(episode_reward)

    avg_return = total_return / num_episodes
    success_rate = (success_count / num_episodes) * 100
    
    return Q, avg_return, success_rate, episode_returns

def visualize_policy(Q, env_size=4):
    
    policy = np.argmax(Q, axis=1)
    
    # Direction mapping: 0=left, 1=down, 2=right, 3=up
    arrows = ['←', '↓', '→', '↑']
    
    policy_grid = np.empty((env_size, env_size), dtype=object)
    for s in range(env_size * env_size):
        i, j = s // env_size, s % env_size
        if s == (env_size * env_size) - 1:  # Goal state
            policy_grid[i, j] = 'G'
        else:
            policy_grid[i, j] = arrows[policy[s]]
    
    return policy_grid


def run_all_algorithms():
    
    results = {}
    
    print("\n--- Starting Monte Carlo Exploring Starts ---")
    start_time = time.time()
    mc_es_Q, mc_es_policy, mc_es_avg_return, mc_es_success_rate, mc_es_returns = monte_carlo_exploring_starts(env, NUM_EPISODES)
    mc_es_time = time.time() - start_time
    mc_es_policy_grid = visualize_policy(mc_es_Q)
    results["Monte Carlo ES"] = {
        "avg_return": mc_es_avg_return,
        "success_rate": mc_es_success_rate,
        "time": mc_es_time,
        "returns": mc_es_returns,
        "policy_grid": mc_es_policy_grid
    }
    
    print("\n--- Starting Monte Carlo Epsilon-Soft ---")
    start_time = time.time()
    mc_soft_Q, mc_soft_policy, mc_soft_avg_return, mc_soft_success_rate, mc_soft_returns = monte_carlo_epsilon_soft(env, NUM_EPISODES, EPSILON)
    mc_soft_time = time.time() - start_time
    mc_soft_policy_grid = visualize_policy(mc_soft_Q)
    results["Monte Carlo ε-Soft"] = {
        "avg_return": mc_soft_avg_return,
        "success_rate": mc_soft_success_rate,
        "time": mc_soft_time,
        "returns": mc_soft_returns,
        "policy_grid": mc_soft_policy_grid
    }
    
    print("\n--- Starting Q-Learning ---")
    start_time = time.time()
    q_learning_Q, q_learning_avg_return, q_learning_success_rate, q_learning_returns = q_learning(env, NUM_EPISODES, ALPHA, EPSILON)
    q_learning_time = time.time() - start_time
    q_learning_policy_grid = visualize_policy(q_learning_Q)
    results["Q-Learning"] = {
        "avg_return": q_learning_avg_return,
        "success_rate": q_learning_success_rate,
        "time": q_learning_time, 
        "returns": q_learning_returns,
        "policy_grid": q_learning_policy_grid
    }
    
    print("\n--- Starting SARSA ---")
    start_time = time.time()
    sarsa_Q, sarsa_avg_return, sarsa_success_rate, sarsa_returns = sarsa(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON)
    sarsa_time = time.time() - start_time
    sarsa_policy_grid = visualize_policy(sarsa_Q)
    results["SARSA"] = {
        "avg_return": sarsa_avg_return,
        "success_rate": sarsa_success_rate,
        "time": sarsa_time,
        "returns": sarsa_returns,
        "policy_grid": sarsa_policy_grid
    }
    
    return results



# Run all algorithms and collect results
print("Starting Reinforcement Learning Algorithm Comparison...")
results = run_all_algorithms()

# Print results
print("\n===== RESULTS SUMMARY =====")
print(f"Number of episodes: {NUM_EPISODES}")
print(f"Environment: FrozenLake-v1 (is_slippery=False)\n")

for algorithm, data in results.items():
    print(f"{algorithm}:")
    print(f"  Average Return: {data['avg_return']:.4f}")
    print(f"  Success Rate: {data['success_rate']:.2f}%")
    print(f"  Total Time: {data['time']:.2f} seconds\n")


import gymnasium as gym
import random
import time
import numpy as np

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", render_mode="human")  # "human" enables visualization
env_unwrapped = env.unwrapped
# Reset the environment
state, info = env.reset()

print("Initial State:", state)



# # Run a single episode
# done = False
# while not done:
#     time.sleep(1)
#     action = env.action_space.sample()  # Select a random action
#     next_state, reward, done, truncated, info = env.step(action)  # Take action
#
#     print(f"Action: {action}, New State: {next_state}, Reward: {reward}, Done: {done}")
#
# env.close()  # Close the rendering window

GAMMA = 0.99  # Discount factor
THRESHOLD = 1e-4  # Convergence threshold

# Initialize value function
V = np.zeros(env.observation_space.n)

# Value Iteration
while True:
    delta = 0
    new_V = np.copy(V)

    for s in range(env.observation_space.n):
        Q_values = []
        for a in range(env.action_space.n):
            q_value = sum([prob * (reward + GAMMA * V[next_state])
                           for prob, next_state, reward, _ in env_unwrapped.P[s][a]])
            Q_values.append(q_value)

        new_V[s] = max(Q_values)
        delta = max(delta, abs(new_V[s] - V[s]))

    V = new_V
    if delta < THRESHOLD:
        break

# Extract optimal policy
policy = np.zeros(env.observation_space.n, dtype=int)
for s in range(env.observation_space.n):
    Q_values = [sum([prob * (reward + GAMMA * V[next_state])
                     for prob, next_state, reward, _ in env_unwrapped.P[s][a]])
                for a in range(env.action_space.n)]
    policy[s] = np.argmax(Q_values)

print("Optimal Policy:", policy.reshape(4, 4))  # Display in grid format
# Run a single episode
state, info = env.reset()
done = False
while not done:
    time.sleep(1)
    action = policy[state]
    next_state, reward, done, truncated, info = env.step(action)  # Take action

    print(f"Action: {action}, New State: {next_state}, Reward: {reward}, Done: {done}")
    state = next_state
env.close()  # Close the rendering window

import numpy as np
from taxienv import Taxi6x6Env

env = Taxi6x6Env()

num_states = env.observation_space.n
num_actions = env.action_space.n

Q = np.zeros((num_states, num_actions))

alpha = 0.1      # öğrenme oranı
gamma = 0.99     # discount
epsilon = 1.0    # başlangıç exploration
epsilon_min = 0.05
epsilon_decay = 0.999  # yavaşça azalsın, daha çok keşif olsun

num_episodes = 8000
max_steps = 200

rewards_history = []

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    for t in range(max_steps):
        mask = info["action_mask"]
        valid_actions = np.where(mask == 1)[0]

        if len(valid_actions) == 0:
            action = env.action_space.sample()
        else:
            if np.random.rand() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                q_vals = Q[state, valid_actions]
                action = valid_actions[np.argmax(q_vals)]

        next_state, reward, terminated, truncated, info = env.step(action)

        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_history.append(total_reward)

    if (episode + 1) % 500 == 0:
        avg_last = np.mean(rewards_history[-100:])
        print(
            f"Episode {episode+1}/{num_episodes} - total_reward={total_reward:.1f} - "
            f"avg_last_100={avg_last:.2f} - epsilon={epsilon:.3f}"
        )

env.close()

np.save("q_table_taxi6x6.npy", Q)
print("Eğitim bitti, Q tablosu kaydedildi: q_table_taxi6x6.npy")

import numpy as np
import matplotlib.pyplot as plt
import imageio

from taxienv import Taxi6x6Env


def draw_frame(env, state):
    taxi_row, taxi_col, pass_loc, dest_idx = env.decode(state)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-0.5, env.num_cols - 0.5)
    ax.set_ylim(-0.5, env.num_rows - 0.5)
    ax.invert_yaxis()

    # Grid çizgileri
    for x in range(env.num_cols + 1):
        ax.axvline(x - 0.5)
    for y in range(env.num_rows + 1):
        ax.axhline(y - 0.5)

    # DUVAR (blocked cells) gri kutu olarak
    for (r, c) in env.blocked_cells:
        rect = plt.Rectangle(
            (c - 0.5, r - 0.5),
            1,
            1,
            facecolor="gray",
            alpha=0.7,
        )
        ax.add_patch(rect)

    # Lokasyonlar (R,G,Y,B)
    colors = ["red", "green", "purple", "blue"]
    labels = ["o", "o", "o", "o"]
    for idx, (r, c) in enumerate(env.locs):
        ax.scatter(c, r, s=300, marker="s", edgecolor="black", facecolor=colors[idx])
        ax.text(c, r, labels[idx], ha="center", va="center", fontsize=12, fontweight="bold")

    # Taksi
    ax.scatter(taxi_col, taxi_row, s=300, marker="o", color="yellow", zorder=3)
    ax.text(taxi_col, taxi_row, "T", color="white", ha="center", va="center", fontsize=12)

    # Yolcu (takside değilse)
    if pass_loc < 4:
        pr, pc = env.locs[pass_loc]
        ax.scatter(pc, pr, s=180, marker="o", edgecolor="black", facecolor="white", zorder=3)
        ax.text(pc, pr, "P", ha="center", va="center", fontsize=10)

    # Hedef (destination)
    dr, dc = env.locs[dest_idx]
    ax.scatter(dc, dr, s=260, marker="*", edgecolor="purple", facecolor="none", zorder=3)
    ax.text(dc, dr, "D", color="purple", ha="center", va="center", fontsize=10)

    ax.axis("off")
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame


def run_many_episodes_and_collect_frames(env, Q, num_episodes=8, max_steps_per_ep=60):
    frames = []

    for ep in range(num_episodes):
        state, info = env.reset()

        for t in range(max_steps_per_ep):
            frame = draw_frame(env, state)
            frames.append(frame)

            mask = env.action_mask(state)
            valid_actions = np.where(mask == 1)[0]

            # Görselde de çok saçma gezmesin diye burada artık greedy politika
            if len(valid_actions) == 0:
                action = 0
            else:
                q_vals = Q[state, valid_actions]
                action = valid_actions[np.argmax(q_vals)]

            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state

            if terminated or truncated:
                break

    return frames


if __name__ == "__main__":
    env = Taxi6x6Env()
    Q = np.load("q_table_taxi6x6.npy")

    frames = run_many_episodes_and_collect_frames(
        env,
        Q,
        num_episodes=8,
        max_steps_per_ep=70,
    )

    imageio.mimsave("taxi6x6_policy.gif", frames, fps=3)
    print("GIF oluşturuldu: taxi6x6_policy.gif")

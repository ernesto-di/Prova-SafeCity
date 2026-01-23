import time
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from agent.agent import Agent
from env.traffic_env import TrafficEnv
from render.dashboard import TrainingDashboard


def train():
    agent = Agent(input_dim=32, output_dim=4)
    env = TrafficEnv(agent.policy_net)
    dashboard = TrainingDashboard(env)
    env.dashboard = dashboard

    num_episodes = 100
    target_cars = config.TARGET_FINISHED_CARS
    os.makedirs("output", exist_ok=True)

    episode_rewards = []

    try:
        for ep in range(num_episodes):
            start = time.time()
            state = env.reset()
            total_reward = 0.0
            steps = 0

            while env.cars_finished < target_cars:
                action = agent.select_action(state)
                next_state, reward, _, _ = env.step(action)

                dashboard.draw()

                agent.memory.push(state, action, reward, next_state, False)
                agent.optimize_model()

                state = next_state
                total_reward += reward
                steps += 1

                if steps % 10 == 0:
                    print(
                        f"[Ep {ep:03d} Step {steps:04d} | "
                        f"Reward: {reward:+.3f} | "
                        f"TotReward: {total_reward:+.2f} | "
                        f"Cars: {env.cars_finished}"
                    )

                if steps > 1000:
                    print("WARNING: max steps reached")
                    break

            agent.step_epsilon()
            eps = agent.epsilon
            episode_rewards.append(total_reward)
            print(
                f"✅ Episodio {ep} concluso | "
                f"TotalReward: {total_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Steps: {steps}"
            )

            m, s = divmod(time.time() - start, 60)
            print(f"Ep {ep} | Reward {total_reward:.2f} | ε {eps:.2f} | {int(m)}m{int(s)}s")

            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid()
            plt.savefig("output/training_rewards.png")
            plt.close()

    except KeyboardInterrupt:
        print("Training interrotto")

    finally:
        torch.save(agent.policy_net.state_dict(), "output/modello.pth")
        dashboard.close()
        print("Training terminato")


if __name__ == "__main__":
    train()

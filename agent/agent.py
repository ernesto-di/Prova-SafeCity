import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from network.network import DualHeadDQN
from agent.replay_buffer import ReplayBuffer
from agent.exploration import EpsilonScheduler


class Agent:

    def __init__(self, input_dim, output_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Networks ---
        self.policy_net = DualHeadDQN(input_dim, output_dim).to(self.device)
        self.target_net = DualHeadDQN(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # --- Optimizer ---
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

        # --- Replay & exploration ---
        self.memory = ReplayBuffer(10000)
        self.exploration = EpsilonScheduler()

        # --- Hyperparams ---
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = self.exploration.epsilon

    # ==================================================================
    # ACTION MASKING (EMERGENZE)
    # ==================================================================
    def _extract_action_mask(self, state):
        masks = []
        idx = 0

        for _ in range(2):  # 2 incroci
            emerg = state[idx+8:idx+12]
            idx += 16

            if sum(emerg) > 0:
                mask = [0, 0, 0, 0]
                mask[np.argmax(emerg)] = 1
            else:
                mask = [1, 1, 1, 1]

            masks.append(torch.tensor(mask, dtype=torch.float32))

        return masks

    # ==================================================================
    # ACTION SELECTION (ε-greedy + masking)
    # ==================================================================
    def select_action(self, state):
        masks = self._extract_action_mask(state)

        if random.random() < self.epsilon:
            a0 = random.choice([i for i, v in enumerate(masks[0]) if v == 1])
            a1 = random.choice([i for i, v in enumerate(masks[1]) if v == 1])
            return a0, a1

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            out0, out1 = self.policy_net(state_t)

            q0 = out0.squeeze(0) * masks[0].to(self.device)
            q1 = out1.squeeze(0) * masks[1].to(self.device)

            return q0.argmax().item(), q1.argmax().item()

    # ==================================================================
    # DOUBLE DQN OPTIMIZATION
    # ==================================================================
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state_batch = torch.from_numpy(np.array(state)).float().to(self.device)
        next_state_batch = torch.from_numpy(np.array(next_state)).float().to(self.device)
        reward_batch = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        action0 = torch.LongTensor([a[0] for a in action]).unsqueeze(1).to(self.device)
        action1 = torch.LongTensor([a[1] for a in action]).unsqueeze(1).to(self.device)

        q0, q1 = self.policy_net(state_batch)
        q_val0 = q0.gather(1, action0)
        q_val1 = q1.gather(1, action1)

        with torch.no_grad():
            pol0, pol1 = self.policy_net(next_state_batch)

            mask0 = torch.stack([self._extract_action_mask(s)[0] for s in next_state]).to(self.device)
            mask1 = torch.stack([self._extract_action_mask(s)[1] for s in next_state]).to(self.device)

            next_a0 = (pol0 * mask0).argmax(dim=1, keepdim=True)
            next_a1 = (pol1 * mask1).argmax(dim=1, keepdim=True)

            tgt0, tgt1 = self.target_net(next_state_batch)
            next_q0 = tgt0.gather(1, next_a0)
            next_q1 = tgt1.gather(1, next_a1)

        exp_q0 = reward_batch + (1 - done_batch) * self.gamma * next_q0
        exp_q1 = reward_batch + (1 - done_batch) * self.gamma * next_q1

        loss = F.mse_loss(q_val0, exp_q0) + F.mse_loss(q_val1, exp_q1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ==================================================================
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def step_epsilon(self):
        self.epsilon = self.exploration.step()
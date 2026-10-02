from game import Game, BOMB
from collections import namedtuple, deque
import numpy as np
import torch
import random 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

GAMMA = 0.95
LEARNING_RATE = 0.001
BATCH_SIZE = 64 
BUFFER_CAPACITY = 20000
LEARNING_STARTS = 1000   # BUFFER SIZE WHICH UPDATES BEGIN
TARGET_NETWORK_UPDATE_INTERVAL = 500
EPSILON_START= 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 20000
TOTAL_TRAINING_STEPS = 100000
GRADIENT_CLIP_NORM = 10
SEED= 42
NEGATIVE_INF = -10**9

GRID_SIZE = 6

class environment():
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.cell_count = self.grid_size ** 2
        self.game = Game( grid_size=self.grid_size, bomb_count=11, n_players=1)
        self.step_count = 0

    def observe(self):
        hidden = [1] *  self.cell_count
        bomb_found = [0] *  self.cell_count
        neighbor_count = [0] * self.cell_count
        for (row,col), val in self.game.revealed.items():
            idx = row*self.grid_size + col
            hidden[idx] = 0
            if val == BOMB:
                bomb_found[idx] = 1
            else:
                neighbor_count[idx] = val / 8
        legal_mask =  np.array(hidden).astype(bool)
        return np.stack([
            np.array(hidden).reshape(self.grid_size,self.grid_size),
            np.array(bomb_found).reshape(self.grid_size,self.grid_size),
            np.array(neighbor_count).reshape(self.grid_size,self.grid_size)
        ], axis = 0, dtype=np.float32) , legal_mask
    
    def reset(self):
        self.game.reset()
        self.step_count = 0
        return self.observe()


    def step(self, action_cell):
        row, col = action_cell // self.grid_size , action_cell % self.grid_size
        result = self.game.pick(row, col)
        if not result['ok']:
            raise ValueError("invalid move")
        self.step_count += 1
        reward = 1 if result['is_bomb'] else 0
        observable, mask = self.observe()
        return observable, mask , reward, result['done'] 

class DQN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32 , kernel_size=3 , padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64 , kernel_size=3 , padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

Transition = namedtuple('Transition',
                        ('observe','action','reward','next_observe','next_legal_mask','is_done'))

class ReplayBuffer():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        obs, action, reward, next_obs, next_legal_mask, is_done = zip(*random.sample(self.memory, batch_size))
        return torch.tensor(np.array(obs),dtype=torch.float32) , \
            torch.tensor(action, dtype=torch.long), \
            torch.tensor(reward,dtype=torch.float32) , \
            torch.tensor(np.array(next_obs),dtype=torch.float32), \
            torch.tensor(np.array(next_legal_mask),dtype=torch.bool), \
             torch.tensor(is_done,dtype=torch.float32)

    def __len__(self):
        return len(self.memory)

class Learning_step():
    def __init__(self, main_network, target_network, optimizer, gamma, device):
        self.main_network = main_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device

    def predict(self,batch):
        obs,action,_,_,_,_ = batch
        obs = obs.to(self.device)
        action = action.to(self.device)
        model = self.main_network.to(self.device)

        pred = model(obs)
        q_pred = torch.gather(pred, dim=1, index= action.unsqueeze(-1)).squeeze(1)
        return q_pred

    def compute_target(self, batch):
        _,_,reward,next_obs,next_mask,is_done = batch

        reward, next_obs, next_mask, is_done = reward.to(self.device), next_obs.to(self.device),  next_mask.to(self.device), is_done.to(self.device)
        model = self.target_network.to(self.device)
        
        with torch.no_grad():
            pred = model(next_obs)
            pred_masked = torch.where(next_mask, pred, NEGATIVE_INF)
            max_val = torch.max(pred_masked, dim=1).values
            q_target = reward + self.gamma * max_val * (1-is_done)

        return q_target

    def update(self, batch):
        q_pred, q_target = self.predict(batch), self.compute_target(batch)
        
        loss = F.smooth_l1_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), GRADIENT_CLIP_NORM)
        self.optimizer.step()
        
        return loss.item()

def epsilon_schedule(step):
    epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * step / EPSILON_DECAY_STEPS)
    return epsilon

def select_action(network, observation, legal_mask, epsilon, device):
    if random.random() < epsilon:
        true_ids = [idx for idx,is_legal in enumerate(legal_mask) if is_legal] 
        return random.choice(true_ids)
    batch = torch.tensor(observation).unsqueeze(0).to(device)
    network = network.to(device)
    with torch.no_grad():
        pred = network(batch)
        pred_masked = torch.where(torch.tensor(legal_mask).to(device), pred, NEGATIVE_INF)
    return torch.argmax(pred_masked).item()
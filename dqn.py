from game import Game, BOMB
from collections import namedtuple, deque
import numpy as np
import torch
import random 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class environment():
    def __init__(self):
        self.grid_size = 6
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
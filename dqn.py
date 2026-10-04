from game import Game, BOMB
from collections import namedtuple, deque
import numpy as np
import torch
import random 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.signal import convolve2d

GAMMA = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 64 
BUFFER_CAPACITY = 20000
LEARNING_STARTS = 1000   # BUFFER SIZE WHICH UPDATES BEGIN
TARGET_NETWORK_UPDATE_INTERVAL = 500
EPSILON_START= 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 10000
TOTAL_TRAINING_STEPS = 30000
GRADIENT_CLIP_NORM = 10
SEED= 42
NEGATIVE_INF = -10**9

GRID_SIZE = 6
CELL_COUNT = GRID_SIZE ** 2
BOMB_COUNT = 11

INPUT_CHANNEL = 7

EVAL_GAME = 500
EVAL_SEED = 12345
EVAL_INTERVAL = 4000


class environment():
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.cell_count = CELL_COUNT
        self.game = Game( grid_size=self.grid_size, bomb_count=BOMB_COUNT, n_players=1)
        self.step_count = 0

    def observe(self):
        hidden = [1] *  self.cell_count
        bomb_found = [0] *  self.cell_count
        count = [0] * self.cell_count

        for (row,col), val in self.game.revealed.items():
            idx = row*self.grid_size + col
            hidden[idx] = 0
            if val == BOMB:
                bomb_found[idx] = 1
            else:
                count[idx] = val 
        
        hidden = np.array(hidden).reshape(self.grid_size,self.grid_size)
        bomb_found = np.array(bomb_found).reshape(self.grid_size,self.grid_size)
        count = np.array(count).reshape(self.grid_size,self.grid_size)
        legal_mask =  hidden.astype(bool).flatten()
        shown_empty = (1-hidden) * (1-bomb_found)
        kernel = np.array([
            [1,1,1],
            [1,0,1],
            [1,1,1]
        ])
        revealed_bomb_neighbor = convolve2d(bomb_found, kernel, mode='same', boundary='fill', fillvalue=0)
        remaining = np.where(shown_empty == 1, np.maximum(0, count - revealed_bomb_neighbor), 0)
        hidden_neighbor = convolve2d(hidden, kernel, mode='same', boundary='fill', fillvalue=0)
        ratio = np.where((shown_empty ==1)&(remaining>0), np.minimum(1, remaining / np.maximum(hidden_neighbor,1)), 0)

        density = np.full((self.grid_size, self.grid_size), (BOMB_COUNT - np.sum(bomb_found)) / max(np.sum(hidden),1))

        return np.stack([
            hidden,             # 1 for hidden, 0 for shown
            bomb_found,         # is it revealed as a bomb, 1 for bomb
            count / 8,          # revealed value when empty (not a bomb)
            remaining / 8,      # remaining bomb in neighbor hidden cell
            hidden_neighbor / 8,    # count of hidden neighbor
            ratio,              # remaining neighbor bomb count divide by hidden neighbor count
            density             # global bomb to hidden cell ration, a constant for all cells
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
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(INPUT_CHANNEL, 64 , kernel_size=3 , padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64 , kernel_size=3 , padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64 , kernel_size=3 , padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64 , kernel_size=3 , padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1 , kernel_size=1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.network(x)

Transition = namedtuple('Transition',
                        ('observe','action','reward','next_observe','next_legal_mask','is_done'))

class Symmetric():
    def __init__(self, grid_size):
        self.base = np.arange(grid_size**2).reshape(grid_size, grid_size)
        self.table, self.reverse_table = self.eight_symmetry(self.base) # each table is (8, N*N)

    def eight_symmetry(self, grid):
        rot90, rot180, rot270 = np.rot90(grid).flatten(), np.rot90(grid, k=2).flatten(), np.rot90(grid, k=3).flatten()
        flipped = np.fliplr(grid)
        flipped_rot90,flipped_rot180,flipped_rot270  = np.rot90(flipped).flatten(), np.rot90(flipped,k=2).flatten(), np.rot90(flipped,k=3).flatten()
        table = torch.tensor(
            np.stack([
            grid.flatten(), rot90, rot180, rot270, \
            flipped.flatten(), flipped_rot90, flipped_rot180, flipped_rot270
            ])
        )
        return table, table.argsort(dim=-1) 

    def reorder(self, batch, symmetry_ids):      # batch is (batch, channel, n, n)
        flat_batch = torch.flatten(batch,start_dim=2,end_dim=3)     # (batch, channel, n*n)
        table = self.table[symmetry_ids]                     # (batch, n*n)
        sample_with_channel = table.unsqueeze(1)             # (batch, 1, n*n)
        expanded = sample_with_channel.expand_as(flat_batch)        # (batch, channel, n*n) 
        reordered = torch.gather(flat_batch, dim=2, index=expanded)      # (batch, channel, n*n)
        reshaped = reordered.reshape_as(batch)
        return reshaped

    def reorder_mask(self, mask, symmetry_ids):
        table = self.table[symmetry_ids]
        return torch.gather(mask, dim=1, index=table)

    def map_action(self, actions, symmetry_ids):
        return self.reverse_table[symmetry_ids, actions]

    def augment(self, batch, symmetry_ids=None):
        obs, action, reward, next_obs, next_mask, is_done =  batch
        if symmetry_ids is None:
            symmetry_ids = torch.randint(0, 8, (obs.shape[0],))    # 8 is size of symmetric map (across 8 tranformation)
        reorder_obs = self.reorder(obs, symmetry_ids)
        reorder_next_obs = self.reorder(next_obs, symmetry_ids)
        reorder_next_mask = self.reorder_mask(next_mask, symmetry_ids)
        map_act = self.map_action(action, symmetry_ids)
        return reorder_obs, map_act, reward, reorder_next_obs, reorder_next_mask, is_done 

class ReplayBuffer():
    def __init__(self, capacity, symmetric=None):
        self.memory = deque([], maxlen=capacity)
        self.symmetric= symmetric

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        obs, action, reward, next_obs, next_legal_mask, is_done = zip(*random.sample(self.memory, batch_size))
        batch = torch.tensor(np.array(obs),dtype=torch.float32) , \
            torch.tensor(action, dtype=torch.long), \
            torch.tensor(reward,dtype=torch.float32) , \
            torch.tensor(np.array(next_obs),dtype=torch.float32), \
            torch.tensor(np.array(next_legal_mask),dtype=torch.bool), \
             torch.tensor(is_done,dtype=torch.float32)
        if self.symmetric is not None:
            batch = self.symmetric.augment(batch)
        return batch
    
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
        target_model = self.target_network.to(self.device)
        main_model = self.main_network.to(self.device)
        
        with torch.no_grad():
            action = main_model(next_obs)   # (batch, n*n)
            action_masked = torch.where(next_mask, action, NEGATIVE_INF)
            max_ids = torch.argmax(action_masked, dim=1, keepdim=True)    # (batch, 1)
            pred = target_model(next_obs)   # (batch, n*n)
            chosen = torch.gather(pred, dim=1, index=max_ids).squeeze(1)    # (batch,)
            q_target = reward + self.gamma * chosen * (1-is_done)

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

if __name__ == '__main__':

    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"device: {device}")

    env = environment()
    main_network = DQN()    
    target_network = DQN()
    target_network.load_state_dict(main_network.state_dict())
    optimizer = optim.Adam(main_network.parameters(), lr=LEARNING_RATE)
    symmetric = Symmetric(GRID_SIZE)
    buffer = ReplayBuffer(BUFFER_CAPACITY, symmetric)
    learning_step = Learning_step(main_network, target_network, optimizer, GAMMA, device)
    episode_length = []
    loss = 0 # init

    observe, mask = env.reset()

    episode_average = 36    # save best 100 episode average so far

    for step in range(TOTAL_TRAINING_STEPS):
        
        epsilon = epsilon_schedule(step)
        action = select_action(main_network, observe, mask, epsilon, device)
        next_observe, mask, reward, is_done = env.step(action)
        buffer.push(Transition(observe, action, reward, next_observe, mask, is_done))
        observe = next_observe

        if len(buffer) >= LEARNING_STARTS:
            batch = buffer.sample(BATCH_SIZE)
            loss = learning_step.update(batch)

            if step % TARGET_NETWORK_UPDATE_INTERVAL == 0:
                target_network.load_state_dict(main_network.state_dict())

        if is_done:
            step_count = env.step_count
            episode_length.append(step_count)
            if len(episode_length) % 100 == 0:
                
                new_avg = np.mean(episode_length[-100:]) 
                if new_avg < episode_average:
                    torch.save(main_network.state_dict(), 'weight/best_dqn.pt')
                    episode_average = new_avg

                print(f"global step: {step} \nstep count: {step_count} \naverage step count: {new_avg} " + \
                    f"epsilon: {epsilon:.2f} loss: {loss:.2f} \n")
            observe, mask = env.reset()

    torch.save(main_network.state_dict(), 'weight/final_dqn.pt')
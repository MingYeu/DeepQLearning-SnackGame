import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from config import device  


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        folder_path = './model'  # folder to store the model file
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # create the folder if it doesn't exist

        file_path = os.path.join(folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")
        
    def load(self, file_name):
        # self.load_state_dict(torch.load(file_name))
        # self.eval()  # good practice when loading models for inference
        
         # Move tensors to device
        self.load_state_dict(torch.load(file_name, map_location=device))
        self.eval()
        self.to(device)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # If only 1 dimension, unsqueeze
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

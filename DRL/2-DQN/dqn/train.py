import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque
import torchvision.transforms as transforms
from ConvNet import ConvNet
from Downsample2xFullY import Downsample2xFullY
from ReLU import Rectifier
from Scale import Scale

class EnvSetup:
    def __init__(self, env, agent_params, gpu=-1, verbose=10):
        self.env = env
        self.agent_params = agent_params
        self.gpu = gpu
        self.verbose = verbose
        
        if self.gpu >= 0:
            self.device = torch.device(f"cuda:{self.gpu}")
        else:
            self.device = torch.device("cpu")
        
        if self.verbose >= 1:
            print(f"Using device: {self.device}")
    
    def reset_environment(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)

class TrainAgent:
    def __init__(self, env_setup, agent, steps=100000, eval_freq=10000, save_freq=50000):
        self.env_setup = env_setup
        self.agent = agent
        self.steps = steps
        self.eval_freq = eval_freq
        self.save_freq = save_freq
    
    def train(self):
        print("Starting training...")
        state = self.env_setup.reset_environment()
        for step in range(1, self.steps + 1):
            action = self.agent.select_action(state)
            next_state, reward, done, _ = self.env_setup.step(action)
            self.agent.memory.add(state, action, reward, next_state, done)
            self.agent.train_step()
            state = next_state if not done else self.env_setup.reset_environment()
            
            if step % self.eval_freq == 0:
                print(f"Step {step}: Evaluating agent...")
                self.agent.evaluate()
            
            if step % self.save_freq == 0:
                print(f"Step {step}: Saving model...")
                torch.save(self.agent.model.state_dict(), f"agent_step_{step}.pt")
        
        print("Training completed!")


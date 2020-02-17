import multiprocessing
import multiprocessing.connection
import time
from collections import deque
from typing import Dict, List
import torch
import cv2
import gym
import numpy as np

def obs_to_torch(obs,device):
    #[N, H, W, C] to [N, C, H, W]

    obs = np.swapaxes(obs, 1, 3)
    obs = np.swapaxes(obs, 3, 2)
    #scale to [0, 1]

    return torch.tensor(obs, dtype=torch.float32, device=device) / 255.

class Game:
    def __init__(self, seed: int):
        self.env = gym.make('BreakoutNoFrameskip-v4')
        self.env.seed(seed)
        self.seed=seed
        self.obs_2_max = np.zeros((2, 84, 84, 1), np.uint8)
        self.obs_4 = np.zeros((84, 84, 4))
        self.rewards = []
        self.lives = 0
    
    def step(self, action):
        reward = 0.
        done = None
        for i in range(4):
            obs, r, done, info = self.env.step(action)
            if i >= 2:
                self.obs_2_max[i % 2] = self._process_obs(obs)
                if self.seed==47:
                    self.env.render()

            reward += r
            lives = self.env.unwrapped.ale.lives()
            if lives < self.lives:
                done = True
            self.lives = lives
            if done:
                break
        self.rewards.append(reward)
        if done:
            episode_info = {"reward": sum(self.rewards),
                        "length": len(self.rewards)}
            self.reset()
        else:
            episode_info = None
            obs = self.obs_2_max.max(axis=0)
            self.obs_4 = np.roll(self.obs_4, shift=-1, axis=-1)
            self.obs_4[..., -1:] = obs
        return self.obs_4, reward, done, episode_info

    def reset(self):
        obs = self.env.reset()
        obs = self._process_obs(obs)
        self.obs_4[..., 0:] = obs
        self.obs_4[..., 1:] = obs
        self.obs_4[..., 2:] = obs
        self.obs_4[..., 3:] = obs
        self.rewards = []

        self.lives = self.env.unwrapped.ale.lives()

        return self.obs_4
    @staticmethod
    def _process_obs(obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs[:, :, None]  # Shape (84, 84, 1)

def worker_process(remote: multiprocessing.connection.Connection, seed: int):
    game = Game(seed)
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(game.step(data))
        elif cmd == "reset":
            remote.send(game.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError

class Worker:
    child: multiprocessing.connection.Connection
    process: multiprocessing.Process  
    def __init__(self, seed):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, seed))
        self.process.start()
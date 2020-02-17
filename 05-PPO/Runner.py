import time
from collections import deque


#import cv2
#import gym
import numpy as np
import torch

from Trainer import *
from Game import *
from Model import *

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

    
class Main(object):
    def __init__(self):
        self.gamma = 0.99
        self.lamda = 0.95
        self.updates = 10000
        self.epochs = 4
        self.n_workers = 8
        self.worker_steps = 256
        self.n_mini_batch = 4
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        assert (self.batch_size % self.n_mini_batch == 0)
        self.workers = [Worker(47 + i) for i in range(self.n_workers)]
        self.obs = np.zeros((self.n_workers, 84, 84, 4), dtype=np.uint8)
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()
        self.model = Model()
        self.model.to(device)
        self.trainer = Trainer(self.model)
        
    def sample(self) -> (Dict[str, np.ndarray], List):
        rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        dones = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        obs = np.zeros((self.n_workers, self.worker_steps, 84, 84, 4), dtype=np.uint8)
        neg_log_pis = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        values = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        episode_infos = []
        for t in range(self.worker_steps):
            obs[:, t] = self.obs
            pi, v = self.model(obs_to_torch(self.obs,device))
            values[:, t] = v.cpu().data.numpy()
            a = pi.sample()
            actions[:, t] = a.cpu().data.numpy()
            neg_log_pis[:, t] = -pi.log_prob(a).cpu().data.numpy()
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w, t]))

            for w, worker in enumerate(self.workers):
                self.obs[w], rewards[w, t], dones[w, t], info = worker.child.recv()
                if info:
                    info['obs'] = obs[w, t, :, :, 3]
                    episode_infos.append(info)
            advantages = self._calc_advantages(dones, rewards, values)
            samples = {
                'obs': obs,
                'actions': actions,
                'values': values,
                'neg_log_pis': neg_log_pis,
                'advantages': advantages
            }        

        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = obs_to_torch(v,device)
            else:
                samples_flat[k] = torch.tensor(v, device=device)

        return samples_flat, episode_infos
    def _calc_advantages(self, dones: np.ndarray, rewards: np.ndarray,
                         values: np.ndarray) -> np.ndarray:
        advantages = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        last_advantage = 0
        _, last_value = self.model(obs_to_torch(self.obs,device))
        last_value = last_value.cpu().data.numpy()
        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - dones[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = rewards[:, t] + self.gamma * last_value - values[:, t]
            last_advantage = delta + self.gamma * self.lamda * last_advantage
            advantages[:, t] = last_advantage
            last_value = values[:, t]
        return advantages
    def train(self, samples: Dict[str, np.ndarray], learning_rate: float, clip_range: float):
        train_info = []
        for _ in range(self.epochs):
            indexes = torch.randperm(self.batch_size)
            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]
                res = self.trainer.train(learning_rate=learning_rate,
                                         clip_range=clip_range,
                                         samples=mini_batch)
                train_info.append(res)
                return np.mean(train_info, axis=0)

    def run_training_loop(self):
        episode_info = deque(maxlen=100)

        for update in range(self.updates):
            time_start = time.time()
            progress = update / self.updates
            learning_rate = 2.5e-4 * (1 - progress)
            clip_range = 0.1 * (1 - progress)
            samples, sample_episode_info = self.sample()
            self.train(samples, learning_rate, clip_range)
            time_end = time.time()
            fps = int(self.batch_size / (time_end - time_start))
            episode_info.extend(sample_episode_info)
            reward_mean, length_mean = Main._get_mean_episode_info(episode_info)
            print(f"{update:4}: fps={fps:3} reward={reward_mean:.2f} length={length_mean:.3f}")
            if update%10==9:
                self.model.save()

    
    @staticmethod
    def _get_mean_episode_info(episode_info):
        if len(episode_info) > 0:
            return (np.mean([info["reward"] for info in episode_info]),
                    np.mean([info["length"] for info in episode_info]))
        else:
            return np.nan, np.nan
    def destroy(self):
        for worker in self.workers:
            worker.child.send(("close", None))
    
if __name__ == "__main__":
    m = Main()
    m.run_training_loop()
    m.destroy()



























        













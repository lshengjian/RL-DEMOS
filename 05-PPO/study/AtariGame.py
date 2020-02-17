from gym.wrappers import FrameStack,AtariPreprocessing,Monitor
import matplotlib.pyplot as plt

from gym.spaces import Box
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
#Hyperparameters
learning_rate = 0.001
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20
def set_lr(optimizer):
    for param_group in optimizer.param_groups:
        lr=param_group['lr']
        if lr>0.0000001:
            lr*=0.98
        param_group['lr'] = lr
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer
class MyNet(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.data=[]
        self.feature_dim = 512
        self.action_dim = 4
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.fc_action = layer_init(nn.Linear(self.feature_dim, self.action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(self.feature_dim, 1), 1e-3)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to('cpu')

        #self.params = list(self.phi_body.parameters())

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        logits = self.fc_action(y)
        v = self.fc_critic(y)
        #dist = torch.distributions.Categorical(logits=logits)
        prob=F.softmax(logits, dim=1)
        #action = dist.sample()
        #log_prob = dist.log_prob(action).unsqueeze(-1)
        #entropy = dist.entropy().unsqueeze(-1)
        return {'pi': prob,
                #'log_pi_a': log_prob,
                #'ent': entropy,
                'v': v}

    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        out1=self.forward(s)
        out2=self.forward(s_prime)
        pi=out1['pi']
        v=out1['v']
        set_lr(self.optimizer)

        for i in range(K_epoch):
            td_target = r + gamma * out2['v'] * done_mask
            delta = td_target - v
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v , td_target.detach())

            
            self.optimizer.zero_grad()
            loss.mean().backward( retain_graph=True )
            self.optimizer.step()
    


if __name__ == '__main__':
    #from gym import envs
    #print(envs.registry.all())
    #SpaceInvadersNoFrameskip-v4,MsPacman-NoFrameskip-v4,BreakoutNoFrameskip-v4
    #env = gym.make('Breakout-v0').unwrapped
    model=MyNet()
    env = gym.make('SpaceInvadersNoFrameskip-v4')
    env=AtariPreprocessing(env,frame_skip=4,scale_obs=True)
    shape = (-1,4,84,84)
    shape0 = (4,84,84)
    
    num_stack=4
    
    agent=RandomAgent(env.action_space)

    #env = monitor = Monitor(env, 'data/monitor',force=True, video_callable=lambda i: i % 1 != 0 )
    env = FrameStack(env, num_stack, False)

    episode_count = 1
    reward = 0
    done = False
    score = 0.0
    print_interval = 3

    for n_epi in range(100):
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                env.render()
                s=np.array(s).reshape(shape)

                od=model(torch.from_numpy(s).float())
                prob = od['pi']
                #print(prob)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                trn=(s.reshape(shape0), a, r/100.0, np.array(s_prime), prob[0][a].item(), done)
                model.put_data(trn)
                s = s_prime
                score += r
                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()



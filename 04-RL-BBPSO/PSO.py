import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
N_THREADS=20#mp.cpu_count()*5

MAX_SCORE=300
MAX_EP=8

EPS=1e-20

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)


    def forward(self, x, softmax_dim=0):
        x = F.relu6(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    def save(self):
        torch.save(self.state_dict(), 'data.pk')
    def load(self):
        self.load_state_dict(torch.load('data.pk'))

class Worker(mp.Process):
    def __init__(self, gnet, gbest,res_queue, idx):
        super().__init__()
        self.idx=idx
        self.res_queue=res_queue
        self.name = 'w%i' % idx
        self.gbest= gbest
        self.gnet = gnet
        self.lnet = MyNet()
        self.pbest = MyNet()
        self.pbestScore =0
        self.env = gym.make('CartPole-v1').unwrapped
        #self.env = gym.make('CartPole-v1')
        self.lnet.load_state_dict(gnet.state_dict())
        self.pbest.load_state_dict(gnet.state_dict())

    def run(self):
        step = 0
        cnt = 0
        self.lnet.eval()
        self.pbest.eval()

        while step < MAX_EP:

            score=0
            done=False
            s = self.env.reset()

            step += 1
            cnt += 1  #控制产生变异

            
            while not done and score<MAX_SCORE:
                if self.idx==0:
                    self.env.render()
                prob = self.lnet(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = self.env.step(a)
                s = s_prime
                if done:
                    score-=10
                    break
                else:
                    score+=r
            if self.idx==0: 
                print('STEP:%d'%(step,))
                self.res_queue.put(score)
                
            if score>self.pbestScore:
                cnt=0
                self.pbestScore=score
                self.pbest.load_state_dict(self.lnet.state_dict())

            if score>self.gbest.value:
                print('thread %d find the best:%d'%(self.idx,score))
                self.gbest.value=int(score)
                self.gnet.load_state_dict(self.lnet.state_dict())
                self.gnet.save()
            if cnt<50:
                for (p,p1,p2) in zip(self.lnet.parameters(), \
                        self.pbest.parameters(), \
                        self.gnet.parameters()):
                    mu=(p1+p2)/2.0
                    std=torch.abs(p1-p2)+EPS
                    d=torch.normal(mu,std)
                    p.data=d.data
            else:
                cnt=0
                for (p,p1,p2) in zip(self.lnet.parameters(), \
                        self.pbest.parameters(), \
                        self.gnet.parameters()):
                    d=torch.normal((p1-p2)/2.,1)
                    p.data=d.data

        if self.idx==0: self.res_queue.put(None)

if __name__ == '__main__':
    gnet = MyNet()
    gnet.load()
    gnet.share_memory()
    gnet.eval()
    gscore,res_queue  = mp.Value('i', 0), mp.Queue()

    workers = [Worker(gnet,gscore,res_queue, i) for i in range(N_THREADS)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    '''
    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
    '''

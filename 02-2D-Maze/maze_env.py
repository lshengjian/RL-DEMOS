"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example. The RL is in RL_brain.py.
"""


import numpy as np
import time
import tkinter as tk


UNIT = 80   # pixels
UNIT_R = UNIT//2
BLOCK_R = UNIT*3//8
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super().__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self.origin = np.array([UNIT_R, UNIT_R])
        self.items=[]
        self.agent=None
        self.agentIdx=0
        self.itemsData={
            'agent':{'idx':0,'offset':[0,0],'color':'red'},
            'hell1':{'idx':9,'offset':[2,1],'color':'black'},
            'hell2':{'idx':6,'offset':[1,2],'color':'black'},
            'oval':{'idx':10,'offset':[2,2],'color':'yellow'}
        }


        self._buildMaze()

    def _getIdx(self,loc):
        return loc[0]*MAZE_W+loc[1]
        
    def _buildMaze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)


        for k,v in self.itemsData.items():
            loc=v['offset']
            c = self.origin + np.array([UNIT * loc[0], UNIT* loc[1]])
            f=self.canvas.create_rectangle
            if 'oval'==k:
                f=self.canvas.create_oval
            item=f(c[0] - UNIT_R, c[1] - UNIT_R,c[0] + UNIT_R, c[1] + UNIT_R,fill=v['color'])
            if 'agent'== k:
                self.agent=item
            self.items.append(item)
        self.canvas.pack()

    def reset(self):
        time.sleep(0.5)
        loc=self.itemsData['agent']['offset']
        c = self.origin + np.array([UNIT * loc[0], UNIT* loc[1]])
        self.canvas.coords(self.agent, c[0] - UNIT_R, c[1] - UNIT_R,c[0] + UNIT_R, c[1] + UNIT_R)
        self.agentIdx=self.itemsData['agent']['idx']
        self.update()
        return self.agentIdx
    
    def step(self, action):
        loc = (self.agentIdx//MAZE_H,self.agentIdx%MAZE_H)
        d = np.array([0, 0])
        #['u', 'd', 'l', 'r']
        if action == 0:   # up
            if loc[1] > 0:
                d[1] -= 1
        elif action == 1:   # down
            if loc[1] < (MAZE_H - 1) :
                d[1] += 1
        elif action == 2:   # left
            if loc[0] > 0:
                d[0] -= 1
        elif action == 3:   # right
            if loc[0] < (MAZE_W - 1) :
                d[0] += 1

        #print('\r%s'% self.action_space[action],end='')

        self.canvas.move(self.agent, d[0]*UNIT, d[1]*UNIT)  # move agent

        s_ = self._getIdx(loc+d)  # next state

        # reward function
        if s_ == self.itemsData['oval']['idx']:
            reward = 1
            done = True
 
        elif s_ == self.itemsData['hell1']['idx'] or s_ == self.itemsData['hell2']['idx']:
            reward = -1
            done = True

        else:
            reward = 0
            done = False

        self.agentIdx= s_

        return s_, reward, done

    def render(self):
        time.sleep(0.2)
        self.update()


def update():
    for t in range(2):
        s = env.reset()
        done=False
        env.render()

        while not done:
            
            a = 1
            s, r, done = env.step(a)
            env.render()


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
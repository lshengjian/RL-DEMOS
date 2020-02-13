"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

"""

from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(20):
        # initial observation
        observation = env.reset()
        done=False
        env.render()
        cnt=0
        while not done:
            cnt+=1
            # RL choose action based on observation
            action = RL.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # RL learn from this transition
            RL.learn(observation, action, reward, observation_)
            # swap observation
            observation = observation_

            # fresh env
            env.render()
        print('\r\ntotal steps:%s'% cnt)
        print(RL.q_table)
    # end of game
    print('game over')
    env.destroy()
    

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
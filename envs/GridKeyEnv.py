import time
import random
from os import system
import sys
from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab import spaces
import numpy as np
from copy import deepcopy

def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class GridWorldKey(Env):
    metadata={}
    def __init__(self, max_time=12000, n_keys = 2, sensor_range=2, normlize_obs=False, use_nearby_obs=False, random_act_thereshold = 1.0, manual=False):
        self.n_key = n_keys
        self.max_time = max_time
        self.normlize_obs = normlize_obs
        self.use_nearby_obs = use_nearby_obs
        self.sensor_range = sensor_range

        self._map = np.zeros((self.n_key*12-1,11), dtype = np.int64)

        self._init_pos = np.array((1,4))
        
        self.reward_range = (-np.inf,np.inf)

        self.time = 0
        self.finish = 0
        self.haveKey = 0
        self.last_haveKey = 0
        self.rseed = None
        self.prev_reward = 0
        self.mid_x = np.ceil(self._map.shape[0]/2.0)
        self.mid_y = np.ceil(self._map.shape[1]/2.0)
        self.mid_key = self.n_key/2.0
        self._dir = [[-1, 0], [0, 1], [1, 0], [0, -1]] 

        assert random_act_thereshold >= 0 and random_act_thereshold <= 1.0
        self._random_act_thereshold = random_act_thereshold

        self.manual = manual
        self.reset()

    def seed(self,sd=0):
        np.random.seed(sd)
        
    def _inBoard(self, pos):
        return not np.any([pos < [0,0], pos >= self._map.shape])

    def _eq(self, pos1, pos2):
        return not np.any(pos1 != pos2)

    @property
    def action_space(self):
        # 0 - UP, 1 - RIGHT, 2 - DOWN, 3 - LEFT
        return spaces.Box(0,1,(4,))
        #return spaces.Discrete(4)

    @property
    def observation_space(self):
        # self.observation_space = spaces.Box(np.array([-1,-1,0,0,-2,-1]), np.array([1,1,1,1,2,1]))
        if self.normlize_obs:
            if self.use_nearby_obs:
                return spaces.Box(-1, 1, ( (2*self.sensor_range+1)**2+2,))
            else:
                return spaces.Box(-1, 1, (3+2*self.n_key,))
        else:
            if self.use_nearby_obs:
                return spaces.Box(np.array([0, 0, 0] + [0] * 24),np.array([self._map.shape[0], self._map.shape[1], self.n_key] + [5] * ((2*self.sensor_range+1)**2-1) ))
            else:
                return spaces.Box(np.array([0, 0, 0]+[0]*self.n_key),np.array([self._map.shape[0], self._map.shape[1], self.n_key]+[self._map.shape[0], self._map.shape[1]]*self.n_key))

    def reset(self, rseed=None):
        self.rseed = random.randint(1, 1000) if rseed == None else rseed
        
        rand_state = np.random.get_state()
        
        self.seed(self.rseed)

        self.time = 0
        self.haveKey = 0
        self.last_haveKey = 0
        self.finish = 0
        
        # 0 - road, 1 - walls, 2 - doors, 3 - keys, 4 - goal, 5 - the door to goal
        self._map[:] = 0
        self._map[:,5] = 1
        self._map[5::6,] = 1
        self._map[:,2] = 0
        self._map[2,:] = 0
        self._map[-6,2] = 5
        self._map[14::12,5] = 2

        self.agent_pos = deepcopy(self._init_pos)
        self.goal_pos = np.array((self.n_key * 12 - 4, 2))
        self.key_pos = np.vstack((np.arange(0,self.n_key) * 12 + 2, [8]* self.n_key)).T
        self.key_pos += np.random.randint(-2,2,size=self.key_pos.shape)
        self.door_pos = np.asarray(np.where(self._map == 2)).T
        self.door_pos = np.concatenate((self.door_pos,[[self._map.shape[0] - 6, 2]]))
        for item in self.key_pos: 
            self._map[tuple(item)] = 3
        self._map[tuple(self.goal_pos)] = 4

        np.random.set_state(rand_state)
        
        # TODO:
        return self.get_state()

    def move(self, action):
        if type(action)==int:
            real_act = action
        else:
            #try:
            p = np_softmax(action)
            action = np.random.choice(4,1,replace=False,p=p)[0]
            #except Exception as e:
            #    print(action,p)
        
        assert action < 4 and action >= 0
        
        if action >= 4 or action < 0:
            print ('unexpected action:{}'.format(action))
        
        if np.random.random() > self._random_act_thereshold:
            action = np.random.randint(4)

        new_pos = self.agent_pos + self._dir[action]
        if not self._inBoard(new_pos):
            new_pos = deepcopy(self.agent_pos)
        elif self._map[tuple(new_pos)] in [1,2,5]:
            new_pos = deepcopy(self.agent_pos)
        elif self._map[tuple(new_pos)] == 3:
            self._map[tuple(new_pos)] = 0
            self.haveKey += 1
            self._map[tuple(self.door_pos[self.haveKey - 1])] = 0

        self.agent_pos = new_pos
        
        return action

    def step(self, action):
        #assert action < 4 and action >= 0
        message = {}

        action = self.move(action)

        self.time += 1

        self.prev_reward = self.get_reward()
        return Step(observation=self.get_state(), reward=self.prev_reward, done=self.get_finish())

    def get_Map(self):
        return self._map

    def print_action_set(self):
        print('0:UP, 1:RIGHT, 2:DOWN, 3:LEFT')

    def get_finish(self):
        if self._eq(self.agent_pos, self.goal_pos):
            return True
        if self.time > self.max_time:
            return True
        return False

    def get_reward(self):
        if self._eq(self.agent_pos, self.goal_pos):
            return 10.0
        if self.last_haveKey+1 == self.haveKey:
            self.last_haveKey = self.haveKey
            return 2.0
        else:
            return 0.0

    def get_state(self):
        nearby_obs = self.get_nearby_obs()
        if self.normlize_obs:
            x, y = self.agent_pos / [50.0, 12.0]
            k = self.haveKey / 5.0
            key_pos = self.key_pos / [50.0, 12.0]
            nearby_obs = (nearby_obs-2)/2
        else:
            x, y = self.agent_pos
            k = self.haveKey
            key_pos = self.key_pos

        if self.use_nearby_obs:
            return np.hstack(([x,y,k],nearby_obs))
        else:
            return np.hstack(([x,y,k],np.reshape(key_pos,(-1,))))

    def get_nearby_obs(self):
        # 0 - road, 1 - walls, 2 - doors, 3 - keys, 4 - goal, 5 - the door to goal
        res = []
        for i in np.arange(-1*self.sensor_range, self.sensor_range+1):
            for j in np.arange(-1*self.sensor_range, self.sensor_range+1):
                if i == 0 and j == 0:
                    continue
                new_pos = self.agent_pos + [i,j]
                if not self._inBoard(new_pos):
                    res.append(1)
                else:
                    tmp = self._map[tuple(new_pos)]
                    tmp = tmp if tmp != 5 else 2
                    res.append(tmp)
        res = np.array(res)
        if self.normlize_obs:
            res = (res-2)/2
        return res

    def get_action_dim(self):
        return 4

    def if_exceed_time_limit(self):
        if self.time > self.max_time:
            return True
        return False

    def render(self):
        sig_arr = [' ', '#', '|', 'K', 'G', '-', 'A']
        s = ""
        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                if self._eq([i,j], self.agent_pos):
                    s += sig_arr[-1]
                else:
                    s += sig_arr[self._map[i, j]]
            s += '\n'
        if not self.manual:
            if sys.platform.startswith("win"):
                system("cls")
            else:
                system("clear")
            print(s)
        else:
            return s

    def replay(self, actions = [], seed = None, gap = 0.2):
        res = []
        self.reset(rseed=seed)
        self.render()
        time.sleep(0.5)
        for each in actions:
            observation, reward, finish, mess = self.step(each)
            res.append((observation, reward, finish, mess))
            self.render()
            time.sleep(gap)
        print('replay done')
        return res

    def gameLoop(self):
        def valid(x):
            return x == ord('q') or x == ord('r') or x == 119 or x == 115 or x == 97 or x == 100
        import curses

        seed = np.random.randint(10000)
        self.reset(seed)
        screen = curses.initscr()
        record = []
        while True:
            screen.clear()
            screen.addstr(self.render())
            screen.addstr('key_pos:' + str(self.key_pos))
            char = screen.getch()
            while not valid(char):
                char = screen.getch()
            screen.refresh()
            action = 0
            if char == ord('q'): break
            elif char == ord('r'): 
                seed = np.random.randint(10000)
                self.reset(seed)
                record = []
            elif char == 100: action = 1
            elif char == 97: action = 3
            elif char == 119: action = 0
            elif char == 115: action = 2

            self.step(action)
            record.append(action)
            if self.get_finish():
                break
        curses.endwin()
        self.manual = False
        return record, seed
        
if __name__ == '__main__':
    env = GridWorldKey(n_keys = 2, manual = True)
    rec, seed = env.gameLoop()
    env.replay(rec, seed)
    
import os
import numpy as np
from datetime import datetime
from utils import plot
from rl.policy import Policy

class Agent():
    def __init__(self,
                 name,
                 envs,
                 rew_scale,
                 batch_size,
                 n_ways
                 ):
        self.name = name
        self.envs = envs
        self.rew_scale = rew_scale
        self.obs_dim, self.act_dim = envs[0].observation_space.shape[0]+1, envs[0].action_space.shape[0]
        self.batch_size = batch_size
        self.n_envs = len(envs)
        self.n_ways = n_ways

        now = datetime.now().strftime("%b_%d_%H_%M_%S")
        self.log_path = os.path.join('log-files', name, now)

        self._build_policy()
        
    def _build_policy(self):
        self.policy = Policy(name='policy', obs_dim=self.obs_dim, act_dim=self.act_dim, n_ways=self.n_ways, batch_size=self.batch_size, log_path=self.log_path)
        
    def _rollout(self,seed, n_iter, L, mask, render=False, plot_ret = True):
        mask = np.array(mask)
        for i in range(0, n_iter):
            ep_r = 0.0
            observes, actions, rewards = [], [], []
            done = False
            step = 0

            choose_idx = np.random.choice(np.where(mask == 1)[0])
            env = self.envs[choose_idx]

            obs = env.reset(rseed=seed)  # obs here are unscaled
            while not done:
                if render:
                    env.render()
                obs = obs.astype(np.float32).reshape((1, -1))
                obs = np.append(obs, [[0.001*step]], axis=1)
                observes.append(obs)

                action = self.policy.act(obs)
                obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
                if not isinstance(reward, float):
                    reward = np.asscalar(reward)
                reward = reward / self.rew_scale[choose_idx]
                ep_r += reward

                actions.append(action)
                rewards.append(reward)

                if done:
                    break
                step += 1

            if plot_ret:
                plot.plot('ep_r_L{}'.format(L),ep_r)
                plot.tick('ep_r_L{}'.format(L))
                plot.flush(self.log_path,verbose=False)
                print('average episode return={}'.format(ep_r))
            yield np.array(observes), np.array(actions), np.array(rewards)

    def init_scaler(self,n_iter):
        for observes, actions, rewards in self._rollout(0, n_iter, L='init', mask=[1]*self.n_envs, plot_ret = False):
            if observes.shape[0] > 0: self.policy.update_scaler(observes)
            
    def train(self, n_iter, L, mask, trainWeight=False):
        for i in range(n_iter):
            print("Learning L{}, iteration={}".format(L,i),end='\t')
            seed = np.random.randint(0, 100000)
            for observes, actions, rewards in self._rollout(seed, 1, L=L, mask=mask):
                obs = observes.reshape(-1,self.obs_dim)
                acts = actions.reshape(-1,self.act_dim)
                rews = rewards
                if obs.shape[0] > 0: self.policy.update(obs,acts,rews,L,trainWeight=trainWeight)

    def add_module(self, L_name):
        self.n_ways += 1
        self.policy.add_module(L_name)

    def addenv(self,env,rew_scale):
        self.envs.append(env)
        self.rew_scale.append(rew_scale)
        self.n_envs += 1

    def save_model(self):
        self.policy.save_model()

    def load_model(self,path):
        self.policy.load_model(path)
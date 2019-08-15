import os
import numpy as np
from utils.utils import Scaler
from nets.trpo_net import TrpoNet
from nets.value_function import NNValueFunction
import copy
import scipy
import gc

class Policy():
    def __init__(self,
                 name,
                 obs_dim,
                 act_dim,
                 n_ways,
                 batch_size,
                 log_path,
                 gamma=0.995,
                 lam = 0.98,
                 kl_targ=0.003,
                 hid1_mult=10,
                 policy_logvar=1.0
                 ):
        self.name = name
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.n_ways = n_ways
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.kl_targ = kl_targ
        self.hid1_mult = hid1_mult
        self.policy_logvar = policy_logvar
        self.log_path = log_path

        self.scaler = Scaler(self.obs_dim)
        self.val_func = NNValueFunction(self.obs_dim, hid1_mult=10)
        self.trpo_net = TrpoNet(name, self.obs_dim, self.act_dim, n_ways=n_ways, kl_targ=kl_targ, hid1_mult=hid1_mult, policy_logvar=policy_logvar)

        self.trajectories = []
        self.episode = 0

    def update_scaler(self,unscaled):
        self.scaler.update(unscaled)  # update running statistics for scaling observations

    def update(self,unscaled_obs, actions, rewards, L, trainWeight=False):
        scale, offset = self.scaler.get()
        scale[-1] = 1.0
        offset[-1] = 0.0
        observes = (unscaled_obs - offset) * scale
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        self.trajectories.append(trajectory)
        if len(self.trajectories) > self.batch_size:
            unscaled = np.concatenate([t['unscaled_obs'] for t in self.trajectories])
            self.scaler.update(unscaled)  # update running statistics for scaling observations
            trajs = copy.deepcopy(self.trajectories)
            self.trajectories = []

            self.episode += len(trajs)
            self._add_value(trajs, self.val_func)  # add estimated values to episodes
            self._add_disc_sum_rew(trajs, self.gamma)  # calculated discounted sum of Rs
            self._add_gae(trajs, self.gamma, self.lam)  # calculate advantage
            # concatenate all episodes into single NumPy arrays
            observes, actions, advantages, disc_sum_rew = self._build_train_set(trajs)
            self.trpo_net.update(observes, actions, advantages, L, trainWeight=trainWeight)  # update policy
            self.val_func.fit(observes, disc_sum_rew)  # update value function



    def act(self,unscaled_obs):
        scale, offset = self.scaler.get()
        scale[-1] = 1.0  # don't scale time step feature
        offset[-1] = 0.0  # don't offset time step feature
        #print(self.name,unscaled_obs.shape,len(offset))
        obs = (unscaled_obs - offset) * scale
        action = self.trpo_net.sample(obs).reshape((1, -1)).astype(np.float32)
        return action

    def add_module(self,L_name):
        self.n_ways+=1

        var_dict = self.trpo_net.get_vars()
        L_list = self.trpo_net.get_L_list()
        L_list = np.hstack((L_list, [L_name]))

        new_pi = TrpoNet(self.name, self.obs_dim, self.act_dim, self.n_ways, self.kl_targ,self.hid1_mult,self.policy_logvar)
        new_pi.set_vars(var_dict)
        new_pi.set_L_list(L_list)

        self.trpo_net.close_sess()
        self.trpo_net = new_pi
        print('ws={}'.format(self.trpo_net.getWs()))
        gc.collect()

    def save_model(self):
        p_save_path = os.path.join(self.log_path,'models','policy_net')
        v_save_path = os.path.join(self.log_path,'models','value_func')
        os.makedirs(p_save_path, exist_ok=True)
        os.makedirs(v_save_path, exist_ok=True)
        self.trpo_net.save_model(p_save_path)
        self.val_func.save_model(v_save_path)
        print('Models are saved in {}'.format(os.path.join(self.log_path,'models')))

    def load_model(self,path):
        self.trpo_net.load_model(path)
        self.val_func.load_model(path)
        print('Models are loaded')

    def close_session(self):
        self.val_func.close_sess()
        self.trpo_net.close_sess()


    def _discount(self, x, gamma):
        """ Calculate discounted forward sum of a sequence at each point """
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

    def _add_value(self, trajectories, val_func):
        """ Adds estimated value to all time steps of all trajectories

        Args:
            trajectories: as returned by run_policy()
            val_func: object with predict() method, takes observations
                and returns predicted state value

        Returns:
            None (mutates trajectories dictionary to add 'values')
        """
        for trajectory in trajectories:
            observes = trajectory['observes']
            values = val_func.predict(observes)
            trajectory['values'] = values

    def _add_disc_sum_rew(self, trajectories, gamma):
        """ Adds discounted sum of rewards to all time steps of all trajectories

        Args:
            trajectories: as returned by run_policy()
            gamma: discount

        Returns:
            None (mutates trajectories dictionary to add 'disc_sum_rew')
        """
        for trajectory in trajectories:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - gamma)
            else:
                rewards = trajectory['rewards']
            disc_sum_rew = self._discount(rewards, gamma)
            trajectory['disc_sum_rew'] = disc_sum_rew

    def _add_gae(self, trajectories, gamma, lam):
        """ Add generalized advantage estimator.
        https://arxiv.org/pdf/1506.02438.pdf

        Args:
            trajectories: as returned by run_policy(), must include 'values'
                key from add_value().
            gamma: reward discount
            lam: lambda (see paper).
                lam=0 : use TD residuals
                lam=1 : A =  Sum Discounted Rewards - V_hat(s)

        Returns:
            None (mutates trajectories dictionary to add 'advantages')
        """
        for trajectory in trajectories:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - gamma)
            else:
                rewards = trajectory['rewards']
            values = trajectory['values']
            # temporal differences
            tds = rewards - values + np.append(values[1:] * gamma, 0)
            advantages = self._discount(tds, gamma * lam)
            trajectory['advantages'] = advantages

    def _build_train_set(self, trajectories):
        """

        Args:
            trajectories: trajectories after processing by add_disc_sum_rew(),
                add_value(), and add_gae()

        Returns: 4-tuple of NumPy arrays
            observes: shape = (N, obs_dim)
            actions: shape = (N, act_dim)
            advantages: shape = (N,)
            disc_sum_rew: shape = (N,)
        """
        observes = np.concatenate([t['observes'] for t in trajectories])
        actions = np.concatenate([t['actions'] for t in trajectories])
        disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
        advantages = np.concatenate([t['advantages'] for t in trajectories])
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        return observes, actions, advantages, disc_sum_rew
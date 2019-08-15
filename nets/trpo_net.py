import os
import numpy as np
import tensorflow as tf

class TrpoNet(object):
    """ NN-based policy approximation """
    def __init__(self, name, obs_dim, act_dim, n_ways, kl_targ, hid1_mult, policy_logvar, clipping_range=None):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence between pi_old and pi_new
            hid1_mult: size of first hidden layer, multiplier of obs_dim
            policy_logvar: natural log of initial policy variance
        """
        self.name = name
        self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
        self.eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
        self.kl_targ = kl_targ
        self.hid1_mult = hid1_mult
        self.policy_logvar = policy_logvar
        self.epochs = 20
        self.lr = None
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.kl_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_ways = n_ways
        self.clipping_range = clipping_range

        self.weight_vars = []
        self.sub_means = []
        self.train_sub_ops = []
        self.L_list = []

        self._build_graph()
        self._init_session()

    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._kl_entropy()
            self._sample()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def _placeholders(self):
        """ Input placeholders"""
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        # strength of D_KL loss terms:
        self.kl_multiplier_ph = tf.placeholder(tf.float32, (), 'kl_mul_ph')
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        # learning rate:
        self.lr_ph = tf.placeholder(tf.float32, (), 'lr_ph')
        # log_vars and means with pi_old (previous step's policy parameters):
        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

    def _policy_nn(self):
        """ Neural net for policy approximation function

        Policy parameterized by Gaussian means and variances. NN outputs mean
         action based on observation. Trainable variables hold log-variances
         for each action dimension (i.e. variances not determined by NN).
        """
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        hid1_size = self.obs_dim * self.hid1_mult  # 10 empirically determined
        hid3_size = self.act_dim * 10  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined

        self.weight_vars = []
        self.sub_means = []

        with tf.variable_scope("average_pi", reuse=tf.AUTO_REUSE):
            # 3 hidden layers with tanh activations
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid1_size)), name="h2")
            out = tf.layers.dense(out, hid3_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3")
            self.avg_mean = tf.layers.dense(out, self.act_dim,
                                         kernel_initializer=tf.random_normal_initializer(
                                             stddev=np.sqrt(1 / hid3_size)), name="means")

            self.weight_vars.append(tf.get_variable('avg_weight', shape=(), dtype=tf.float32,
                                                    initializer=tf.constant_initializer(1)))

        # logvar_speed is used to 'fool' gradient descent into making faster updates
        # to log-variances. heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * hid3_size) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.policy_logvar

        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
              .format(hid1_size, hid2_size, hid3_size, self.lr, logvar_speed))

        for i in range(self.n_ways):
            with tf.variable_scope('env_pi{}'.format(i), reuse=tf.AUTO_REUSE):
                l1 = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1")
                l2 = tf.layers.dense(l1, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid1_size)), name="h2")
                l_out = tf.layers.dense(l2, self.act_dim,
                                         kernel_initializer=tf.random_normal_initializer(
                                             stddev=np.sqrt(1 / hid3_size)), name="means")
                self.sub_means.append(l_out)

                self.weight_vars.append(tf.get_variable('mix_weight{}'.format(i),shape=(),dtype=tf.float32,initializer=tf.constant_initializer(0)))

        w = tf.nn.softmax(tf.stack(self.weight_vars),name='stack_w')
        means = w[0]*self.avg_mean
        for i in range(self.n_ways):
            means += w[i+1]*self.sub_means[i]
        self.means = means
        #means = tf.stack([self.avg_mean]+self.sub_means,name='stack_mean')
        #self.means = tf.reshape(tf.matmul(tf.reshape(w,[1,-1]), means), [-1,self.act_dim])



    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions

        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

    def _kl_entropy(self):
        """
        Add to Graph:
            1. KL divergence between old and new distributions
            2. Entropy of present policy given states and actions

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """
        log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.means - self.old_means_ph) /
                                                     tf.exp(self.log_vars), axis=1) -
                                       self.act_dim)
        self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_sum(self.log_vars))

    def _sample(self):
        """ Sample from distribution, given observation """
        self.sampled_act = (self.means +
                            tf.exp(self.log_vars / 2.0) *
                            tf.random_normal(shape=(self.act_dim,)))

    def _loss_train_op(self):
        """
        Three loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2

        See: https://arxiv.org/pdf/1707.02286.pdf
        """
        if self.clipping_range is not None:
            print('setting up loss with clipping objective')
            pg_ratio = tf.exp(self.logp - self.logp_old)
            clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - self.clipping_range[0], 1 + self.clipping_range[1])
            surrogate_loss = tf.minimum(self.advantages_ph * pg_ratio,
                                        self.advantages_ph * clipped_pg_ratio)
            self.loss = -tf.reduce_mean(surrogate_loss)
        else:
            print('setting up loss with KL penalty')
            loss1 = -tf.reduce_mean(self.advantages_ph *
                                    tf.exp(self.logp - self.logp_old))
            loss2 = self.kl_multiplier_ph * tf.reduce_mean(self.beta_ph * self.kl)
            loss3 = self.kl_multiplier_ph * self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))
            self.loss = loss1 + loss2 + loss3
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_avg_op = optimizer.minimize(self.loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='average_pi')+self.weight_vars)
        self.train_weight_op2 = [] ## without weight of self way
        self.train_sub_ops = []
        self.train_weight_op = optimizer.minimize(self.loss, var_list=self.weight_vars)
        for i in range(self.n_ways):
            self.train_sub_ops.append(
                optimizer.minimize(
                    self.loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='env_pi{}'.format(i))+self.weight_vars)
            )
            self.train_weight_op2.append(
                optimizer.minimize(
                    self.loss, var_list=[x for x in self.weight_vars if x is not self.weight_vars[i]]
                )
            )


    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(graph=self.g, config=config)
        self.sess.run(self.init)

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages, L , trainWeight = False):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
            idx: index to select
            logger: Logger object, see utils.py
        """
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier,
                     self.kl_multiplier_ph: self.kl_multiplier}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            if L=='0':
                self.sess.run(self.train_avg_op, feed_dict)
            else:
                if trainWeight:
                    self.sess.run(self.train_weight_op, feed_dict)
                else:
                    idx = np.where(self.L_list == L)[0]
                    assert len(idx) == 1
                    self.sess.run(self.train_sub_ops[idx[0]], feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl > self.kl_targ * 2:
            self.kl_multiplier /= 1.5
        # if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
        #     self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
        #     if self.beta > 30 and self.lr_multiplier > 0.1:
        #         self.lr_multiplier /= 1.5
        # elif kl < self.kl_targ / 2:
        #     self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
        #     if self.beta < (1 / 30) and self.lr_multiplier < 10:
        #         self.lr_multiplier *= 1.5


    def getWs(self):
        ws = self.sess.run(self.weight_vars)
        return ws

    def get_vars(self):
        with self.g.as_default():
            avg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='average_pi')
            env_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='env_pi')
            tfvars = avg_vars + env_vars
            names = [x.name for x in tfvars]
            val = self.sess.run(tfvars)
            var_dict = dict(zip(names,val))

        return var_dict

    def set_vars(self,var_dict):
        with self.g.as_default():
            avg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='average_pi')
            env_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='env_pi')
            tfvars = avg_vars + env_vars
            names = [x.name for x in tfvars]
            
            local_dict = dict(zip(names,tfvars))
            for key in var_dict.keys():
                self.sess.run(local_dict[key].assign(var_dict[key]))

    def get_L_list(self):
        return self.L_list

    def set_L_list(self,L_list):
        self.L_list = L_list

    def save_model(self,path):
        path = os.path.join(path,'policy_net_model')
        self.saver.save(self.sess,save_path=path)

    def load_model(self,path):
        self.saver.restore(self.sess, save_path=path)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

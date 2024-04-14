import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from agents.CacheAgent import LearnerAgent
from agents.ReflexAgent import RandomAgent, LRUAgent, LFUAgent

np.random.seed(1)
tf.random.set_seed(1)

tf.compat.v1.disable_eager_execution()


# Deep Q Network
class DQNAgent(LearnerAgent):
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,

            e_greedy_min=(0.1, 0.1),
            e_greedy_max=(0.1, 0.1),

            # leave either e_greedy_init or e_greedy_decrement None to disable epsilon greedy
            # only leave e_greedy_increment to disable dynamic bidirectional epsilon greedy
            e_greedy_init=None,
            e_greedy_increment=None,
            e_greedy_decrement=None,

            reward_threshold=None,
            history_size=10,
            dynamic_e_greedy_iter=5,
            explore_mentor='LRU',

            replace_target_iter=300,
            memory_size=500,
            batch_size=32,

            output_graph=False,
            verbose=0
    ):
        self.cost = None
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.batch_size = batch_size

        self.epsilons_min = e_greedy_min
        self.epsilons_max = e_greedy_max
        self.epsilons_increment = e_greedy_increment
        self.epsilons_decrement = e_greedy_decrement

        self.epsilons = list(e_greedy_init)
        if (e_greedy_init is None) or (e_greedy_decrement is None):
            self.epsilons = list(self.epsilons_min)

        self.explore_mentor = None
        if explore_mentor.upper() == 'LRU':
            self.explore_mentor = LRUAgent
        elif explore_mentor.upper() == 'LFU':
            self.explore_mentor = LFUAgent

        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_counter = 0

        # initialize a history set for rewards
        self.reward_history = []
        self.history_size = history_size
        self.dynamic_e_greedy_iter = dynamic_e_greedy_iter
        self.reward_threshold = reward_threshold

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.compat.v1.get_collection('target_net_params')
        e_params = tf.compat.v1.get_collection('eval_net_params')
        self.replace_target_op = [tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.compat.v1.Session()

        if output_graph:
            tf.compat.v1.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.cost_his = []

        self.verbose = verbose

    def _build_net(self):

        tf.compat.v1.reset_default_graph()

        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')  # eval_net's input
        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_')  # target_net's input
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating Q(s',a')

        # ------below is eval_net------

        with tf.compat.v1.variable_scope('eval_net'):
            # l1
            layer1 = tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu,
                                           kernel_initializer=tf.initializers.random_normal,
                                           bias_initializer=tf.initializers.constant)
            a1 = layer1(self.s)

            # l2
            layer2 = tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu,
                                           kernel_initializer=tf.initializers.random_normal,
                                           bias_initializer=tf.initializers.constant)
            a2 = layer2(a1)

            # l3
            layer3 = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu,
                                           kernel_initializer=tf.initializers.random_normal,
                                           bias_initializer=tf.initializers.constant)
            a3 = layer3(a2)

            # l_out
            layer_out = tf.keras.layers.Dense(units=self.n_actions, activation=tf.keras.activations.relu,
                                              kernel_initializer=tf.initializers.random_normal,
                                              bias_initializer=tf.initializers.constant)
            self.q_eval = layer_out(a3)

            # self.eval_model = tf.keras.models.Sequential([layer1, layer2, layer3])
            # self.eval_model.compile(loss=tf.keras.losses.mean_squared_error)
            # self.callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')

            # -------below is target_net-------

        with tf.compat.v1.variable_scope('target_net'):
            # l1
            layer1 = tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu,
                                           kernel_initializer=tf.initializers.random_normal,
                                           bias_initializer=tf.initializers.constant)
            a1 = layer1(self.s_)
            # l2
            layer2 = tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu,
                                           kernel_initializer=tf.initializers.random_normal,
                                           bias_initializer=tf.initializers.constant)
            a2 = layer2(a1)
            # l3
            layer3 = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu,
                                           kernel_initializer=tf.initializers.random_normal,
                                           bias_initializer=tf.initializers.constant)
            a3 = layer3(a2)
            # l_out
            layer_out = tf.keras.layers.Dense(units=self.n_actions, activation=tf.keras.activations.relu,
                                              kernel_initializer=tf.initializers.random_normal,
                                              bias_initializer=tf.initializers.constant)
            self.q_next = layer_out(a3)

        with tf.compat.v1.variable_scope('loss'):
            self.loss = tf.keras.losses.mean_squared_error(self.q_target, self.q_eval)

        with tf.compat.v1.variable_scope('train'):
            self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        s, s_ = s['features'], s_['features']
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

        # Record reward
        if len(self.reward_history) == self.history_size:
            self.reward_history.pop(0)
        self.reward_history.append(r)

    def choose_action(self, observation):
        # draw probability sample
        coin = np.random.uniform()
        if coin < self.epsilons[0]:
            action = RandomAgent._choose_action(self.n_actions)
        elif self.epsilons[0] <= coin and coin < self.epsilons[0] + self.epsilons[1]:
            action = self.explore_mentor._choose_action(observation)
        else:
            observation = observation['features']
            # to have batch dimension when feed into tf placeholder
            observation = observation[np.newaxis, :]

            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)

        if action < 0 or action > self.n_actions:
            raise ValueError("DQNAgent: Error index %d" % action)

        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # verbose
            if self.verbose >= 1:
                print('Target DQN params replaced')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # update q_target parameters due to the latest q_eval
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # plot loss function in tensorboard

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features], self.q_target: q_target}
                                     )
        self.cost_his.append(self.cost)
        # verbose                    
        if (self.verbose == 2 and self.learn_step_counter % 100 == 0) or \
                (self.verbose >= 3 and self.learn_step_counter % 20 == 0):
            print("Step=%d: Cost=%d" % (self.learn_step_counter, self.cost))

        # increasing or decreasing epsilons
        if self.learn_step_counter % self.dynamic_e_greedy_iter == 0:

            # if we have e-greedy?
            if self.epsilons_decrement is not None:
                # dynamic bidirectional e-greedy
                if self.epsilons_increment is not None:
                    rho = np.median(np.array(self.reward_history))
                    if rho >= self.reward_threshold:
                        self.epsilons[0] -= self.epsilons_decrement[0]
                        self.epsilons[1] -= self.epsilons_decrement[1]
                        # verbose
                        if self.verbose >= 3:
                            print("Eps down: rho=%f, e1=%d, e2=%f" % (rho, self.epsilons[0], self.epsilons[1]))
                    else:
                        self.epsilons[0] += self.epsilons_increment[0]
                        self.epsilons[1] += self.epsilons_increment[1]
                        # verbose                    
                        if self.verbose >= 3:
                            print("Eps up: rho=%f, e1=%d, e2=%f" % (rho, self.epsilons[0], self.epsilons[1]))
                # traditional e-greedy
                else:
                    self.epsilons[0] -= self.epsilons_decrement[0]
                    self.epsilons[1] -= self.epsilons_decrement[1]

            # enforce upper bound and lower bound
            truncate = lambda x, lower, upper: min(max(x, lower), upper)
            self.epsilons[0] = truncate(self.epsilons[0], self.epsilons_min[0], self.epsilons_max[0])
            self.epsilons[1] = truncate(self.epsilons[1], self.epsilons_min[1], self.epsilons_max[1])

        self.learn_step_counter += 1

    # def plot_loss(self):
    # eval_model, callback = self.eval_model, self.callback
    # return eval_model, callback

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

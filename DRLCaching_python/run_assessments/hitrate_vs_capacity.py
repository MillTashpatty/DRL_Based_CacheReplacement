import numpy as np
from cache.Cache import Cache
from agents.CacheAgent import *
from agents.DQNAgent import DQNAgent
from agents.ReflexAgent import *
from cache.DataLoader import DataLoaderPintos

import pandas as pd
import pickle

if __name__ == "__main__":

    dataloader = DataLoaderPintos(
        ["DRLCaching_python/dataset_and_generate/zipf_300id_1000000req.csv"])

    assess_hr_DQN = []
    assess_hr_LRU = []
    assess_hr_LFU = []
    assess_hr_MRU = []
    assess_hr_Random = []

    sizes = [5, 25, 50, 100, 150, 200, 250]  # 5, 25, 50, 100, 150, 200, 250
    for capacity in sizes:

        print("==================== Cache Size: {} ====================".format(capacity))

        # cache
        env = Cache(dataloader, capacity
                    , feature_selection=('Base', 'UT')
                    , reward_params=dict(name='our', alpha=0.5, psi=50, mu=10, beta=0.3)
                    , allow_skip=False
                    )

        # agents
        agents = {}
        agents['DQN'] = DQNAgent(env.n_actions, env.n_features,
                                 learning_rate=0.01,
                                 reward_decay=0.99,  # 也就是γ伽马，默认0.9

                                 # Epsilon greedy
                                 e_greedy_min=(0.01, 0.1),  # egreedy元组第一个元素是ε1也就是随机动作，第二个元素是ε2也就是跟着lru或lfu
                                 e_greedy_max=(0.1, 0.9),
                                 e_greedy_init=(0.1, 0.8),
                                 e_greedy_increment=(0.005, 0.001),
                                 e_greedy_decrement=(0.005, 0.001),

                                 history_size=50,
                                 dynamic_e_greedy_iter=25,
                                 reward_threshold=5,
                                 explore_mentor='LFU',  # default LRU

                                 replace_target_iter=100,
                                 memory_size=10000,
                                 batch_size=32,  # default:128

                                 output_graph=False,
                                 verbose=0
                                 )
        agents['Random'] = RandomAgent(env.n_actions)
        agents['LRU'] = LRUAgent(env.n_actions)
        agents['LFU'] = LFUAgent(env.n_actions)
        agents['MRU'] = MRUAgent(env.n_actions)

        hrper100setps_DQN = []
        hrper100setps_Random = []
        hrper100setps_LRU = []
        hrper100setps_LFU = []
        hrper100setps_MRU = []

        for (name, agent) in agents.items():

            print("-------------------- %s --------------------" % name)

            step = 0
            episodes = 1
            s = env.reset()

            while True:
                a = agent.choose_action(s)

                s_, r = env.step(a)

                if env.hasDone():
                    break

                if step == 40000 and name == 'DQN':
                    break
                elif step == 20000 and name != 'DQN':
                    break

                agent.store_transition(s, a, r, s_)

                if isinstance(agent, LearnerAgent) and (step > 20) and (step % 5 == 0):
                    agent.learn()
                    # q_target = agent.learn()

                s = s_

                if step % 100 == 0:
                    mr = env.miss_rate()  # calculate missrate
                    hr = 1 - mr  # hitrate
                    # report per 100 steps
                    print('agent={}, step={}, accesses={}, hitrate={}, capacity={}'.format(
                        name, step, env.total_count, hr, capacity
                    ))
                    if name == 'DQN':  # record hitrate per 100 steps
                        hrper100setps_DQN.append(hr)  # lists contained (total step counts)/100
                    if name == 'Random':
                        hrper100setps_Random.append(hr)
                    if name == 'LRU':
                        hrper100setps_LRU.append(hr)
                    if name == 'LFU':
                        hrper100setps_LFU.append(hr)
                    if name == 'MRU':
                        hrper100setps_MRU.append(hr)

                step += 1

            # report every capacity
            # record average hitrate of every capacity
            if name == 'DQN':
                assess_hr_DQN.append(np.mean(hrper100setps_DQN))
                print('agent={}, capacity={}, average hitrate={}'.format(name, capacity, np.max(hrper100setps_DQN)))
            if name == 'Random':
                assess_hr_Random.append(np.mean(hrper100setps_Random))
                print('agent={}, capacity={}, average hitrate={}'.format(name, capacity, np.mean(hrper100setps_Random)))
            if name == 'LRU':
                assess_hr_LRU.append(np.mean(hrper100setps_LRU))
                print('agent={}, capacity={}, average hitrate={}'.format(name, capacity, np.mean(hrper100setps_LRU)))
            if name == 'LFU':
                assess_hr_LFU.append(np.mean(hrper100setps_LFU))
                print('agent={}, capacity={}, average hitrate={}'.format(name, capacity, np.mean(hrper100setps_LFU)))
            if name == 'MRU':
                assess_hr_MRU.append(np.mean(hrper100setps_MRU))
                print('agent={}, capacity={}, average hitrate={}'.format(name, capacity, np.mean(hrper100setps_MRU)))

cachesize_vs_hitrate = [assess_hr_DQN, assess_hr_LRU, assess_hr_LFU, assess_hr_MRU, assess_hr_Random]
df = pd.DataFrame(cachesize_vs_hitrate, columns=['5', '25', '50', '100', '150', '200', '250'],
                  index=['assess_hr_DQN', 'assess_hr_LRU', 'assess_hr_LFU', 'assess_hr_MRU', 'assess_hr_Random'])
print(df)
with open('C:/Users/tashm/Desktop/毕业设计180910437/DRLCaching_python/hitrate_test/hitrate_capacity/2schemes.pkl',
          'wb') as f:
    pickle.dump(df, f)

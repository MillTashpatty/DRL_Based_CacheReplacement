import sys, os
from cache.Cache import Cache
from agents.CacheAgent import *
from agents.DQNAgent import DQNAgent
from agents.ReflexAgent import *
from cache.DataLoader import DataLoaderPintos
import pickle
import pandas as pd

if __name__ == "__main__":
    # disk activities
    file_paths = ["RLCaching_python/dataset_and_generate/zipf_200id_1000000req.csv",
                  "DRLCaching_python/dataset_and_generate/zipf_250id_1000000req.csv",
                  "DRLCaching_python/dataset_and_generate/zipf_300id_1000000req.csv",
                  "DRLCaching_python/dataset_and_generate/zipf_350id_1000000req.csv",
                  "DRLCaching_python/dataset_and_generate/zipf_400id_1000000req.csv",
                  "DRLCaching_python/dataset_and_generate/zipf_450id_1000000req.csv",
                  'DRLCaching_python/dataset_and_generate/zipf_500id_1000000req.csv'
                  ]

    assess_hr_DQN = []
    assess_hr_LRU = []
    assess_hr_LFU = []
    assess_hr_MRU = []
    assess_hr_Random = []

    for path in file_paths:

        #case_name = os.path.basename(path)
        print("==================== {} ====================".format(path))
        capacity = 150
        dataloader = DataLoaderPintos(path)

        env = Cache(dataloader, capacity
                    , feature_selection=('Base', 'UT')
                    , reward_params=dict(name='our', alpha=0.5, psi=50, mu=10, beta=0.3)
                    , allow_skip=False
                    )

        # agents
        agents = {}
        agents['DQN'] = DQNAgent(env.n_actions, env.n_features,
                                 learning_rate=0.01,
                                 reward_decay=0.99,

                                 # Epsilon greedy
                                 e_greedy_min=(0.0, 0.1),
                                 e_greedy_max=(0.1, 0.8),
                                 e_greedy_init=(0.01, 0.8),
                                 e_greedy_increment=(0.005, 0.001),
                                 e_greedy_decrement=(0.005, 0.001),

                                 history_size=50,
                                 dynamic_e_greedy_iter=25,
                                 reward_threshold=5,
                                 explore_mentor='LFU',

                                 replace_target_iter=100,
                                 memory_size=10000,
                                 batch_size=32,

                                 output_graph=False,
                                 verbose=0
                                 )
        agents['Random'] = RandomAgent(env.n_actions)
        agents['LRU'] = LRUAgent(env.n_actions)
        agents['LFU'] = LFUAgent(env.n_actions)
        agents['MRU'] = MRUAgent(env.n_actions)

        hr_perfile_DQN = []
        hr_perfile_Random = []
        hr_perfile_LRU = []
        hr_perfile_LFU = []
        hr_perfile_MRU = []

        for (name, agent) in agents.items():
            print("-------------------- {} --------------------".format(name))
            step = 0
            episodes = 1

            s = env.reset()

            while True:
                # agent choose action based on observation
                a = agent.choose_action(s)

                # agent take action and get next observation and reward
                s_, r = env.step(a)

                # break while loop when end of this episode
                if env.hasDone():
                    break

                if step == 40000 and name == 'DQN':
                    break
                elif step == 20000 and name != 'DQN':
                    break

                agent.store_transition(s, a, r, s_)

                if isinstance(agent, LearnerAgent) and (step > 20) and (step % 5 == 0):
                    agent.learn()

                # swap observation
                s = s_

                if step % 100 == 0:
                    mr = env.miss_rate()
                    hr = 1 - mr
                    # report per 100 steps
                    print('agent={}, step={}, accesses={}, hitrate={}, capacity={}'.format(
                        name, step, env.total_count, hr, capacity
                    ))
                    if name == 'DQN':  # record hitrate per 100 steps
                        hr_perfile_DQN.append(hr)  # lists contained (total step counts)/100
                    if name == 'Random':
                        hr_perfile_Random.append(hr)
                    if name == 'LRU':
                        hr_perfile_LRU.append(hr)
                    if name == 'LFU':
                        hr_perfile_LFU.append(hr)
                    if name == 'MRU':
                        hr_perfile_MRU.append(hr)

                step += 1

            # report every different files, the number of ID in each file is different
            # record average hitrate of every capacity
            if name == 'DQN':
                assess_hr_DQN.append(np.mean(hr_perfile_DQN))
                print(
                    'agent={}, capacity={}, average hitrate={}, file={}'.format(name, capacity,
                                                                                np.max(hr_perfile_DQN),
                                                                                path))
            if name == 'Random':
                assess_hr_Random.append(np.mean(hr_perfile_Random))
                print('agent={}, capacity={}, average hitrate={}, file={}'.format(name, capacity,
                                                                                  np.mean(hr_perfile_Random),
                                                                                  path))
            if name == 'LRU':
                assess_hr_LRU.append(np.mean(hr_perfile_LRU))
                print(
                    'agent={}, capacity={}, average hitrate={}, file={}'.format(name, capacity,
                                                                                np.mean(hr_perfile_LRU),
                                                                                path))
            if name == 'LFU':
                assess_hr_LFU.append(np.mean(hr_perfile_LFU))
                print(
                    'agent={}, capacity={}, average hitrate={}, file={}'.format(name, capacity,
                                                                                np.mean(hr_perfile_LFU),
                                                                                path))
            if name == 'MRU':
                assess_hr_MRU.append(np.mean(hr_perfile_MRU))
                print(
                    'agent={}, capacity={}, average hitrate={}, file={}'.format(name, capacity,
                                                                                np.mean(hr_perfile_MRU),
                                                                                path))

varfiles_vs_hitrate = [assess_hr_DQN, assess_hr_LRU, assess_hr_LFU, assess_hr_MRU, assess_hr_Random]
df = pd.DataFrame(varfiles_vs_hitrate, columns=['200ids', '250ids', '300ids', '350ids', '400ids', '450ids', '500ids'],
                    index=['assess_hr_DQN', 'assess_hr_LRU', 'assess_hr_LFU', 'assess_hr_MRU', 'assess_hr_Random'])
print(df)
with open('C:/Users/tashm/Desktop/毕业设计180910437/DRLCaching_python/hitrate_test/hitrate_variousIDs/zipf_200-500id2.pkl',
            'wb') as f:
    pickle.dump(df, f)

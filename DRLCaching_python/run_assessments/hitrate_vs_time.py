from cache.Cache import Cache
from agents.CacheAgent import *
from agents.DQNAgent import DQNAgent
from agents.ReflexAgent import *
from cache.DataLoader import DataLoaderPintos
import pandas as pd
import pickle

if __name__ == "__main__":

    dataloader = DataLoaderPintos(
        ['DRLCaching_python/dataset_and_generate/zipf_500id_1000000req.csv'])
    env = Cache(dataloader, 100
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
                             e_greedy_min=(0.01, 0.01),  # egreedy元组第一个元素是ε1也就是随机动作，第二个元素是ε2也就是跟着lru或lfu
                             e_greedy_max=(0.1, 0.1),
                             e_greedy_init=(0.15, 0.15),
                             e_greedy_increment=(0.005, 0.001),
                             e_greedy_decrement=(0.005, 0.001),

                             history_size=10000,  # 50 by default
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

    assess_hr_DQN = []
    assess_hr_Random = []
    assess_hr_LRU = []
    assess_hr_LFU = []
    assess_hr_MRU = []

    Time_vs_hitrate = {'assess_hr_DQN': None, 'assess_hr_Random': None, 'assess_hr_LRU': None,
                       'assess_hr_LFU': None, 'assess_hr_MRU': None}

    for (name, agent) in agents.items():

        print("-------------------- {} --------------------".format(name))

        step = 0
        episode = 1

        s = env.reset()

        hrper100setps_DQN = []
        hrper100setps_Random = []
        hrper100setps_LRU = []
        hrper100setps_LFU = []
        hrper100setps_MRU = []

        while True:
            # agent choose action based on observation
            a = agent.choose_action(s)

            # agent take action and get next observation and reward
            s_, r = env.step(a)

            # break while loop when end of this episode
            if env.hasDone():
                break

            if step == 35000:  # to control the steps in order to have the same length of dataframe
                break

            agent.store_transition(s, a, r, s_)

            if isinstance(agent, LearnerAgent) and (step > 20) and (step % 5 == 0):
                agent.learn()

            if isinstance(agent, DQNAgent) and ((step == 25) or (step == 30) or (step == 35) or (step == 40)):
                agent.plot_cost()

                # swap observation
            s = s_

            if step % 100 == 0:
                mr = env.miss_rate()
                hit_rate = 1 - mr
                print("Agent=%s, Episode=%d, Step=%d: Accesses=%d, Misses=%d, HitRate=%f"
                      % (name, episode, step, env.total_count, env.miss_count, hit_rate)
                      )

                if name == 'DQN':  # record hitrate per 100 steps
                    hrper100setps_DQN.append(hit_rate)
                if name == 'Random':
                    hrper100setps_Random.append(hit_rate)
                if name == 'LRU':
                    hrper100setps_LRU.append(hit_rate)
                if name == 'LFU':
                    hrper100setps_LFU.append(hit_rate)
                if name == 'MRU':
                    hrper100setps_MRU.append(hit_rate)

            step += 1

        if name == 'DQN':
            Time_vs_hitrate['assess_hr_DQN'] = hrper100setps_DQN
        if name == 'Random':
            Time_vs_hitrate['assess_hr_Random'] = hrper100setps_Random
        if name == 'LRU':
            Time_vs_hitrate['assess_hr_LRU'] = hrper100setps_LRU
        if name == 'LFU':
            Time_vs_hitrate['assess_hr_LFU'] = hrper100setps_LFU
        if name == 'MRU':
            Time_vs_hitrate['assess_hr_MRU'] = hrper100setps_MRU

Time_vs_hitrate_list = [Time_vs_hitrate['assess_hr_DQN'],
                        Time_vs_hitrate['assess_hr_LRU'],
                        Time_vs_hitrate['assess_hr_LFU'],
                        Time_vs_hitrate['assess_hr_MRU'],
                        Time_vs_hitrate['assess_hr_Random']]

# Time_vs_hissrate = [assess_hr_DQN, assess_hr_LRU, assess_hr_LFU, assess_hr_MRU, assess_hr_Random]
df = pd.DataFrame(Time_vs_hitrate_list,
                  index=['assess_hr_DQN', 'assess_hr_LRU', 'assess_hr_LFU', 'assess_hr_MRU', 'assess_hr_Random'])
# 每一个方法的列表长度为episode，x轴是episode
print(df)
with open('C:/Users/tashm/Desktop/毕业设计180910437/DRLCaching_python/hitrate_test/hitrate_time/zipf_500id_1000000req3.pkl',
          'wb') as f:
    pickle.dump(df, f)

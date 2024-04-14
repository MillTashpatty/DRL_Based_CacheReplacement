import pickle
import pandas as pd
import matplotlib.pyplot as plt

with open('DRLCaching_python/hitrate_test/hitrate_time/zipf_500id_1000000req_test1.pkl', 'rb') as f:
    df = pd.DataFrame(pickle.load(f))
    #file_name = 'zipf_500id_1000000req2.csv'
    #df.to_csv(file_name)
    plt.xlabel('time(per100steps)')
    plt.ylabel('hit rate')
    plt.title('IDs=500, capacity=100, requests=1000000, ε1,2=0.15,0.15')
    plt.plot(list(df.loc['assess_hr_DQN']), color='red', label='DQN', marker='*')
    plt.plot(list(df.loc['assess_hr_LRU']), color='blue', label='LRU', marker='^')
    plt.plot(list(df.loc['assess_hr_LFU']), color='green', label='LFU', marker='.')
    plt.plot(list(df.loc['assess_hr_MRU']), color='black', label='MRU', marker='o')
    plt.plot(list(df.loc['assess_hr_Random']), color='yellow', label='Random', marker='p')
    plt.legend()
    plt.show()

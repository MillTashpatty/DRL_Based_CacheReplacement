import pickle
import pandas as pd
from matplotlib import pyplot as plt

with open('DRLCaching_python/hitrate_test/hitrate_capacity/2schemes.pkl', 'rb') as f:
    df = pd.DataFrame(pickle.load(f))
    #file_name = '2schemes.csv'
    print(df)
    #df.to_csv(file_name)
    labels = ['5', '25', '50', '100', '150', '200', '250']
    plt.xlabel('cache capacity')
    plt.ylabel('hit rate')
    plt.title('capacity_vs_hitrate IDs=300 ε1,ε2=0.1,0.8')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], labels)
    plt.plot(list(df.loc['assess_hr_DQN']), color='red', label='DQN', marker='*')
    plt.plot(list(df.loc['assess_hr_LRU']), color='blue', label='LRU', marker='^')
    plt.plot(list(df.loc['assess_hr_LFU']), color='green', label='LFU', marker='.')
    plt.plot(list(df.loc['assess_hr_MRU']), color='black', label='MRU', marker='o')
    plt.plot(list(df.loc['assess_hr_Random']), color='yellow', label='Random', marker='p')
    plt.legend()
    plt.show()

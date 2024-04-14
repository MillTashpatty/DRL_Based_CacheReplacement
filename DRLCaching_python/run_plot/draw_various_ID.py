import pickle
import pandas as pd
from matplotlib import pyplot as plt

with open('DRLCaching_python/hitrate_test/hitrate_variousIDs/zipf_200-500id2.pkl', 'rb') as f:
    df = pd.DataFrame(pickle.load(f))
    #file_name = 'zipf_200-500id.csv'
    print(df)
    #df.to_csv(file_name)
    labels = ['200ids', '250ids', '300ids', '350ids', '400ids', '450ids', '500ids']
    plt.xlabel('cache capacity')
    plt.ylabel('hit rate')
    plt.title('capacity=150 ε1,ε2=0.01,0.8')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], labels)
    plt.plot(list(df.loc['assess_hr_DQN']), color='red', label='DQN', marker='*')
    plt.plot(list(df.loc['assess_hr_LRU']), color='blue', label='LRU', marker='^')
    plt.plot(list(df.loc['assess_hr_LFU']), color='green', label='LFU', marker='.')
    plt.plot(list(df.loc['assess_hr_MRU']), color='black', label='MRU', marker='o')
    plt.plot(list(df.loc['assess_hr_Random']), color='yellow', label='Random', marker='p')
    plt.legend()
    plt.show()

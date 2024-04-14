import numpy as np
import pandas as pd

if __name__ == "__main__":

    info = input('input as following:saving path,number of IDs,number of requests,distribution parameters,number of '
                 'programs\n'
                 'split with space button:')
    info_list = info.split(' ')
    if len(info_list) == 5:

        save_path = str(info_list[0]+'.csv')
        num_IDs = int(info_list[1])
        num_requests = int(info_list[2])
        param = float(info_list[3])
        num_programs = int(info_list[4])

        df = None
        for i in range(num_programs):  # num_progs = 0,1
            IDs = np.arange(num_IDs)  # files=[0,1,2,3,4]
            ranks = np.random.permutation(IDs) + 1  # ranks=随机排列的ID

            pdf = 1 / np.power(ranks, param)  # 概率分布
            pdf /= np.sum(pdf)

            requests = np.random.choice(IDs, size=num_requests, p=pdf)
            operations = np.full_like(requests, 0)
            executions = np.full_like(requests, 1)

            one_program = pd.DataFrame({'IDs': requests, 'read/write': operations, 'boot/exec': executions})
            df = pd.concat((df, one_program), axis=0)

        df.to_csv(save_path, index=False, header=True)

    else:
        print('input 5 information split with space button.\n'
              'rerun this program to retry')

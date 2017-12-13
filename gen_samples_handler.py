import pandas as pd
import numpy as np
import os


def collect_samples():
    gen_data = np.matrix(pd.read_csv('dataset/gen_samples.csv', header=None))
    gen_data[:, -1] = 1 - gen_data[:, -1]
    if not os.path.exists('dataset/collect.csv'):
        pd.DataFrame(gen_data).to_csv('dataset/collect.csv', header=None, index=False)
        # os.remove('dataset/gen_samples.csv')
        return np.sum(gen_data[:, -1] == 0), np.sum(gen_data[:, -1] == 1)
    else:
        collected_data = np.array(pd.read_csv('dataset/collect.csv', header=None))
        collected_data = np.concatenate((collected_data, gen_data), axis=0)
        pd.DataFrame(collected_data).to_csv('dataset/collect.csv', header=None, index=False)
        # os.remove('dataset/gen_samples.csv')
        return np.sum(collected_data[:, -1] == 0), np.sum(collected_data[:, -1] == 1)


def merge_to_trainset():
    origin_data = pd.read_csv('dataset/kddcup.train.data.sub.csv')
    collected_data = np.array(pd.read_csv('dataset/collect.csv', header=None))
    merged_data = np.concatenate((np.array(origin_data), collected_data), axis=0)
    np.random.shuffle(merged_data)
    pd.DataFrame(merged_data).to_csv('dataset/kddcup.train.data.sub1.csv', index=False, header=origin_data.columns)
    print '%s added to trainset' % collected_data.shape[0]


if __name__ == '__main__':
    merge_to_trainset()

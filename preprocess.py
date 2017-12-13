import pandas as pd
import numpy as np
import random


def drop_and_concat_dummies(data, attr):
    dummies = pd.get_dummies(data[attr], prefix=attr)
    data = pd.concat([data, dummies], axis=1)
    data.drop(attr, axis=1, inplace=True)
    return data


def normalize(data, attr):
    data[attr] = np.log10(data[attr] + 1)
    sub_item = data[attr].min()
    div_item = data[attr].max() - data[attr].min()
    data[attr] = (data[attr] - sub_item) / div_item
    return data


def parse_label(data, attr):
    type_dict = {
        # Type 0: 972781
        'normal.': 0,
        # Type 1: 3883370
        'back.': 1,
        'land.': 1,
        'neptune.': 1,
        'pod.': 1,
        'smurf.': 1,
        'teardrop.': 1,
        # Type 2: 41102
        'ipsweep.': 2,
        'nmap.': 2,
        'portsweep.': 2,
        'satan.': 2,
        # Type 3: 1126
        'ftp_write.': 3,
        'guess_passwd.': 3,
        'imap.': 3,
        'spy.': 3,
        'phf.': 3,
        'multihop.': 3,
        'warezclient.': 3,
        'warezmaster.': 3,
        # Type 4: 52
        'rootkit.': 4,
        'perl.': 4,
        'loadmodule.': 4,
        'buffer_overflow.': 4,
        # Type 5: Only occured in test set
        'apache2.': 5,
        'httptunnel.': 5,
        'snmpgetattack.': 5,
        'named.': 5,
        'xlock.': 5,
        'xsnoop.': 5,
        'sendmail.': 5,
        'saint.': 5,
        'udpstorm.': 5,
        'xterm.': 5,
        'mscan.': 5,
        'processtable.': 5,
        'ps.': 5,
        'worm.': 5,
        'mailbomb.': 5,
        'sqlattack.': 5,
        'snmpguess.': 5
    }
    data[attr] = data[attr].map(lambda c: type_dict[c])
    return data


def get_class(combined, class_num):
    bool_list = np.array(combined['label'] == class_num)
    idx_list = bool_list * np.array(xrange(bool_list.shape[0]))
    idx_list = idx_list[idx_list != 0]
    return combined.iloc[idx_list]


def del_class(combined, class_num):
    bool_list = np.array(combined['label'] != class_num)
    idx_list = bool_list * np.array(xrange(bool_list.shape[0]))
    idx_list = idx_list[idx_list != 0]
    return combined.iloc[idx_list]

print 'Loading data ...'
train_data = pd.read_csv('dataset/kddcup.train.data.csv')
test_data = pd.read_csv('dataset/kddcup.test.data.csv')
combined = train_data.append(test_data)

del train_data
del test_data

print 'Filtering DoS ...'
combined = parse_label(combined, 'label')
combined = del_class(combined, 2)
combined = del_class(combined, 3)
combined = del_class(combined, 4)
combined = del_class(combined, 5)


def symbols_to_step(combined, attr):
    symbols = sorted(list(set(combined[attr].values)))
    step = 1.0 / len(symbols)
    combined[attr] = map(lambda c: symbols.index(c) * step, combined[attr])


print 'Processing discrete attributes ...'
symbols_to_step(combined, 'protocol_type')
symbols_to_step(combined, 'service')
symbols_to_step(combined, 'flag')

print 'Normalizing continous attributes ...'
combined = normalize(combined, 'duration')
combined = normalize(combined, 'src_bytes')
combined = normalize(combined, 'dst_bytes')
combined = normalize(combined, 'wrong_fragment')
combined = normalize(combined, 'urgent')
combined = normalize(combined, 'hot')
combined = normalize(combined, 'num_failed_logins')
combined = normalize(combined, 'num_compromised')
combined = normalize(combined, 'num_root')
combined = normalize(combined, 'num_file_creations')
combined = normalize(combined, 'num_shells')
combined = normalize(combined, 'num_access_files')
combined = normalize(combined, 'count')
combined = normalize(combined, 'srv_count')
combined = normalize(combined, 'dst_host_count')
combined = normalize(combined, 'dst_host_srv_count')

print 'Drop num_outbound_cmds'
combined.drop('num_outbound_cmds', axis=1, inplace=True)

headers = combined.columns

class_0 = get_class(combined, 0)
class_1 = get_class(combined, 1)

del combined

print 'Remove same samples ...'
class_0 = set(map(tuple, np.array(class_0)))
class_0 = np.array(list(class_0))
class_0 = pd.DataFrame(class_0)

class_1 = set(map(tuple, np.array(class_1)))
class_1 = np.array(list(class_1))
class_1 = pd.DataFrame(class_1)

print 'Balancing class ...'
test_random_samples = random.sample(xrange(class_0.shape[0]), 200000)
test_data_from_0 = class_0.iloc[test_random_samples]
class_0.drop(test_random_samples, axis=0, inplace=True)
class_0.reset_index()
class_0 = class_0.iloc[random.sample(xrange(659181), 10000)]

test_random_samples = random.sample(xrange(class_1.shape[0]), 200000)
test_data_from_1 = class_1.iloc[test_random_samples]
class_1.drop(test_random_samples, axis=0, inplace=True)
class_1 = class_1.iloc[random.sample(xrange(57460), 10000)]

test_data_processed = test_data_from_0.append(test_data_from_1)
test_data_processed = test_data_processed.sample(frac=1)

train_data_processed = class_0.append(class_1)
train_data_processed = train_data_processed.sample(frac=1)


def symbols_add_noise(combined, attr):
    symbols = set(combined[attr].values)
    step = 1.0 / len(symbols)
    combined[attr] = map(lambda c: c + random.random() * step, combined[attr])

print 'Add noise to discrete attributes ...'
symbols_add_noise(train_data_processed, 1)  # protocol_type
symbols_add_noise(train_data_processed, 2)  # service
symbols_add_noise(train_data_processed, 3)  # flag
symbols_add_noise(test_data_processed, 1)
symbols_add_noise(test_data_processed, 2)
symbols_add_noise(test_data_processed, 3)

print 'Saving ...'
train_data_processed.to_csv('dataset/kddcup.train.data.sub.csv', index=False, header=headers)
test_data_processed.to_csv('dataset/kddcup.test.data.sub.csv', index=False, header=headers)

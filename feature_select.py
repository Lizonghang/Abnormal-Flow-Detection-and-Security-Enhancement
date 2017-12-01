import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


train_set = pd.read_csv('dataset/kddcup.train.data.preprocessed.csv')
test_set = pd.read_csv('dataset/kddcup.test.data.preprocessed.csv')

train_labels = train_set['label']
test_labels = test_set['label']
train_set.drop('label', axis=1, inplace=True)
test_set.drop('label', axis=1, inplace=True)

# clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
# clf.fit(train_set, train_labels)

# with open('feature_model.pickle', 'wb') as fp:
#     pickle.dump(clf, fp)

with open('feature_model.pickle') as fp:
    clf = pickle.load(fp)

feature = pd.DataFrame()
feature['feature'] = train_set.columns
feature['importance'] = clf.feature_importances_
feature.sort_values(by=['importance'], ascending=True, inplace=True)
feature.set_index('feature', inplace=True)
# feature.plot(kind='barh', figsize=(20, 20))
# plt.savefig('figure1.png')

model = SelectFromModel(clf, prefit=True, threshold=0.1)
train_reduced = model.transform(train_set)
test_reduced = model.transform(test_set)
print 'Dimension after Feature Selection: ', train_reduced.shape[1]

header = feature.index.tolist()[::-1][:train_reduced.shape[1]]
header.append('label')

train_reduced = np.concatenate([train_reduced, np.array(train_labels).reshape((-1, 1))], axis=1)
test_reduced = np.concatenate([test_reduced, np.array(test_labels).reshape((-1, 1))], axis=1)

pd.DataFrame(train_reduced).to_csv('dataset/kddcup.train.data.reduced.csv', index=False, header=header)
pd.DataFrame(test_reduced).to_csv('dataset/kddcup.test.data.reduced.csv', index=False, header=header)

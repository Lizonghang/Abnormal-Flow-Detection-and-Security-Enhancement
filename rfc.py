from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import pickle


class RFC:
    def __init__(self):
        pass

    def fit(self):
        train_set = np.array(pd.read_csv('dataset/kddcup.train.data.preprocessed.csv'))
        train_labels = train_set[:, -1]
        train_set = np.delete(train_set, -1, 1)

        grid_search = GridSearchCV(
            RandomForestClassifier(),
            {'n_estimators': [100]},
            scoring='accuracy'
        )
        grid_search.fit(train_set, train_labels)

        with open('rfc.pickle', 'wb') as fp:
            pickle.dump(grid_search, fp)

        print grid_search.best_params_
        print grid_search.best_score_

    def eval(self):
        test_set = np.array(pd.read_csv('dataset/kddcup.test.data.preprocessed.csv'))
        test_labels = test_set[:, -1]
        test_set = np.delete(test_set, -1, 1)

        with open('rfc.pickle') as fp:
            grid_search = pickle.load(fp)

        predict = grid_search.predict(test_set)
        accuracy = sum(predict == test_labels) / float(len(test_labels))
        print 'Accuracy = {0}% in test set.'.format(accuracy * 100)
        print 'Confusion Matrix:\n', confusion_matrix(test_labels, predict)

    def predict(self, samples):
        assert samples.shape[1:] == [40]
        with open('rfc.pickle') as fp:
            grid_search = pickle.load(fp)

        return grid_search.predict(samples)

    def drop_invalid(self, samples, labels):
        with open('rfc.pickle') as fp:
            grid_search = pickle.load(fp)

        predict = grid_search.predict(samples)
        corr_idx = predict == labels
        return np.concatenate((samples[corr_idx], labels[corr_idx].reshape((-1, 1))), axis=1)

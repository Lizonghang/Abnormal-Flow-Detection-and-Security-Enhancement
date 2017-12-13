import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import argparse
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from keras.layers import Input, Dense
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import Callback


def load_train_data():
    print 'Loading train data ...'
    train_X = np.array(pd.read_csv('dataset/train.sub.features.csv', header=None))
    train_y = np.array(pd.read_csv('dataset/kddcup.train.data.sub.csv')['label'])
    return shuffle(train_X, train_y)


def load_test_data():
    print 'Loading test data ...'
    test_X = np.array(pd.read_csv('dataset/test.sub.features.csv', header=None))
    test_y = np.array(pd.read_csv('dataset/kddcup.test.data.sub.csv')['label'])
    return test_X, test_y


def train():
    # model = Sequential()
    # model.add(Dense(units=256, input_shape=(640,), activation='relu'))
    # model.add(Dense(units=64, activation='relu'))
    # model.add(Dense(units=1, activation='sigmoid'))

    model = load_model('model/model.h5')

    model.compile(
        optimizer=Adam(1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    class EpochSave(Callback):
        def on_train_begin(self, logs=None):
            if not os.path.exists('model'):
                os.mkdir('model')
            print 'initialize model dir'

        def on_epoch_end(self, epoch, logs=None):
            model.save('model/model.h5')

    train_X, train_y = load_train_data()

    model.fit(
        x=train_X,
        y=train_y,
        batch_size=256,
        epochs=200,
        verbose=2,
        callbacks=[EpochSave()]
    )


def eval():
    test_X, test_y = load_test_data()
    model = load_model('model/model.h5')
    metrics = model.evaluate(test_X, test_y)
    metrics_names = model.metrics_names
    for i in range(len(metrics_names)):
        print '{}: {}'.format(metrics_names[i], metrics[i])
    print 'Error:', int(test_X.shape[0] * (1 - metrics[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool)
    parser.add_argument('--eval', type=bool)
    args = parser.parse_args()
    if args.train:
        train()
        eval()
    elif args.eval:
        eval()

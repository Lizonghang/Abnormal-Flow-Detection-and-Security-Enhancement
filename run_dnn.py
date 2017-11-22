import tensorflow as tf
import numpy as np
import pandas as pd
import os
from dnn import DNN

flags = tf.app.flags
flags.DEFINE_integer("epoch", 10, "Epoch to train [10]")
flags.DEFINE_integer("train_size", np.inf, "The limits size of training data [INF]")
flags.DEFINE_integer("batch_size", 64, "The size of batch inputs [64]")
flags.DEFINE_float("lr", 0.0001, "Learning rate for Adam [0.0001]")
flags.DEFINE_string("dataset", "dataset/kddcup.train.data.preprocessed.csv", "The train dataset")
flags.DEFINE_string("testset", "dataset/kddcup.test.data.preprocessed.csv", "The test dataset")
flags.DEFINE_string("predictset", "dataset/gen_samples.csv", "File to predict")
flags.DEFINE_string("checkpoint_dir", "dnn_ckpt", "Dir to save the checkpoint files")
flags.DEFINE_string("log_dir", "dnn_logs", "Dir to save te logs")
flags.DEFINE_boolean("train", False, "True for training")
flags.DEFINE_boolean("predict", False, "True for predicting")
flags.DEFINE_boolean("eval", True, "True for eval")
FLAGS = flags.FLAGS


def main(_):
    assert sum([FLAGS.train, FLAGS.predict, FLAGS.eval]) == 1

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    with tf.Session() as sess:
        dnn = DNN(sess, FLAGS)
        if FLAGS.train:
            dnn.fit()
        elif FLAGS.predict:
            os.environ["CUDA_VISIBLE_DEVICES"] = '1'
            dnn.load_network()
            samples = np.array(pd.read_csv(FLAGS.predictset, header=None))
            _ = samples.copy()
            gen_y = samples[:, -1]
            samples = np.delete(samples, -1, 1)
            predict = dnn.predict(samples)
            assert gen_y.shape[0] == predict.shape[0]
            print 'Accuracy: {}%'.format((predict == gen_y).sum() / float(predict.shape[0]) * 100)
        elif FLAGS.eval:
            os.environ["CUDA_VISIBLE_DEVICES"] = '1'
            dnn.load_network()
            dnn.eval()


if __name__ == '__main__':
    """
    Fit:        python run_dnn.py --train
    Predict:    python run_dnn.py --predict
    Eval:       python run_dnn.py --eval
    """
    tf.app.run()

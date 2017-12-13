import tensorflow as tf
import numpy as np
import pandas as pd
from gan import GAN
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


flags = tf.app.flags
flags.DEFINE_integer("epoch", 10, "Epoch to train [5]")
flags.DEFINE_float("lr", 0.0001, "Learning rate for RMSProp [0.0001]")
flags.DEFINE_float("clip", 0.01, "C of WGAN [0.01]")
flags.DEFINE_integer("z_dim", 100, "Number of noise [100]")
flags.DEFINE_integer("train_size", np.inf, "The limits of train data [INF]")
flags.DEFINE_integer("batch_size", 64, "The size of batch inputs [128]")
flags.DEFINE_string("dataset", "dataset/kddcup.train.data.sub.csv", "Path of dataset file")
flags.DEFINE_string("checkpoint_dir", "gan_ckpt", "Dir to save the checkpoint files")
flags.DEFINE_string("logs_dir", "gan_logs", "Dir to save the logs")
flags.DEFINE_boolean("train", False, "True for training [False]")
flags.DEFINE_boolean("extract", False, "True for extracting features")
FLAGS = flags.FLAGS


def main(_):
    assert sum([FLAGS.train, FLAGS.extract]) == 1

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.logs_dir):
        os.makedirs(FLAGS.logs_dir)

    with tf.Session() as sess:
        if FLAGS.train:
            gan = GAN(sess, FLAGS)
            gan.train()
        elif FLAGS.extract:
            df_X = pd.read_csv('dataset/kddcup.test.data.sub.csv')
            df_X.drop('label', axis=1, inplace=True)
            df_X = np.array(df_X)
            FLAGS.batch_size = df_X.shape[0]
            gan = GAN(sess, FLAGS)
            features = gan.extract_features(df_X)
            pd.DataFrame(features).to_csv('dataset/test.sub.features.csv', header=None, index=False)


if __name__ == '__main__':
    """
    Fit:        python run_gan.py --train
    """
    tf.app.run()

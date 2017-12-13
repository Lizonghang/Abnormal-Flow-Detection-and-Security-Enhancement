import tensorflow as tf
import numpy as np
import pandas as pd
import os
from cgan import CGAN
from rfc import RFC
from gen_samples_handler import collect_samples


flags = tf.app.flags
flags.DEFINE_integer("epoch", 5, "Epoch to train [5]")
flags.DEFINE_float("lr", 0.0001, "Learning rate for RMSProp [0.0001]")
flags.DEFINE_float("clip", 0.01, "C of WGAN [0.01]")
flags.DEFINE_integer("z_dim", 100, "Number of noise [100]")
flags.DEFINE_integer("train_size", np.inf, "The limits of train data [INF]")
flags.DEFINE_integer("batch_size", 128, "The size of batch inputs [128]")
flags.DEFINE_string("dataset", "dataset/kddcup.train.data.preprocessed.csv", "Path of dataset file")
flags.DEFINE_string("checkpoint_dir", "cgan_ckpt", "Dir to save the checkpoint files")
flags.DEFINE_string("logs_dir", "cgan_logs", "Dir to save the logs")
flags.DEFINE_boolean("train", False, "True for training [False]")
flags.DEFINE_boolean("generate", False, "True for generating [False]")
flags.DEFINE_integer("gen_size", 2000, "Num of samples to generate [1280]")
flags.DEFINE_integer("gen_y", None, "Type for generating, 0 or 1, None means random generate [None]")
FLAGS = flags.FLAGS


def main(_):
    assert sum([FLAGS.train, FLAGS.generate]) == 1

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.logs_dir):
        os.makedirs(FLAGS.logs_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    with tf.Session() as sess:
        if FLAGS.train:
            cgan = CGAN(sess, FLAGS)
            cgan.train()
        elif FLAGS.generate:
            cgan = CGAN(sess, FLAGS)
            if os.path.exists('dataset/gen_samples.csv'):
                os.remove('dataset/gen_samples.csv')

            rfc = RFC()

            try:
                collected_data = np.array(pd.read_csv('dataset/collect.csv', header=None))
                num_0 = sum(collected_data[:, -1] == 1)
                num_1 = sum(collected_data[:, -1] == 0)
                del collected_data
            except Exception:
                num_0 = 0
                num_1 = 0

            while num_0 < FLAGS.gen_size / 2:
                fake_samples = np.zeros((FLAGS.batch_size, 41))
                gen_y = [0 for _ in range(FLAGS.batch_size)]
                fake_samples[0: FLAGS.batch_size, :] = cgan.generate(gen_y)
                pd.DataFrame(fake_samples).to_csv('dataset/gen_samples.csv', index=False, header=False)

                os.system('python run_dnn.py --predict')

                try:
                    samples = np.array(pd.read_csv('dataset/gen_samples.csv', header=None))
                    samples = rfc.drop_invalid(samples[:, :40], samples[:, -1])
                    pd.DataFrame(samples).to_csv('dataset/gen_samples.csv', index=False, header=False)
                    num_1, num_0 = collect_samples()
                    print 'Collecting Type 0->1: {}'.format(num_0)
                except Exception:
                    print 'No counter samples found'

            while num_1 < FLAGS.gen_size / 2:
                fake_samples = np.zeros((FLAGS.batch_size, 41))
                gen_y = [1 for _ in range(FLAGS.batch_size)]
                fake_samples[0:FLAGS.batch_size, :] = cgan.generate(gen_y)
                pd.DataFrame(fake_samples).to_csv('dataset/gen_samples.csv', index=False, header=False)

                os.system('python run_dnn.py --predict')

                try:
                    counter_samples = np.array(pd.read_csv('dataset/gen_samples.csv', header=None))
                    counter_samples = rfc.drop_invalid(counter_samples[:, :40], counter_samples[:, -1])
                    pd.DataFrame(counter_samples).to_csv('dataset/gen_samples.csv', index=False, header=False)
                    num_1, num_0 = collect_samples()
                    print 'Collecting Type 1->0: {}'.format(num_1)
                except Exception:
                    print 'No counter samples found'


if __name__ == '__main__':
    """
    Fit:        python run_cgan.py --train
    Generate:   python run_cgan.py --generate
    """
    tf.app.run()

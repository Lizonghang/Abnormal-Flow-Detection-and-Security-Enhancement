import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os
from cgan import CGAN

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
flags.DEFINE_string("gen_samples", "dataset/gen_samples.csv", "Path to save the generated samples")
flags.DEFINE_boolean("train", False, "True for training [False]")
flags.DEFINE_boolean("generate", False, "True for generating [False]")
flags.DEFINE_integer("gen_size", 1280, "Num of samples to generate [1280]")
flags.DEFINE_integer("gen_y", None, "Type for generating, 0 or 1, None means random generate [None]")
FLAGS = flags.FLAGS


def main(_):
    assert sum([FLAGS.train, FLAGS.generate]) == 1

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.logs_dir):
        os.makedirs(FLAGS.logs_dir)

    with tf.Session() as sess:
        cgan = CGAN(sess, FLAGS)
        if FLAGS.train:
            cgan.train()
        elif FLAGS.generate:
            os.environ["CUDA_VISIBLE_DEVICES"] = '1'
            if os.path.exists(FLAGS.gen_samples):
                os.remove(FLAGS.gen_samples)
            gen_size = FLAGS.gen_size / FLAGS.batch_size * FLAGS.batch_size
            fake_samples = np.zeros((gen_size, 41))
            for i in range(gen_size / FLAGS.batch_size):
                if FLAGS.gen_y in [0, 1]:
                    gen_y = [FLAGS.gen_y for _ in range(FLAGS.batch_size)]
                else:
                    gen_y = [random.randint(0, 1) for _ in range(FLAGS.batch_size)]
                fake_samples[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size, :] = cgan.generate(gen_y)
            pd.DataFrame(fake_samples).to_csv(FLAGS.gen_samples, index=False, header=False)
            print '{} samples has saved to {}'.format(gen_size, FLAGS.gen_samples)


if __name__ == '__main__':
    """
    Fit:        python run_dcgan.py --train
    Generate:   python run_dcgan.py --generate
    """
    tf.app.run()

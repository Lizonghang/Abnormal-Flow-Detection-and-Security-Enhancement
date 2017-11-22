import os
import random
import numpy as np
import pandas as pd
from ops import *


class CGAN(object):
    def __init__(self, sess, FLAGS):
        self.sess = sess
        self.FLAGS = FLAGS

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.build_model()

    def build_model(self):
        self.y = tf.placeholder(tf.int32, [self.FLAGS.batch_size, ], name='y')
        self.one_hot_y = tf.one_hot(self.y, 2, 1., 0.)
        self.inputs = tf.placeholder(tf.float32, [self.FLAGS.batch_size, 40], name='real_images')
        self.z = tf.placeholder(tf.float32, [self.FLAGS.batch_size, self.FLAGS.z_dim], name='z')

        self.G = self.generator(self.z, self.one_hot_y)

        self.D, self.D_logits = self.discriminator(self.inputs, self.one_hot_y, reuse=False)

        self.D_, self.D_logits_ = self.discriminator(self.G, self.one_hot_y, reuse=True)

        self.sampler_op = self.sampler(self.z, self.one_hot_y)

        self.d_loss_real = tf.reduce_mean(tf.scalar_mul(-1, self.D_logits))
        self.d_loss_fake = tf.reduce_mean(self.D_logits_)
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.scalar_mul(-1, self.D_logits_))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

    def train(self):
        d_optim = tf.train.RMSPropOptimizer(self.FLAGS.lr).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.RMSPropOptimizer(self.FLAGS.lr).minimize(self.g_loss, var_list=self.g_vars)

        clip_d_op = [var.assign(tf.clip_by_value(var, -self.FLAGS.clip, self.FLAGS.clip)) for var in self.d_vars]

        self.sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(self.FLAGS.logs_dir, self.sess.graph)
        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        print 'Loading trainset ...'
        df_train = pd.read_csv(self.FLAGS.dataset)
        labels = df_train['label'].tolist()
        df_train.drop('label', axis=1, inplace=True)
        df_train = np.array(df_train)
        print 'Loading Succeed'

        max_train_batch_idx = min(self.FLAGS.train_size, df_train.shape[0]) / self.FLAGS.batch_size

        counter = 0
        for epoch in xrange(self.FLAGS.epoch):
            for idx in xrange(0, max_train_batch_idx):
                if idx < 25:
                    critic_num = 25
                else:
                    critic_num = 5

                batch_inputs = df_train[idx * self.FLAGS.batch_size: (idx + 1) * self.FLAGS.batch_size, :]
                batch_labels = np.array(labels[idx * self.FLAGS.batch_size: (idx + 1) * self.FLAGS.batch_size])\
                    .reshape([self.FLAGS.batch_size, ])

                for _ in range(critic_num):
                    batch_z = np.random.uniform(-1, 1, [self.FLAGS.batch_size, self.FLAGS.z_dim]).astype(np.float32)
                    self.sess.run(d_optim, feed_dict={
                        self.inputs: batch_inputs,
                        self.y: batch_labels,
                        self.z: batch_z
                    })
                    self.sess.run(clip_d_op)

                batch_z = np.random.uniform(-1, 1, [self.FLAGS.batch_size, self.FLAGS.z_dim]).astype(np.float32)
                batch_labels = [random.randint(0, 1) for _ in range(self.FLAGS.batch_size)]

                # Update G network
                self.sess.run(g_optim, feed_dict={
                    self.z: batch_z,
                    self.y: batch_labels
                })

                errD_fake = self.d_loss_fake.eval({
                    self.z: batch_z,
                    self.y: batch_labels
                })
                errD_real = self.d_loss_real.eval({
                    self.inputs: batch_inputs,
                    self.y: batch_labels
                })
                errG = self.g_loss.eval({
                    self.z: batch_z,
                    self.y: batch_labels
                })

                counter += 1
                print("Epoch: [%2d] [%4d/%4d], d_loss: %.8f, g_loss: %.8f" %
                      (epoch, idx, max_train_batch_idx, errD_fake + errD_real, errG))

                if np.mod(counter, 100) == 0:
                    writer.add_summary(self.sess.run(summary_op, feed_dict={
                        self.inputs: batch_inputs,
                        self.y: batch_labels,
                        self.z: batch_z
                    }), counter)

                    # fake_samples = np.zeros((1024, 41))
                    # for i in range(1024 / self.FLAGS.batch_size):
                    #     gen_y = [random.randint(0, 1) for _ in range(self.FLAGS.batch_size)]
                    #     fake_samples[i*self.FLAGS.batch_size:(i+1)*self.FLAGS.batch_size, :] = self.generate(gen_y)
                    # pd.DataFrame(fake_samples).to_csv('tmp/gen_samples_{}.csv'.format(counter), index=False, header=False)

            saver.save(self.sess, os.path.join(self.FLAGS.checkpoint_dir, 'cgan.ckpt'), global_step=epoch)

    def discriminator(self, inputs, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, [self.FLAGS.batch_size, 1, 2])
            x = tf.reshape(inputs, [self.FLAGS.batch_size, 40, 1])
            x = conv1d_cond_concat(x, yb)
            assert x.shape == [self.FLAGS.batch_size, 40, 3]

            h0 = lrelu(conv1d(x, 32, name='d_h0_conv'))
            h0 = conv1d_cond_concat(h0, yb)
            assert h0.shape == [self.FLAGS.batch_size, 20, 34]

            h1 = lrelu(self.d_bn1(conv1d(h0, 64, name='d_h1_conv')))
            assert h1.shape == [self.FLAGS.batch_size, 10, 64]
            h1 = tf.reshape(h1, [self.FLAGS.batch_size, -1])
            h1 = tf.concat([h1, y], 1)
            assert h1.shape == [self.FLAGS.batch_size, 642]

            h2 = lrelu(self.d_bn2(linear(h1, 64, 'd_h2_lin')))
            h2 = tf.concat([h2, y], 1)
            assert h2.shape == [self.FLAGS.batch_size, 66]

            h3 = linear(h2, 1, 'd_h3_lin')
            return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        assert z.shape == [self.FLAGS.batch_size, self.FLAGS.z_dim]
        assert y.shape == [self.FLAGS.batch_size, 2]

        with tf.variable_scope("generator") as scope:
            yb = tf.reshape(y, [self.FLAGS.batch_size, 1, 1, 2])
            z = tf.concat([z, y], 1)
            assert z.shape == [self.FLAGS.batch_size, self.FLAGS.z_dim+2]

            h0 = tf.nn.relu(self.g_bn0(linear(z, 128, 'g_h0_lin')))
            h0 = tf.concat([h0, y], 1)
            assert h0.shape == [self.FLAGS.batch_size, 130]

            h1 = tf.nn.relu(self.g_bn1(linear(h0, 320, 'g_h1_lin')))
            h1 = tf.reshape(h1, [self.FLAGS.batch_size, 1, 10, 32])
            h1 = conv2d_cond_concat(h1, yb)
            assert h1.shape == [self.FLAGS.batch_size, 1, 10, 34]

            h2 = tf.nn.relu(self.g_bn2(deconv1d(h1, [self.FLAGS.batch_size, 1, 20, 16], name='g_h2_conv')))
            h2 = conv2d_cond_concat(h2, yb)
            assert h2.shape == [self.FLAGS.batch_size, 1, 20, 18]

            h3 = deconv1d(h2, [self.FLAGS.batch_size, 1, 40, 1], name='g_h3_conv')
            h3 = tf.reshape(h3, [self.FLAGS.batch_size, 40])

            return tf.nn.sigmoid(h3)

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            yb = tf.reshape(y, [self.FLAGS.batch_size, 1, 1, 2])
            z = tf.concat([z, y], 1)
            assert z.shape == [self.FLAGS.batch_size, self.FLAGS.z_dim+2]

            h0 = tf.nn.relu(self.g_bn0(linear(z, 128, 'g_h0_lin')))
            h0 = tf.concat([h0, y], 1)
            assert h0.shape == [self.FLAGS.batch_size, 130]

            h1 = tf.nn.relu(self.g_bn1(linear(h0, 320, 'g_h1_lin')))
            h1 = tf.reshape(h1, [self.FLAGS.batch_size, 1, 10, 32])
            h1 = conv2d_cond_concat(h1, yb)
            assert h1.shape == [self.FLAGS.batch_size, 1, 10, 34]

            h2 = tf.nn.relu(self.g_bn2(deconv1d(h1, [self.FLAGS.batch_size, 1, 20, 16], name='g_h2_conv')))
            h2 = conv2d_cond_concat(h2, yb)
            assert h2.shape == [self.FLAGS.batch_size, 1, 20, 18]

            h3 = deconv1d(h2, [self.FLAGS.batch_size, 1, 40, 1], name='g_h3_conv')
            h3 = tf.reshape(h3, [self.FLAGS.batch_size, 40])

            return tf.nn.sigmoid(h3)

    def generate(self, gen_y):
        saver = tf.train.Saver(tf.global_variables())
        if not self.load_checkpoint(saver):
            raise Exception("[ERROR] No checkpoint file found!")

        sample_z = np.random.uniform(-1, 1, [self.FLAGS.batch_size, self.FLAGS.z_dim]).astype(np.float32)
        samples = self.sess.run(self.sampler_op, feed_dict={self.z: sample_z, self.y: gen_y})
        return np.concatenate([samples, np.array(gen_y).reshape((self.FLAGS.batch_size, 1))], axis=1)

    def load_checkpoint(self, saver):
        import re
        ckpt = tf.train.get_checkpoint_state(self.FLAGS.checkpoint_dir)
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(self.FLAGS.checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            return True, counter
        else:
            return False, 0

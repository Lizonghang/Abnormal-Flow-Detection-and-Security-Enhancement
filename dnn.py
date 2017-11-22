import os
import numpy as np
import pandas as pd
import tensorflow as tf
from ops import linear


class DNN(object):

    def __init__(self, sess, FLAGS):
        self.sess = sess
        self.FLAGS = FLAGS

    def inference(self, inputs):
        h1 = tf.nn.relu(linear(inputs, 512, 'h1_lin'))
        h2 = tf.nn.relu(linear(h1, 256, 'h2_lin'))
        h3 = tf.nn.relu(linear(h2, 128, 'h3_lin'))
        return linear(h3, 2, 'h4_lin')

    def calculate_loss(self, logits, one_hot_labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels))
        tf.summary.scalar('loss', loss)
        return loss

    def train(self, loss, global_step):
        return tf.train.AdamOptimizer(self.FLAGS.lr).minimize(loss, global_step)

    def fit(self):
        global_step = tf.Variable(0, trainable=False)

        inputs = tf.placeholder(tf.float32, [None, 40], name='batch_inputs')
        labels = tf.placeholder(tf.int32, [None, ], name='batch_labels')
        one_hot_labels = tf.one_hot(labels, 2, 1, 0)

        logits = self.inference(inputs)

        loss = self.calculate_loss(logits, one_hot_labels)

        train_op = self.train(loss, global_step)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        self.sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(self.FLAGS.log_dir, self.sess.graph)
        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(tf.global_variables())

        print 'Loading trainset ...'
        train_data = pd.read_csv(self.FLAGS.dataset)
        train_label = train_data['label'].astype(np.int32).tolist()
        train_data.drop('label', axis=1, inplace=True)
        train_data = train_data.as_matrix()
        print 'Loading testset ...'
        test_data = pd.read_csv(self.FLAGS.testset)
        test_label = test_data['label'].astype(np.int32).tolist()
        test_data.drop('label', axis=1, inplace=True)
        test_data = test_data.as_matrix()
        print 'Load Success'

        train_size = min(train_data.shape[0], self.FLAGS.train_size)
        max_train_batch_idx = train_size / self.FLAGS.batch_size

        for epoch in range(self.FLAGS.epoch):
            for batch_idx in xrange(max_train_batch_idx):
                batch_samples = train_data[batch_idx * self.FLAGS.batch_size: (batch_idx + 1) * self.FLAGS.batch_size, :]
                batch_labels = train_label[batch_idx * self.FLAGS.batch_size: (batch_idx + 1) * self.FLAGS.batch_size]
                self.sess.run(train_op, feed_dict={inputs: batch_samples, labels: batch_labels})

            acc, sum_op = self.sess.run([accuracy, summary_op], feed_dict={inputs: test_data, labels: test_label})
            saver.save(self.sess, os.path.join(self.FLAGS.checkpoint_dir, 'dnn.ckpt'), global_step=self.sess.run(global_step))
            writer.add_summary(sum_op, global_step=self.sess.run(global_step))
            print 'Accuracy = {0}% at epoch {1}'.format(acc * 100, epoch)

    def load_network(self):
        self.global_step = tf.Variable(0, trainable=False)

        self.inputs = tf.placeholder(tf.float32, [None, 40], name='batch_inputs')
        self.labels = tf.placeholder(tf.int32, [None, ], name='batch_labels')
        self.one_hot_labels = tf.one_hot(self.labels, 2, 1, 0)

        self.logits = self.inference(self.inputs)

        self.loss = self.calculate_loss(self.logits, self.one_hot_labels)

        self.train_op = self.train(self.loss, self.global_step)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.saver = tf.train.Saver(tf.global_variables())

        if not self.load_checkpoint(self.saver):
            raise Exception("[ERROR] No checkpoint file found!")

    def predict(self, samples):
        logits_ = self.sess.run(self.logits, feed_dict={self.inputs: samples})
        return logits_.argmax(axis=1)

    def eval(self):
        print 'Loading testset ...'
        test_data = pd.read_csv(self.FLAGS.testset)
        test_labels = test_data['label'].tolist()
        test_data.drop('label', axis=1, inplace=True)
        test_data = test_data.as_matrix()
        print 'Load Success'

        acc = self.sess.run(self.accuracy, feed_dict={self.inputs: test_data, self.labels: test_labels})
        print 'Accuracy = {0}% in test set.'.format(acc * 100)

    def load_checkpoint(self, saver):
        ckpt = tf.train.get_checkpoint_state(self.FLAGS.checkpoint_dir)
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(self.FLAGS.checkpoint_dir, ckpt_name))
            return True
        else:
            return False

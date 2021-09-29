import tensorflow as tf
import numpy as np

data_x = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]], dtype=np.float32)

data_y = np.array([[0],
                     [1],
                     [1],
                     [0]], dtype=np.float32)


class NN:
    def __init__(
            self,
            learning_rate=0.05,
            episodes=10000
    ):
        self.lr = learning_rate
        self.loss = None
        self.train = None
        self.x = None
        self.y = None
        self.episodes = episodes

        self.build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        # define input and output
        self.x = tf.placeholder(tf.float32, [None, 2], name='input')
        self.y = tf.placeholder(tf.float32, [None, 1], name='output')

        # define variables of layer_1
        with tf.name_scope('layer_1'):
            w1 = tf.Variable(tf.random_uniform([2, 4], -1, 1), name='w1')
            b1 = tf.Variable(tf.zeros(4, dtype=np.float32), name='b1')
            l1 = tf.sigmoid(tf.matmul(self.x, w1) + b1)

        # define variables of layer_2
        with tf.name_scope('layer_2'):
            w2 = tf.get_variable('w2', shape=[4, 1], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.zeros(1, dtype=np.float32), name='b2')
            l2 = tf.sigmoid(tf.matmul(l1, w2) + b2)

        # define loss function
        with tf.name_scope('loss'):
            self.loss = tf.nn.l2_loss(l2 - self.y)

        # define optimizer
        with tf.name_scope('train'):
            opt = tf.train.GradientDescentOptimizer(self.lr)
            self.train = opt.minimize(self.loss)

    def run(self):
        for iter in range(self.episodes):
            loss_val, _ = self.sess.run([self.loss, self.train], feed_dict={self.x: data_x, self.y: data_y})
            if iter % 1000 == 0:
                print('{}:loss = {}'.format(iter, loss_val))

    def show_graph(self):
        writer = tf.compat.v1.summary.FileWriter("logs/", self.sess.graph)
        writer.close()


if __name__ == '__main__':
    net = NN(episodes=10000)
    net.run()
    net.show_graph()

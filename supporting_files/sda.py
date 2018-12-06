from __future__ import print_function

import numpy as np
import tensorflow as tf
from supporting_files.helpers import optimize

class StackedDenoisingAutoencoder:
    """A stacked deep autoencoder with denoising capability"""

    def __init__(self, dims=[100,100,100], epochs=[100,100,100], activations=['sigmoid']*3, noise=None, loss='rmse', lr=0.001, batch_size=100, print_step=10, weight_init='default', optimizer="Adam", decay='none', verbose=True):
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epochs = epochs
        self.dims = dims
        self.depth = len(dims)
        self.optimizer = optimizer
        self.decay = decay
        self.weight_init = weight_init
        self.weights, self.biases = [], []
        self.weights_enc, self.biases_enc = [], []
        self.weights_dec, self.biases_dec = [], []
        self.verbose = verbose
        # assert len(dims) == len(epochs)

    def _fit(self, x):
        #modification: only run for half the weight layers
        #use the decoder weight values for the latter half
        for i in range(self.depth//2):
            if(self.verbose):
                print('Layer {0}'.format(i + 1))
            x = self._run(data_x=self._add_noise(x), activation=self.activations[i], data_x_=x,
                         hidden_dim=self.dims[i], epochs=self.epochs[i], loss=self.loss, 
                         batch_size=self.batch_size, lr=self.lr, print_step=self.print_step,
                         weight_init=self.weight_init, optimizer=self.optimizer, decay=self.decay)
        self.weights = self.weights_enc+list(reversed(self.weights_dec))
        self.biases = self.biases_enc+list(reversed(self.biases_dec))
    
    def _add_noise(self, x):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if self.noise == 'mask':
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), round(frac * len(i)), replace=False)
                i[n] = 0
            return temp
        if self.noise == None:
            return x

    def _transform(self, data):
        sess = tf.Session()
        x = tf.constant(data, dtype=tf.float32)
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)
        return x.eval(session=sess)

    def get_transformed_data(self, x):
        self._fit(x)
        return self._transform(x)

    def _run(self, data_x, data_x_, hidden_dim, activation, loss, lr, print_step, epochs, batch_size, weight_init, optimizer, decay):
        input_dim = len(data_x[0])
        if(self.verbose):
            print(str(input_dim) + " -> " + str(hidden_dim))
        sess = tf.Session()
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_')
        if weight_init == 'uniform':
            r = 4*np.sqrt(6.0/(input_dim+hidden_dim))
            encode = {'weights': tf.Variable(tf.random_uniform([input_dim, hidden_dim], minval=-r, maxval=r, dtype=tf.float32)),
                      'biases': tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32))}
            decode = {'biases': tf.Variable(tf.zeros([input_dim], dtype=tf.float32)),
                      'weights': tf.Variable(tf.transpose(encode['weights'].initialized_value()))}
        elif weight_init == 'default':
            encode = {'weights': tf.Variable(tf.truncated_normal([input_dim, hidden_dim], dtype=tf.float32)),
                      'biases': tf.Variable(tf.truncated_normal([hidden_dim], dtype=tf.float32))}
            decode = {'biases': tf.Variable(tf.truncated_normal([input_dim], dtype=tf.float32)),
                      'weights': tf.Variable(tf.transpose(encode['weights'].initialized_value()))}

        encoded = self.activate(tf.matmul(x, encode['weights']) + encode['biases'], activation)
        decoded = tf.matmul(encoded, decode['weights']) + decode['biases']

        # reconstruction loss
        if loss == 'rmse':
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_, decoded))))
        elif loss == 'cross-entropy':
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(decoded, x_))
        train_op = optimize(loss, lr, optimizer, decay, tf.Variable(1, dtype=tf.float32, trainable=False))

        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            b_x, b_x_ = self._get_batch(data_x, data_x_, batch_size)
            sess.run(train_op, feed_dict={x: b_x, x_: b_x_})
            if i % print_step == 0:
                l = sess.run(loss, feed_dict={x: data_x, x_: data_x_})
                if(self.verbose):
                    print('epoch {0}: global loss = {1}'.format(i, l))
        # debug
        #print('Decoded', sess.run(decoded, feed_dict={x: data_x, x_: data_x_})[0])
        self.weights_enc.append(sess.run(encode['weights']))
        self.biases_enc.append(sess.run(encode['biases']))
        self.weights_dec.append(sess.run(decode['weights']))
        self.biases_dec.append(sess.run(decode['biases']))
        return sess.run(encoded, feed_dict={x: data_x_})

    def _get_batch(self, X, X_, size):
        a = np.random.choice(len(X), size, replace=False)
        return X[a], X_[a]

    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')

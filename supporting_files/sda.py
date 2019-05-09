from __future__ import print_function

import numpy as np
import tensorflow as tf
from supporting_files.helpers import optimize

class StackedDenoisingAutoencoder:
    """A stacked deep autoencoder with denoising capability"""

    def __init__(self, dims=[100,100,100], epochs_max=[100,100,100], activations=['sigmoid']*3,
                 noise=None, loss='rmse', lr=0.001, batch_num=1, print_step=10, validation_step=-1,
                 stop_crteria=-1, weight_init='default', optimizer="Adam", decay='none', verbose=True):
        self.print_step = print_step
        self.validation_step = validation_step
        self.stop_criteria = stop_crteria
        self.batch_num = batch_num
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epochs = epochs_max
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

    def _fit(self, x, x_val=None):
        #modification: only run for half the weight layers
        #use the decoder weight values for the latter half
        loss_total = 0
        for i in range(self.depth//2):
            if(self.verbose):
                print('Layer {0}'.format(i + 1))
            x, x_val, loss = self._run(data_x=self._add_noise(x), data_val=self._add_noise(x_val), activation=self.activations[i], data_x_=x,
                                 data_val_=x_val, hidden_dim=self.dims[i], epochs=self.epochs[i], loss=self.loss, 
                                 batch_num=self.batch_num, lr=self.lr, print_step=self.print_step, validation_step=self.validation_step,
                                 stop_criteria=self.stop_criteria, weight_init=self.weight_init, optimizer=self.optimizer, decay=self.decay)
            loss_total += loss
        self.weights = self.weights_enc+list(reversed(self.weights_dec))
        self.biases = self.biases_enc+list(reversed(self.biases_dec))
        print('Final combined validation loss = {0}'.format(loss_total))
        return loss_total
    
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

    def _run(self, data_x, data_x_, data_val, data_val_, hidden_dim, activation, loss, lr, print_step, validation_step, stop_criteria, epochs, batch_num, weight_init, optimizer, decay):
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
        elif weight_init == 'normal':
            r = np.sqrt(2.0/(input_dim+hidden_dim))
            encode = {'weights': tf.Variable(tf.random_normal([input_dim, hidden_dim], mean=0.0, stddev=r, dtype=tf.float32)),
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
        loss_v_best = float("inf")
        loss_v_prev = float("inf")
        consec_increases = 0
        enc_best = dec_best = None
        for i in range(epochs):
            bs_x, bs_x_ = self._get_batches(data_x, data_x_, batch_num)
            for b_x, b_x_ in zip(bs_x, bs_x_):
                sess.run(train_op, feed_dict={x: b_x, x_: b_x_})
            if print_step > 0:
                if i % print_step == 0:
                    loss_g = sess.run(loss, feed_dict={x: data_x, x_: data_x_})
                    if(self.verbose):
                        print('epoch {0}: global loss = {1}'.format(i, loss_g))
            if data_val is not None and validation_step > 0:
                if i % validation_step == 0:
                    loss_v, enc, dec = sess.run([loss, encode, decode], feed_dict={x: data_val, x_: data_val_})
                    if(self.verbose):
                        print('epoch {0}: validation loss = {1}'.format(i, loss_v))
                    if(loss_v < loss_v_best):
                        # save reference to best weights
                        loss_v_best = loss_v
                        enc_best = enc
                        dec_best = dec
                    if(stop_criteria > 0):
                        if(loss_v > loss_v_prev):
                            consec_increases += 1
                            if(consec_increases >= stop_criteria):
                                print("Training stopped after {0} epochs with loss = {1}".format(i, loss_v_best))
                                break
                        else:
                            consec_increases = 0
                        loss_v_prev = loss_v
        if(stop_criteria <= 0 or consec_increases < stop_criteria):
            print("Training exceeded max limit of {0} epochs".format(i+1))

        # debug
        #print('Decoded', sess.run(decoded, feed_dict={x: data_x, x_: data_x_})[0])
        if enc_best is None:
            enc_best, dec_best = sess.run([encode, decode])
        self.weights_enc.append(enc_best['weights'])
        self.biases_enc.append(enc_best['biases'])
        self.weights_dec.append(dec_best['weights'])
        self.biases_dec.append(dec_best['biases'])
        # replace current weights with best
        sess.run([tf.assign(encode['weights'], tf.convert_to_tensor(enc_best['weights'], dtype=tf.float32)),
                  tf.assign(encode['biases'], tf.convert_to_tensor(enc_best['biases'], dtype=tf.float32)),
                  tf.assign(decode['weights'], tf.convert_to_tensor(dec_best['weights'], dtype=tf.float32)),
                  tf.assign(decode['biases'], tf.convert_to_tensor(dec_best['biases'], dtype=tf.float32))])
        return sess.run(encoded, feed_dict={x: data_x_}), sess.run(encoded, feed_dict={x: data_val_}) if data_val_ is not None else None, loss_v_best

    def _get_batch(self, X, X_, size):
        a = np.random.choice(len(X), size, replace=False)
        return X[a], X_[a]

    def _get_batches(self, X, X_, batch_num):
        indx = np.array(range(len(X)))
        np.random.shuffle(indx)
        indx_split = np.array_split(indx, batch_num)
        return [X[indx] for indx in indx_split], [X_[indx] for indx in indx_split]

    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')

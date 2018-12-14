import supporting_files.sda
import tensorflow as tf
import numpy as np
import os
from supporting_files.nncomponents import *
from supporting_files.helpers import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class DeepSubspaceClustering:

    def __init__(self, inputX, C=None, hidden_dims=[300,150,300], lambda1=0.01, lambda2=0.01, activation='tanh', \
                 weight_init='uniform', noise=None, learning_rate=0.1, optimizer='Adam', decay='none', \
                 sda_optimizer='Adam', sda_decay='none', weight_init_params=[100, 0.001, 100, 100], seed=None, verbose=True):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.verbose = verbose

        self.noise = noise
        n_sample, n_feat = inputX.shape

        # M must be a even number
        assert len(hidden_dims) % 2 == 1

        # Add the end layer
        hidden_dims.append(n_feat)

        # self.depth = len(dims)

        # This is not the symbolic variable of tensorflow, this is real!
        self.inputX = inputX

        self.inputC = C

        self.C = tf.placeholder(dtype=tf.float32, shape=[None, None], name='C')

        self.hidden_layers = []
        self.X = self._add_noise(tf.placeholder(dtype=tf.float32, shape=[None, n_feat], name='X'))

        input_hidden = self.X
        weights, biases = self.init_layer_weight(weight_init, hidden_dims, [weight_init_params[0]]*len(hidden_dims),
                                                 [activation]*len(hidden_dims), lr=weight_init_params[1],
                                                 batch_size=weight_init_params[2], sda_optimizer=sda_optimizer,
                                                 sda_decay=sda_decay, sda_printstep=weight_init_params[3])

        # J3 regularization term
        J3_list = []
        for init_w, init_b in zip(weights, biases):
            self.hidden_layers.append(DenseLayer(input_hidden, init_w, init_b, activation=activation))
            input_hidden = self.hidden_layers[-1].output
            J3_list.append(tf.reduce_mean(tf.square(self.hidden_layers[-1].w)))
            J3_list.append(tf.reduce_mean(tf.square(self.hidden_layers[-1].b)))

        J3 = lambda2 * tf.add_n(J3_list)

        self.H_M = self.hidden_layers[-1].output
        # H(M/2) the output of the mid layer
        self.H_M_2 = self.hidden_layers[int((len(hidden_dims)-1)/2)].output

        # calculate loss J1
        # J1 = tf.nn.l2_loss(tf.subtract(self.X, self.H_M))

        J1 = tf.reduce_mean(tf.square(tf.subtract(self.X, self.H_M)))

        # calculate loss J2
        J2 = lambda1 * tf.reduce_mean(tf.square(tf.subtract(tf.transpose(self.H_M_2), \
                                     tf.matmul(tf.transpose(self.H_M_2), self.C))))

        self.cost = J1 + J2 + J3

        self.global_step = tf.Variable(1, dtype=tf.float32, trainable=False)
        self.optimizer = optimize(self.cost, learning_rate, optimizer, decay, self.global_step)

    def init_layer_weight(self, name, dims, epochs, activations, noise=None, loss='rmse', lr=0.001, batch_size=100, sda_optimizer='Adam', sda_decay='none', sda_printstep=100):
        weights, biases = [], []
        if name == 'sda-uniform':
            sda = supporting_files.sda.StackedDenoisingAutoencoder(dims, epochs, activations, noise, loss, lr, batch_size, sda_printstep, 'uniform', sda_optimizer, sda_decay, self.verbose)
            sda._fit(self.inputX)
            weights, biases = sda.weights, sda.biases
        elif name == 'sda':
            sda = supporting_files.sda.StackedDenoisingAutoencoder(dims, epochs, activations, noise, loss, lr, batch_size, sda_printstep, 'default', sda_optimizer, sda_decay, self.verbose)
            sda._fit(self.inputX)
            weights, biases = sda.weights, sda.biases
        elif name == 'uniform':
            n_in = self.inputX.shape[1]
            for d in dims:
                r = 4*np.sqrt(6.0/(n_in+d))
                weights.append(tf.Variable(tf.random_uniform([n_in, d], minval=-r, maxval=r)))
                biases.append(tf.Variable(tf.zeros([d,])))
                n_in = d

        return weights, biases

    def train(self, batch_size=100, epochs=100, print_step=100):
        if(self.verbose):
            print()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        batch_generator = GenBatch(self.inputX, C=self.inputC, batch_size=batch_size)
        n_batch = batch_generator.n_batch

        self.losses = []
        for i in range(epochs):
            # x_batch, y_batch = get_batch(self.X_train, self.y_train, batch_size)
            batch_generator.resetIndex()
            for j in range(int(n_batch+1)):
                x_batch, c_batch = batch_generator.get_batch()
                out=sess.run(self.optimizer, feed_dict={self.X: x_batch, self.C: c_batch})

            self.losses.append(sess.run(self.cost, feed_dict={self.X: x_batch, self.C: c_batch}))

            if(self.verbose and i % print_step == 0):
                print('epoch {0}: global loss = {1}'.format(i, self.losses[-1]))

        # for i in range(1, epochs+1):
        #     x_batch, c_batch = get_batch_XC(self.inputX, self.inputC, batch_size)  
        #     self.losses.append(sess.run(self.cost, feed_dict={self.X: x_batch, self.C: c_batch}))      
        #     if i % print_step == 0:
        #         print('epoch {0}: global loss = {1}'.format(i, self.losses[-1]))

        self.result, self.reconstr = sess.run([self.H_M_2, self.H_M], feed_dict={self.X: x_batch, self.C: c_batch})


    def _add_noise(self, x):
        if self.noise is None:
            return x
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

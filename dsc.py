import supporting_files.sda
import tensorflow as tf
import numpy as np
import os
from supporting_files.nncomponents import *
from supporting_files.helpers import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# print any active GPUs
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.close() 

class DeepSubspaceClustering:

    def __init__(self, inputX, load_path=None, save_path=None, C=None, trainC=False, hidden_dims=[300,150,300], \
                 activation='tanh', weight_init='uniform', noise=None, sda_optimizer='Adam', sda_decay='none', \
                 weight_init_params=[100, 0.001, 100, 100], seed=None, verbose=True):
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

        self.givenC = not C is None
        self.trainC = trainC
        if self.givenC:
            self.inputC = C
        else:
            self.inputC = np.random.normal(0.0, np.sqrt(1.0 / self.inputX.shape[0]), (self.inputX.shape[0], self.inputX.shape[0]))
            self.inputC -= np.diag(np.diag(self.inputC))

        self.C = tf.placeholder(dtype=tf.float32, shape=[None, None], name='C')
        self.trainedC = tf.matrix_set_diag(tf.Variable(self.inputC, dtype=tf.float32), tf.zeros(self.inputX.shape[0]))

        self.hidden_layers = []
        self.X = self._add_noise(tf.placeholder(dtype=tf.float32, shape=[None, n_feat], name='X'))

        input_hidden = self.X
        if (load_path is None):
            weights, biases = self.init_layer_weight(weight_init, hidden_dims, [weight_init_params[0]]*len(hidden_dims),
                                                     [activation]*len(hidden_dims), save_path=save_path, lr=weight_init_params[1],
                                                     batch_size=weight_init_params[2], sda_optimizer=sda_optimizer,
                                                     sda_decay=sda_decay, sda_printstep=weight_init_params[3])
            if(save_path is not None):
                np.savez(save_path, *weights, *biases)
                print("\nModel saved to " + save_path + '.npz')
        else:
            npzfile = np.load(load_path+'.npz')
            ndarrays = [npzfile['arr_'+str(i)] for i in range(len(npzfile))]
            npzfile.close()
            weights = ndarrays[:len(ndarrays)//2]
            biases = ndarrays[len(ndarrays)//2:]
            print("\nModel loaded from " + load_path)

        # J3 regularization term
        J3_list = []
        for init_w, init_b in zip(weights[0:len(weights)//2], biases[0:len(weights)//2]):
            self.hidden_layers.append(DenseLayer(input_hidden, init_w, init_b, activation=activation))
            input_hidden = self.hidden_layers[-1].output
            J3_list.append(tf.reduce_mean(tf.square(self.hidden_layers[-1].w)))
            J3_list.append(tf.reduce_mean(tf.square(self.hidden_layers[-1].b)))
        #self-expressive layer
        if(self.trainC):
            input_hidden = tf.matmul(self.trainedC, input_hidden)
            self.H_M_2_post = input_hidden
        for init_w, init_b in zip(weights[len(weights)//2:], biases[len(weights)//2:]):
            self.hidden_layers.append(DenseLayer(input_hidden, init_w, init_b, activation=activation))
            input_hidden = self.hidden_layers[-1].output
            J3_list.append(tf.reduce_mean(tf.square(self.hidden_layers[-1].w)))
            J3_list.append(tf.reduce_mean(tf.square(self.hidden_layers[-1].b)))

        self.J3 = tf.add_n(J3_list)

        self.H_M = self.hidden_layers[-1].output
        # H(M/2) the output of the mid layer
        self.H_M_2 = self.hidden_layers[int((len(hidden_dims)-1)/2)].output

        # calculate loss J1
        # self.J1 = tf.nn.l2_loss(tf.subtract(self.X, self.H_M))

        self.J1 = tf.reduce_mean(tf.square(tf.subtract(self.X, self.H_M)))

        # calculate loss J2
        self.J2 = 0
        self.J4 = 0
        if self.givenC or self.trainC:
            if self.trainC:
                self.J2 = tf.reduce_mean(tf.square(tf.subtract(self.H_M_2, self.H_M_2_post)))
                self.J4 = tf.reduce_mean(tf.square(self.trainedC))
            else:
                self.J2 = tf.reduce_mean(tf.square(tf.subtract(tf.transpose(self.H_M_2), \
                                            tf.matmul(tf.transpose(self.H_M_2), self.C))))

        self.global_step = tf.Variable(1, dtype=tf.float32, trainable=False)

    def init_layer_weight(self, name, dims, epochs, activations, save_path=None, noise=None, loss='rmse', lr=0.001, batch_size=100, sda_optimizer='Adam', sda_decay='none', sda_printstep=100):
        weights, biases = [], []
        if name == 'sda-uniform':
            sda = supporting_files.sda.StackedDenoisingAutoencoder(dims, epochs, activations, noise, loss, lr, batch_size, sda_printstep, 'uniform', sda_optimizer, sda_decay, self.verbose)
            sda._fit(self.inputX)
            weights, biases = sda.weights, sda.biases
        if name == 'sda-normal':
            sda = supporting_files.sda.StackedDenoisingAutoencoder(dims, epochs, activations, noise, loss, lr, batch_size, sda_printstep, 'normal', sda_optimizer, sda_decay, self.verbose)
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

    def train(self, lambda1=0.01, lambda2=0.01, lambda3=0.0, learning_rate=0.1, optimizer='Adam', \
              decay='none', batch_size=100, epochs=100, print_step=100):
        if(self.verbose):
            print()
        cost = self.J1 + lambda1 * self.J2 + lambda2 * self.J3 + lambda3 * self.J4
        self.optimizer = optimize(cost, learning_rate, optimizer, decay, self.global_step)
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
                self.inputC -= np.diag(np.diag(self.inputC))

            self.losses.append(sess.run(cost, feed_dict={self.X: x_batch, self.C: c_batch}))

            if(self.verbose and i % print_step == 0):
                print('epoch {0}: global loss = {1}'.format(i, self.losses[-1]))

        # for i in range(1, epochs+1):
        #     x_batch, c_batch = get_batch_XC(self.inputX, self.inputC, batch_size)  
        #     self.losses.append(sess.run(self.cost, feed_dict={self.X: x_batch, self.C: c_batch}))      
        #     if i % print_step == 0:
        #         print('epoch {0}: global loss = {1}'.format(i, self.losses[-1]))

        self.result, self.reconstr, self.outC = sess.run([self.H_M_2, self.H_M, self.trainedC], feed_dict={self.X: x_batch, self.C: c_batch})
        return sess


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

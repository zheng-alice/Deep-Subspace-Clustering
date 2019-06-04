import supporting_files.sda
import tensorflow as tf
import numpy as np
import os
from copy import copy
from supporting_files.nncomponents import *
from supporting_files.helpers import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# print any active GPUs
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.close()

class DeepSubspaceClustering:

    def __init__(self, inputX, inputX_val=None, load_path=None, save_path=None, C=None, trainC=False, hidden_dims=[300,150,300], \
                 activation='tanh', weight_init='uniform', noise=None, sda_optimizer='Adam', sda_decay='none', \
                 weight_init_params={'epochs_max': 100,
                                     'sda_printstep': 100,
                                     'validation_step': 10,
                                     'stop_criteria': 3},
                 lr=0.001, batch_num=1, seed=None, verbose=True):
        # lr and batch_num are pretraining parameters
        # listed separately b/c need to be reached by optimization
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.verbose = verbose

        self.noise = noise
        n_sample, n_feat = inputX.shape

        # avoid overwriting parameters
        hidden_dims = copy(hidden_dims)
        weight_init_params = copy(weight_init_params)

        # M must be a even number
        assert len(hidden_dims) % 2 == 1

        # Add the end layer
        hidden_dims.append(n_feat)

        # self.depth = len(dims)

        # This is not the symbolic variable of tensorflow, this is real!
        self.inputX = inputX
        self.inputX_val = inputX_val

        # numpy ndarrays
        self.givenC = not C is None
        self.trainC = trainC
        if self.givenC:
            self.inputC = np.float32(C)
        else:
            self.inputC = np.random.normal(0.0, np.sqrt(1.0 / self.inputX.shape[0]), (self.inputX.shape[0], self.inputX.shape[0]))
            self.inputC -= np.diag(np.diag(self.inputC))

        # tensorflow Tensors
        self.C = tf.Variable(self.inputC, dtype=tf.float32, name='C')
        self.C_indxs = tf.placeholder(dtype=tf.int32, shape=[None], name='C_indxs')
        C_batch = tf.gather(self.C, self.C_indxs)

        self.hidden_layers = []
        # one for beginning, one for end -> solves batches with C-layer
        self.X = self._add_noise(tf.placeholder(dtype=tf.float32, shape=[None, n_feat], name='X'))
        self.X2 = self._add_noise(tf.placeholder(dtype=tf.float32, shape=[None, n_feat], name='X2'))

        input_hidden = self.X
        self.pre_loss = 1.0
        if (load_path is None):
            if('epochs_max' in weight_init_params):
                weight_init_params['epochs_max'] = [weight_init_params['epochs_max']]*len(hidden_dims)
            weights, biases, self.pre_loss = self.init_layer_weight(weight_init, hidden_dims,
                                                           lr=lr, activations=[activation]*len(hidden_dims),
                                                           sda_optimizer=sda_optimizer, sda_decay=sda_decay,
                                                           batch_num=batch_num, **weight_init_params)
            if(save_path is not None):
                save_path = save_path.format(self.pre_loss)
                np.savez(save_path, *weights, *biases)
                if(self.verbose):
                    print("\nModel saved to " + save_path + '.npz')
        else:
            load_path += '.npz'
            npzfile = np.load(load_path)
            ndarrays = [npzfile['arr_'+str(i)] for i in range(len(npzfile))]
            npzfile.close()
            weights = ndarrays[:len(ndarrays)//2]
            biases = ndarrays[len(ndarrays)//2:]
            if(self.verbose):
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
            input_hidden = tf.matmul(C_batch, input_hidden)
        elif(self.givenC):
            input_hidden = tf.gather(input_hidden, self.C_indxs)
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

        self.J1 = tf.reduce_mean(tf.square(tf.subtract(self.X2, self.H_M)))

        # calculate loss J2
        self.J2 = 0
        self.J4 = 0
        if self.trainC:
            self.J2 = tf.reduce_mean(tf.square(tf.subtract(tf.gather(self.H_M_2, self.C_indxs), self.H_M_2_post)))
            self.J4 = tf.reduce_mean(tf.square(C_batch))
        elif self.givenC:
            self.J2 = tf.reduce_mean(tf.square(tf.subtract(self.H_M_2_post, \
                                        tf.gather(tf.matmul(tf.transpose(self.inputC), self.H_M_2), self.C_indxs))))

        self.global_step = tf.Variable(1, dtype=tf.float32, trainable=False)

    def init_layer_weight(self, name, dims, epochs_max, activations, noise=None, loss='rmse', lr=0.001, batch_num=1, sda_optimizer='Adam', sda_decay='none', sda_printstep=100, validation_step=10, stop_criteria=3):
        weights, biases = [], []
        if name == 'sda-uniform':
            sda = supporting_files.sda.StackedDenoisingAutoencoder(dims, epochs_max, activations, noise, loss, lr, batch_num, sda_printstep, validation_step, stop_criteria, 'uniform', sda_optimizer, sda_decay, self.verbose)
            loss = sda._fit(self.inputX, self.inputX_val)
            weights, biases = sda.weights, sda.biases
        if name == 'sda-normal':
            sda = supporting_files.sda.StackedDenoisingAutoencoder(dims, epochs_max, activations, noise, loss, lr, batch_num, sda_printstep, validation_step, stop_criteria, 'normal', sda_optimizer, sda_decay, self.verbose)
            loss = sda._fit(self.inputX, self.inputX_val)
            weights, biases = sda.weights, sda.biases
        elif name == 'sda':
            sda = supporting_files.sda.StackedDenoisingAutoencoder(dims, epochs_max, activations, noise, loss, lr, batch_num, sda_printstep, validation_step, stop_criteria, 'default', sda_optimizer, sda_decay, self.verbose)
            loss = sda._fit(self.inputX, self.inputX_val)
            weights, biases = sda.weights, sda.biases
        elif name == 'uniform':
            n_in = self.inputX.shape[1]
            for d in dims:
                r = 4*np.sqrt(6.0/(n_in+d))
                weights.append(tf.Variable(tf.random_uniform([n_in, d], minval=-r, maxval=r)))
                biases.append(tf.Variable(tf.zeros([d,])))
                n_in = d

        return weights, biases, loss

    def train(self, lambda1=0.01, lambda2=0.01, lambda3=0.0, learning_rate=0.1, optimizer='Adam', \
              decay='none', batch_num=1, epochs=100, print_step=100, validation_step=-1, stop_criteria=-1):
        if(self.verbose):
            print()
        cost = self.J1 + lambda1 * self.J2 + lambda2 * self.J3 + lambda3 * self.J4
        self.optimizer = optimize(cost, learning_rate, optimizer, decay, self.global_step)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        loss_v_prev = float("inf")
        loss_v_best = float("inf")
        consec_increases = 0
        autoenc = [{'weights': layer.w, 'biases': layer.b} for layer in self.hidden_layers]
        encode = autoenc[:len(autoenc)//2]
        decode = autoenc[len(autoenc)//2:]
        enc_best = dec_best = c_best = None
        for i in range(epochs):
            batch = self._get_batches(len(self.inputX), batch_num=batch_num)
            for indx in batch:
                x_batch = self.inputX[indx]
                out=sess.run(self.optimizer, feed_dict={self.X: self.inputX if self.trainC or self.givenC else x_batch, self.X2: x_batch, self.C_indxs: indx})
            if print_step > 0:
                if i % print_step == 0:
                    loss_g = sess.run(cost, feed_dict={self.X: self.inputX, self.X2: self.inputX, self.C_indxs: range(len(self.inputX))})
                    if(self.verbose):
                        print('epoch {0}: global loss = {1}'.format(i, loss_g))
            if(self.inputX_val is not None and validation_step > 0):
                if(i % validation_step == 0):
                    # note: C-matrix values are entry-specific, thus can't run the validation set through the middle layer
                    if(self.trainC):
                        H_M_2, enc = sess.run([self.H_M_2, encode], feed_dict={self.X: self.inputX_val})
                        j1, j3, j4, dec, c = sess.run([self.J1, self.J3, self.J4, decode, self.C], feed_dict={self.H_M_2_post: H_M_2, self.X2: self.inputX_val, self.C_indxs: range(len(self.inputX))})
                    else:
                        j1, j3, enc, dec = sess.run([self.J1, self.J3, encode, decode], feed_dict={self.X: self.inputX_val, self.X2: self.inputX_val, self.C_indxs: range(len(self.inputX_val))})
                        j4 = 0
                        c = None
                    loss_v = j1 + lambda2*j3 + lambda3*j4
                    if(self.verbose):
                        print('epoch {0}: validation loss = {1}'.format(i, loss_v))
                    if(loss_v < loss_v_best):
                        loss_v_best = loss_v
                        enc_best = enc
                        dec_best = dec
                        c_best = c
                    if(stop_criteria > 0):
                        if(loss_v > loss_v_prev):
                            consec_increases += 1
                            if(consec_increases >= stop_criteria):
                                if(self.verbose):
                                    print("Training stopped after {0} epochs with loss = {1}".format(i, loss_v_best))
                                break
                        else:
                            consec_increases = 0
                        loss_v_prev = loss_v
        if(stop_criteria <= 0 or consec_increases < stop_criteria):
            if(self.verbose):
                print("Training exceeded max limit of {0} epochs".format(i+1))

        # for i in range(1, epochs+1):
        #     x_batch, c_batch = get_batch_XC(self.inputX, self.inputC, batch_num)  
        #     self.losses.append(sess.run(self.cost, feed_dict={self.X: x_batch, self.C: c_batch}))      
        #     if i % print_step == 0:
        #         print('epoch {0}: global loss = {1}'.format(i, self.losses[-1]))

        if enc_best is None:
            enc_best, dec_best = sess.run([encode, decode])
            if(c_best is None):
                c_best = sess.run(self.C)
        # replace current weights with best
        for i in range(len(encode)):
            sess.run([tf.assign(encode[i]['weights'], tf.convert_to_tensor(enc_best[i]['weights'], dtype=tf.float32)),
                      tf.assign(encode[i]['biases'], tf.convert_to_tensor(enc_best[i]['biases'], dtype=tf.float32)),
                      tf.assign(decode[i]['weights'], tf.convert_to_tensor(dec_best[i]['weights'], dtype=tf.float32)),
                      tf.assign(decode[i]['biases'], tf.convert_to_tensor(dec_best[i]['biases'], dtype=tf.float32))])
        if(self.trainC):
            sess.run(tf.assign(self.C, tf.convert_to_tensor(c_best, dtype=tf.float32)))
        self.result, self.reconstr, self.outC = sess.run([self.H_M_2, self.H_M, self.C], feed_dict={self.X: self.inputX, self.X2: self.inputX, self.C_indxs: range(len(self.inputX))})
        return sess

    def _get_batches(self, N, batch_num):
        indx = np.array(range(N))
        np.random.shuffle(indx)
        indx_split = np.array_split(indx, batch_num)
        return indx_split

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

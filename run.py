from full_model import *
from optimize import *
from skopt.space import Integer, Real


data_loaded = loadmat("./saved/rescaled/Coil20")
images_norm = data_loaded['X']
images_norm_val = data_loaded['X_val']
labels = data_loaded['Y'].reshape(-1)
labels_val = data_loaded['Y_val'].reshape(-1)


load_path = "./saved/models/coil20/256.64_10000.10.3_0.2098"
hidden_dims = [256,64,256]

ssc = {'model':run_ssc, 'n_rand':10, 'images_norm':images_norm, 'labels':labels, \
       'space': [Real(1, 100, "log-uniform", name='alpha')]}

ae = {'model':run_ae, 'n_rand':10, 'images_norm':images_norm, 'images_norm_val':images_norm_val, 'labels':labels, \
      'load_path':load_path, 'hidden_dims':hidden_dims, \
      'space': [Real(1.0E-05, 1.0E-02, "log-uniform", name='lr'), \
                Integer(1, 200, name='batch_num'),
                Real(1.0E-04, 1.0E-01, "log-uniform", name='lambda2'),
                Real(1.0E+00, 1.0E+03, "log-uniform", name='alpha2')]}

glob = {'model':run_model, 'n_rand':10, 'images_norm':images_norm, 'images_norm_val':images_norm_val, 'labels':labels, \
        'load_path':load_path, 'hidden_dims':hidden_dims, 'trainC':False, 'symmC':False, \
        'space': [Real(1.0E-05, 1.0E-02, "log-uniform", name='lr'), \
                  Integer(1, 200, name='batch_num'),
                  Real(1.0E+00, 1.0E+03, "log-uniform", name='alpha1'),
                  Real(1.0E-02, 1.0E+01, "log-uniform", name='lambda1'),
                  Real(1.0E-04, 1.0E-01, "log-uniform", name='lambda2'),
                  Real(1.0E+00, 1.0E+03, "log-uniform", name='alpha2')]}

symm = {'model':run_model, 'n_rand':10, 'images_norm':images_norm, 'images_norm_val':images_norm_val, 'labels':labels, \
        'load_path':load_path, 'hidden_dims':hidden_dims, 'trainC':False, 'symmC':True, \
        'space': [Real(1.0E-05, 1.0E-02, "log-uniform", name='lr'), \
                  Integer(1, 200, name='batch_num'),
                  Real(1.0E+00, 1.0E+03, "log-uniform", name='alpha1'),
                  Real(1.0E-02, 1.0E+01, "log-uniform", name='lambda1'),
                  Real(1.0E-04, 1.0E-01, "log-uniform", name='lambda2'),
                  Real(1.0E+00, 1.0E+03, "log-uniform", name='alpha2')]}

conj = {'model':run_model, 'n_rand':10, 'images_norm':images_norm, 'images_norm_val':images_norm_val, 'labels':labels, \
        'load_path':load_path, 'hidden_dims':hidden_dims, 'trainC':True, 'giveC':False, \
        'space': [Real(1.0E-05, 1.0E-02, "log-uniform", name='lr'), \
                  Integer(1, 200, name='batch_num'),
                  Real(1.0E-02, 1.0E+01, "log-uniform", name='lambda1'),
                  Real(1.0E-04, 1.0E-01, "log-uniform", name='lambda2'),
                  Real(1.0E+02, 1.0E+05, "log-uniform", name='lambda3')]}

init = {'model':run_model, 'n_rand':10, 'images_norm':images_norm, 'images_norm_val':images_norm_val, 'labels':labels, \
        'load_path':load_path, 'hidden_dims':hidden_dims, 'trainC':True, 'giveC':False, \
        'space': [Real(1.0E-05, 1.0E-02, "log-uniform", name='lr'), \
                  Integer(1, 200, name='batch_num'),
                  Real(1.0E+00, 1.0E+03, "log-uniform", name='alpha1'),
                  Real(1.0E-02, 1.0E+01, "log-uniform", name='lambda1'),
                  Real(1.0E-04, 1.0E-01, "log-uniform", name='lambda2'),
                  Real(1.0E+02, 1.0E+05, "log-uniform", name='lambda3')]}


print("1: SSC")
result = optimize(forest_minimize, ssc, 100, random_seed=0, verb_model=False, verb=True)
dump(result, "optims/train/1_SSC/256.64_10000.10.3.opt")

print("2: Autoencoder+SSC")
result = optimize(forest_minimize, ae, 100, random_seed=0, verb_model=False, verb=True)
dump(result, "optims/train/2_SSC+AE/256.64_10000.10.3.opt")

print("3: Global")
result = optimize(forest_minimize, glob, 100, random_seed=0, verb_model=False, verb=True)
dump(result, "optims/train/3_Global/256.64_10000.10.3.opt")

print("4: Global+Symmetric")
result = optimize(forest_minimize, symm, 100, random_seed=0, verb_model=False, verb=True)
dump(result, "optims/train/4_Glob+Symm/256.64_10000.10.3.opt")

print("5: Conjoined")
result = optimize(forest_minimize, conj, 100, random_seed=0, verb_model=False, verb=True)
dump(result, "optims/train/5_Conjoin/256.64_10000.10.3.opt")

print("6: Conjoined+Initialize")
result = optimize(forest_minimize, init, 100, random_seed=0, verb_model=False, verb=True)
dump(result, "optims/train/6_Conj+Init/256.64_10000.10.3.opt")
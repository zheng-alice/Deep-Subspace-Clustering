from optimize import *
from skopt.space import Integer, Real
import dsc
import glob
import os

def clean(path):
    for file in glob.glob(path)[5:]:
        file_remove = file.replace('\\', '/')
        os.remove(file_remove)
        print("Deleted " + file_remove)

def pretrain(**kwargs):
    d = dsc.DeepSubspaceClustering(**kwargs)
    if('save_path' in kwargs):
        clean(kwargs['save_path'].replace('{0:.4g}', '*.npz'))
    return d.pre_loss, 1.0

opt_params = {'model':pretrain, 'dataset':'Coil20', 'n_rand':10, \
              'hidden_dims':[256,64,256], 'sda_optimizer':'Adam', 'sda_decay':'none', 'weight_init':'sda-normal', \
              'weight_init_params':{'epochs_max': 10000, \
                                    'sda_printstep': -100, \
                                    'validation_step': 10, \
                                    'stop_criteria': 3}, \
              'space': [Real(1.0E-05, 1.0E-01, "log-uniform", name='lr'), \
                        Integer(1, 200, name='batch_num')],
              'save_path':"./saved/models/coil20/256.64_10000.10.3_{0:.4g}"}
data_loaded = loadmat("./saved/rescaled/" + opt_params.pop('dataset'))
opt_params['inputX'] = data_loaded['X']
opt_params['inputX_val'] = data_loaded['X_val']

result = optimize(forest_minimize, opt_params, 100, random_seed=0, verb_model=False, verb=True)
dump(result, "optims/pretrain/256.64_10000.10.3.opt")

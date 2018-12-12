import matplotlib.pyplot as plt
import numpy as np
import os
import time
import warnings
from copy import copy, deepcopy
from full_model import run_model
from scipy.io import savemat, loadmat
from sklearn.utils import check_random_state
from skopt import dump, load
from skopt import gp_minimize, dummy_minimize, forest_minimize, gbrt_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.optimizer import base_minimize
from skopt.plots import plot_convergence
from skopt.space import Integer, Real
from skopt.utils import use_named_args

show_plot = False
all_params = [
    {'model':run_model, 'dataset':'YaleB', 'n_rand':10, 'epochs_pretrain':201 , 'epochs':101, 'space':
         [Real(10**-2, 10**0, "log-uniform", name='lr_pretrain'),
          Real(10**-3, 10**-1, "log-uniform", name='lr'),
          Real(10**0, 10**2, "log-uniform", name='alpha1'),
          Integer(2, 32, name='maxIter1'),
          Real(10**0, 10**2, "log-uniform", name='alpha2'),
          Integer(2, 32, name='maxIter2')]},
    {'model':run_model, 'dataset':'YaleB', 'n_rand':10, 'epochs_pretrain':201 , 'epochs':101, 'space':
         [Real(10**-2, 10**0, "log-uniform", name='lr_pretrain'),
          Real(10**-3, 10**-1, "log-uniform", name='lr'),
          Real(10**0, 10**2, "log-uniform", name='alpha1'),
          Integer(10, 32, name='maxIter1'),
          Real(10**0, 10**2, "log-uniform", name='alpha2'),
          Integer(10, 32, name='maxIter2')]},
    {'model':run_model, 'dataset':'Coil20', 'n_rand':10, 'epochs_pretrain':201 , 'epochs':101, 'space':
         [Real(10**-4, 10**0, "log-uniform", name='lr_pretrain'),
          Real(10**-5, 10**-1, "log-uniform", name='lr'),
          Real(10**-1, 10**3, "log-uniform", name='alpha1'),
          Integer(10, 100, name='maxIter1'),
          Real(10**-1, 10**3, "log-uniform", name='alpha2'),
          Integer(10, 100, name='maxIter2')]},
    {'model':run_model, 'dataset':'Coil20', 'n_rand':10, 'epochs_pretrain':1001 , 'epochs':251, 'space':
         [Real(10**-4, 10**0, "log-uniform", name='lr_pretrain'),
          Real(10**-5, 10**-1, "log-uniform", name='lr'),
          Real(10**-1, 10**3, "log-uniform", name='alpha1'),
          Integer(10, 200, name='maxIter1'),
          Real(10**-1, 10**3, "log-uniform", name='alpha2'),
          Integer(10, 200, name='maxIter2')]}]

def get_params(scenario):
    fixed_params = copy(all_params[scenario])
    data_loaded = loadmat("./saved/processed/" + fixed_params.pop('dataset'))
    fixed_params['images_norm'] = data_loaded['X']
    fixed_params['labels'] = data_loaded['Y'].reshape(-1)
    return fixed_params

def opt_stats(res, start_time=None):
    print("------------------")
    for i in range(10, len(res.func_vals)+1, 10):
        print("{0:d}: {1:.4f}".format(i, min(res.func_vals[0:i])))
    print("Best score: {0:.4f}".format(res.fun))
    print("Best parameters: {0:}".format(res.x))
    if(start_time is not None):
        print("Total time elapsed: {0:.2f} sec\n".format(time.time()-start_time))

def opt_plot(res):
    global show_plot
    plot = plot_convergence(res, yscale='log')
    if(show_plot):
        plt.show(plot)
    return plot


def objective(hyper_params):
    global fixed_params_
    fixed_params_copy = copy(fixed_params_)
    fixed_params_copy.pop('n_rand')
    @use_named_args(fixed_params_copy.pop('space'))
    def internal(**hyper_params):
        global seed_, verb_model_
        try:
            fixed_params_copy.update(hyper_params)
            return 1-fixed_params_copy.pop('model')(verbose=verb_model_, **fixed_params_copy)[0]
        except Exception as ex:
            if(type(ex) == KeyError):
                raise ex
            print("Caught a " + type(ex).__name__ + ". Returning 1.0")
            return 1
    return internal(hyper_params)

def optimize(function, fixed_params, iterations, random_seed, verb_model=False, verb=True):
    start_time = time.time()
    
    # global b/c I couldn't find a better way to directly pass these to the objective function
    global fixed_params_, seed_, verb_model_
    fixed_params_ = fixed_params
    seed_ = random_seed
    verb_model_ = verb_model
    
    # kwargs b/c dummy_minimize can not take n_jobs
    optfunc_params = {'n_calls':iterations, 'random_state':random_seed, 'verbose':verb}
    if(function != dummy_minimize):
        optfunc_params['n_random_starts'] = fixed_params['n_rand']
        optfunc_params['n_jobs'] = -1
    result = function(objective, fixed_params['space'], **optfunc_params)
    if(verb):
        opt_plot(result)
        print()
    opt_stats(result, start_time)
    return result

def func_new(hyper_params):
    global func_, xs_, ys_
    if(len(xs_) > 0):
        y = ys_.pop(0)
        if(hyper_params != xs_.pop(0)):
            warnings.warn("Deviated from expected value, re-evaluating", RuntimeWarning)
        else:
            return y
    return func_(hyper_params)

def reload(result, fixed_params, addtl_iters, random_seed, verb_model=False, verb=True):
    start_time = time.time()
    
    # since objective relies on global variables, set them again
    global fixed_params_, seed_, verb_model_
    fixed_params_ = fixed_params
    seed_ = random_seed
    verb_model_ = verb_model
    
    # retrieve optimization call's arguments
    args = deepcopy(result.specs['args'])
    args['n_calls'] += addtl_iters
    args['verbose'] = verb
    
    # global b/c I couldn't find a better way to pass
    global func_, xs_, ys_
    func_ = args['func']
    xs_ = list(result.x_iters)
    ys_ = list(result.func_vals)
    args['func'] = func_new
    
    # recover initial random_state
    if(isinstance(args['random_state'], np.random.RandomState)):
        args['random_state'] = check_random_state(random_seed)
        # if gp_minimize
        if(isinstance(result.specs['args']['base_estimator'], GaussianProcessRegressor)):
            args['random_state'].randint(0, np.iinfo(np.int32).max)
    
    # run the optimization
    result_new = base_minimize(**args)
    
    # change the function back, to reload multiple times
    result_new.specs['args']['func'] = func_
    
    if(verb):
        opt_plot(result_new)
        print()
    opt_stats(result_new, start_time)
    return result_new


def optimize_multiple(scenario, iterations, seeds=range(5), functions={"gp":gp_minimize, "dummy":dummy_minimize, "forest":forest_minimize, "gbrt":gbrt_minimize}, verb_model=False, verb=False):
    fixed_params = get_params(scenario)
    if(not os.path.isdir("optims/scenario" + str(scenario))):
        os.mkdir("optims/scenario" + str(scenario))
    
    for seed in seeds:
        print("Seed: " + str(seed))
        for func_name in function_dict.keys():
            print(func_name + ':')
            result = optimize(function_dict[func_name], fixed_params, iterations, seed, verb_model=verb_model, verb=verb)
            dump(result, "optims/scenario" + str(scenario) + '/' + func_name + '_' + str(seed) + "_" + str(iterations) + ".opt")

def reload_multiple(scenario, init_iters, addtl_iters, seeds=range(5), func_names=["gp", "dummy", "forest", "gbrt"], verb_model=False, verb=False):
    fixed_params = get_params(scenario)
    for seed in seeds:
        print("Seed: " + str(seed))
        for func_name in function_names:
            print(func_name + ':')
            result_loaded = load("optims/scenario" + str(scenario) + '/' + func_name + '_' + str(seed) + "_" + str(init_iters) + ".opt")
            result = reload(result_loaded, fixed_params, addtl_iters, seed, verb_model=verb_model, verb=verb)
            dump(result, "optims/scenario" + str(scenario) + '/' + func_name + '_' + str(seed) + "_" + str(init_iters+addtl_iters) + ".opt")
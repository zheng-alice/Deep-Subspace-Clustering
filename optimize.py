import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import sys
import warnings
from copy import copy, deepcopy
from params import all_params
from pathlib import Path
from pympler.asizeof import asizeof
from scipy.io import savemat, loadmat
from sklearn.utils import check_random_state
from skopt import dump, load
from skopt import gp_minimize, dummy_minimize, forest_minimize, gbrt_minimize
from skopt.callbacks import VerboseCallback
from skopt.learning import GaussianProcessRegressor
from skopt.optimizer import base_minimize
from skopt.plots import plot_convergence
from skopt.utils import use_named_args


show_plot = False

def get_params(scenario):
    opt_params = copy(all_params[scenario])
    data_loaded = loadmat("./saved/processed/" + opt_params.pop('dataset'))
    opt_params['images_norm'] = data_loaded['X']
    opt_params['labels'] = data_loaded['Y'].reshape(-1)
    return opt_params

def reduce(result):
    """ Save space by getting rid of all models except the last. """
    del result.models[:-1]

def reduce_all(directory="./optims"):
    """ Reduce all .opt files in directory. """
    path = Path(directory)

    for resultpath in path.glob("**/*.opt"):
        try:
            result = load(resultpath)
        except:
            print("Could not load "+resultpath.parent.name+"/"+resultpath.name)
            continue
        
        oldsize = asizeof(result)
        reduce(result)
        newsize = asizeof(result)
        
        try:
            dump(result, resultpath)
        except:
            print("Could not save "+resultpath.parent.name+"/"+resultpath.name)
        print("Reduced "+resultpath.parent.name+"/"+resultpath.name+" by "+str((oldsize-newsize)*100/oldsize)+"%, from "+str(oldsize)+" to "+str(newsize))

def res_stats(result, start_time=None):
    print("---------------")
    for i in range(10, len(result.func_vals)+1, 10):
        print("{0:d}: {1:.4f}".format(i, min(result.func_vals[0:i])))
    print("Best score: {0:.4f}".format(result.fun))
    print("Best parameters: {0:}".format(result.x))
    if(start_time is not None):
        print("Total time elapsed: {0:.2f} sec\n".format(time.time()-start_time))

def res_plot(result):
    global show_plot
    if(show_plot):
        return plot_convergence(result, yscale='log')

def res_optimum(result):
    """ Extract the overall optimum from an optimization's surrogate model.

        PARAMETERS
        ----------
        result [OptimizeResult]:
            Result of the optimization, as returned by the optimization method.

        RETURNS
        -------
        x_opt [list of double]:
            The optimum hyperparameters, as predicted by the surrogate model.

        y_opt [double]:
            The corresponding predicted value.
    """
    X = result.space.rvs(result.specs['args']['n_points'], random_state=0)
    Y = result.models[-1].predict(result.space.transform(X))

    min_idx = np.argmin(Y)
    return X[min_idx], Y[min_idx]

def res_optimum_mult(results):
    """ Predict an overall optimum from a combination of
        several optimization's surrogate models.

        PARAMETERS
        ----------
        results [list of OptimizeResult]:
            Results of several optimizations, as returned by the optimization method.

        RETURNS
        -------
        x_opt [list of double]:
            The optimum hyperparameters, as predicted by the surrogate models.

        y_opt [double]:
            The geometric mean corresponding predicted value.
    """
    X = results[0].space.rvs(results[0].specs['args']['n_points'], random_state=0)
    Y_total = [1.0]*len(X)
    for result in results:
        Y_total *= result.models[-1].predict(result.space.transform(X))

    min_idx = np.argmin(Y_total)
    return X[min_idx], Y_total[min_idx]**(1/len(results))


def objective(hyper_params):
    global opt_params_
    all_params = copy(opt_params_)
    all_params.pop('n_rand')
    @use_named_args(all_params.pop('space'))
    def internal(**hyper_params):
        global seed_, verb_model_
        try:
            all_params.update(hyper_params)
            return all_params.pop('model')(seed=random.randrange(1000) if seed_ is None else seed_, verbose=verb_model_, **all_params)[0]
        except Exception as ex:
            if(type(ex) == KeyError):
                raise ex
            print("Caught a " + type(ex).__name__ + ". Returning 1.0")
            print(ex)
            return 1
    return internal(hyper_params)

def callback(result):
    reduce(result)
    sys.stdout.flush()

def optimize(function, opt_params, iterations, random_seed=None, verb_model=False, verb=True):
    """ Find optimum hyperparameters by repeatedly training the model from scratch.

        PARAMETERS
        ----------
        function [callable]:
            The function to use for optimization.
            One of:
                skopt.gp_minimize
                skopt.dummy_minimize
                skopt.forest_minimize
                skopt.gbrt_minimize

        opt_params [dict of str:callable]:
            Parameters that define the optimization.
            Includes both fixed parameters and hyperparameters (as space).
            Examples can be retrieved from get_params().

            model [callable]:
                The model to train.

            n_rand [int]:
                The number of iterations to randomly sample
                points for before fitting a model.
            
            space [list]:
                Hyperparameters to find and their limits.
                Same requirements as that of the optimization method
                All of these get passed to the model
                under their respective names.

            Any additional entries will be passed to the model.

        iterations [int]:
            Number of iterations to run the model for.

        random_seed [int or None, default=None]:
            Passed to the optimizer.
            Used to seed the sequience of model seeds.
        
        verb_model [bool, default=False]:
            Whether to pass verbose=True to the model.
        
        verb [bool, default=True]:
            Whether to print info at each iteration.

        RETURNS
        -------
        result [OptimizeResult]:
            Result of the optimization, as returned by the optimization method.
            Can be saved through dump(), and resumed later through reload().
    """
    start_time = time.time()
    
    # global b/c I couldn't find a better way to directly pass these to the objective function
    global opt_params_, seed_, verb_model_
    opt_params_ = opt_params
    seed_ = None
    random.seed(random_seed)
    verb_model_ = verb_model
    
    # kwargs b/c dummy_minimize can not take n_jobs
    optfunc_params = {'n_calls':iterations, 'random_state':random_seed, 'verbose':verb, 'callback':callback}
    if(function != dummy_minimize):
        optfunc_params['n_random_starts'] = opt_params['n_rand']
        optfunc_params['n_jobs'] = -1
    result = function(objective, opt_params['space'], **optfunc_params)
    del opt_params_, seed_, verb_model_
    if(verb):
        res_plot(result)
        print()
    res_stats(result, start_time)
    return result

def func_new(hyper_params):
    global func_, xs_, ys_
    if(len(xs_) > 0):
        y = ys_.pop(0)
        if(hyper_params != xs_.pop(0)):
            warnings.warn("Deviated from expected value, re-evaluating", RuntimeWarning)
        else:
            return y

    # manually add the verbose callback
    if('callback_' in globals()):
        global callback_, callback_verb_params_
        skip = callback_verb_params_.pop('skip')
        
        # very bad coding practices
        callback_verb = VerboseCallback.__new__(VerboseCallback, **callback_verb_params_)
        callback_verb.n_init = callback_verb_params_['n_init'];
        callback_verb.n_random = callback_verb_params_['n_random'];
        callback_verb.n_total = callback_verb_params_['n_total'];
        callback_verb.iter_no = skip + 1    # avoid printing that "Iteration No: 1" line
        callback_verb._start_time = time.time()
        callback_verb._print_info(start=True)

        callback_.append(callback_verb)
        del callback_, callback_verb_params_
    
    return func_(hyper_params)

def reload(result, opt_params, addtl_iters, random_seed=None, verb_model=False, verb=True, mute_reload=True):
    """ Resume a previous optimization.

        PARAMETERS
        ----------
        result [OptimizeResult]:
            The previously returned optimization.
            Tested functions:
                skopt.gp_minimize
                skopt.dummy_minimize
                skopt.forest_minimize
                skopt.gbrt_minimize

        opt_params [dict of str:callable]:
            Parameters that define the optimization.
            Includes both fixed parameters and hyperparameters (as space).
            Examples can be retrieved from get_params().

            model [callable]:
                The model to train.

            n_rand [int]:
                The number of iterations to randomly sample
                points for before fitting a model.
            
            space [list]:
                Hyperparameters to find and their limits.
                Same requirements as that of the optimization method.
                All of these get passed to the model
                under their respective names.

            Any additional entries will be passed to the model.

        addtl_iters [int]:
            Number of additional iterations to run the model for.

        random_seed [int or None, default=None]:
            Passed to the optimizer.
            Used to seed the sequience of model seeds.
        
        verb_model [bool, default=False]:
            Whether to pass verbose=True to the model.
        
        verb [bool, default=True]:
            Whether to print info at each iteration.

        mute_reload [bool, default=True]:
            Passes False to verb upon initial reloaded iterations.
            Reverts to specified value when recovery is complete.

        RETURNS
        -------
        result_new [OptimizeResult]:
            The updated result of the optimization, as returned by the optimization method.
            Can be saved through dump(), and resumed once again.
    """
    start_time = time.time()
    
    # since objective relies on global variables, set them again
    global opt_params_, seed_, verb_model_
    opt_params_ = opt_params
    seed_ = None
    random.seed(random_seed)
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

    # mute, if necessary
    if(mute_reload and verb):
        args['verbose'] = False
        global callback_, callback_verb_params_
        callback_ = [args['callback']]
        x0 = args['x0']
        callback_verb_params_ = {
            'n_init': 0 if x0 is None else 1 if not isinstance(x0[0], (list, tuple)) else len(x0) \
                if not [args['y0']] else 0,
            'n_random': args['n_random_starts'],
            'n_total': args['n_calls'],
            'skip': len(xs_)}
        args['callback'] = callback_
    
    # recover initial random_state
    if(isinstance(args['random_state'], np.random.RandomState)):
        args['random_state'] = check_random_state(random_seed)
        # if gp_minimize
        if(isinstance(result.specs['args']['base_estimator'], GaussianProcessRegressor)):
            args['random_state'].randint(0, np.iinfo(np.int32).max)
    
    # run the optimization
    result_new = base_minimize(**args)
    del opt_params_, seed_, verb_model_

    # remove the manually-added verbose callback
    if(mute_reload and verb):
        result_new.specs['args']['callback'] = args['callback'][0]

    # change the function back, to reload multiple times
    result_new.specs['args']['func'] = func_
    del func_, xs_, ys_
    
    if(verb):
        res_plot(result_new)
        print()
    res_stats(result_new, start_time)
    return result_new


def optimize_multiple(scenario, iterations, seeds=range(5), functions={"gp":gp_minimize, "dummy":dummy_minimize, "forest":forest_minimize, "gbrt":gbrt_minimize}, verb_model=False, verb=False):
    """ Call optimize() across several functions and seeds.
        Automatically dump results.

        PARAMETERS
        ----------
        scenario [int]:
            id of the scenario.
            Used to get_params().
            Determines the directory to save in.

        iterations [int]:
            Number of iterations to run the model for.
            Used in filenames upon saving.

        seeds [list of int]:
            Values to be passed as seeds.
            Used in filenames upon saving.

        functions [dict of str:callable]:
            Optimization functions to call and their respective names.
            Names are used in the filenames upon saving.
    """
    opt_params = get_params(scenario)
    if(not os.path.isdir("optims/scenario" + str(scenario))):
        os.mkdir("optims/scenario" + str(scenario))
    
    for seed in seeds:
        print("Seed: " + str(seed))
        for func_name in functions.keys():
            print(func_name + ':')
            result = optimize(functions[func_name], opt_params, iterations, seed, verb_model=verb_model, verb=verb)
            dump(result, "optims/scenario" + str(scenario) + '/' + func_name + '_' + str(seed) + "_" + str(iterations) + ".opt")

def reload_multiple(scenario, init_iters, addtl_iters, seeds=range(5), func_names=["gp", "dummy", "forest", "gbrt"], verb_model=False, verb=False, mute_reload=True):
    """ Call reload() across several functions and seeds.
        Automatically dump results.

        PARAMETERS
        ----------
        scenario [int]:
            id of the scenario.
            Used to get_params().
            Determines the directory to save in.

        init_iters [int]:
            Iteration count of optimizations to load.

        addtl_iters [int]:
            Number of additional iterations to run the model for.
            New total is used in filenames upon saving.

        seeds [list of int]:
            Values to be passed as seeds.
            Used in filenames upon saving.

        func_names [list of str]:
            Names of optimization functions to reload.
            Used in filenames upon saving.
    """
    opt_params = get_params(scenario)
    for seed in seeds:
        print("Seed: " + str(seed))
        for func_name in func_names:
            print(func_name + ':')
            result_loaded = load("optims/scenario" + str(scenario) + '/' + func_name + '_' + str(seed) + "_" + str(init_iters) + ".opt")
            result = reload(result_loaded, opt_params, addtl_iters, seed, verb_model=verb_model, verb=verb, mute_reload=mute_reload)
            dump(result, "optims/scenario" + str(scenario) + '/' + func_name + '_' + str(seed) + "_" + str(init_iters+addtl_iters) + ".opt")


def reeval(scenario, x, seeds=range(5), verb_model=False):
    """ Reload an already completed optimization and re-evaluate on its optimum.

        PARAMETERS
        ----------
        scenario [int]:
            id of the loaded optimization.
            Used in filename of loaded optimization.

        x [list of int/double]
            Location at which to re-evaluate.

        seeds [list of int]:
            Seed values used for re-evaluating.

        verb_model [bool, default=False]:
            Whether to pass verbose=True to the model.
    """
    opt_params = get_params(scenario)

    global opt_params_, seed_, verb_model_
    opt_params_ = opt_params
    verb_model_ = verb_model
    total = 1.0
    for seed in seeds:
        print("seed:", seed)
        seed_ = seed
        value = objective(x)
        print("value:", value)
        total *= value

    average = total ** (1/len(seeds))
    print("AVERAGE:", average)
    del opt_params_, seed_, verb_model_
    
    return average

def reeval_optimum(scenario, iterations, best_seed=0, best_func="forest", seeds=range(5), verb_model=False):
    """ Reload an already completed optimization and re-evaluate on its optimum.

        PARAMETERS
        ----------
        scenario [int]:
            id of the loaded optimization.
            Used in filename of loaded optimization.

        iterations [int]:
            Number of iterations of the loaded optimization.
            Used in filename of loaded optimization.

        best_seed [int, default=0]:
            Seed of the loaded optimization.
            Used in filename of loaded optimization.

        best_func [str, default="forest"]:
            Optimization function of the loaded optimization.
            Used in filename of loaded optimization.

        seeds [list of int]:
            Seed values used for re-evaluating.

        verb_model [bool, default=False]:
            Whether to pass verbose=True to the model.
    """
    optimum = res_optimum(load("./optims/scenario"+str(scenario)+"/"+best_func+"_"+str(best_seed)+"_"+str(iterations)+".opt"))
    print("optimum:")
    print(optimum)

    return reeval(scenario, optimum[0], seeds, verb_model)

def reeval_minimum(scenario, iterations, best_seed=0, best_func="forest", seeds=range(5), verb_model=False):
    """ Reload an already completed optimization and re-evaluate on its optimum.

        PARAMETERS
        ----------
        scenario [int]:
            id of the loaded optimization.
            Used in filename of loaded optimization.

        iterations [int]:
            Number of iterations of the loaded optimization.
            Used in filename of loaded optimization.

        best_seed [int, default=0]:
            Seed of the loaded optimization.
            Used in filename of loaded optimization.

        best_func [str, default="forest"]:
            Optimization function of the loaded optimization.
            Used in filename of loaded optimization.

        seeds [list of int]:
            Seed values used for re-evaluating.

        verb_model [bool, default=False]:
            Whether to pass verbose=True to the model.
    """
    minimum = load("./optims/scenario"+str(scenario)+"/"+best_func+"_"+str(best_seed)+"_"+str(iterations)+".opt").x
    print("minimum:")
    print(minimum)

    return reeval(scenario, minimum, seeds, verb_model)
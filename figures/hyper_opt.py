import matlab.engine
from optimize import *
from skopt.acquisition import _gaussian_acquisition


def extract_visual(result, N=10):
	# K - number of axes
	# N - number of eval points per axis
	K = result.space.n_dims
	if(isinstance(N, int)):
	    N = (N,) * K

	# ground-truth
	truth = {'names': [],
	         'priors': [],
	         'x': np.array(result.x_iters),
	         'y': result.func_vals,
	         'best': np.where(result.func_vals == result.fun)[0][0] + 1}    # matlab indexing

	# axes names and values
	axes = {}
	for idx in range(K):
	    dim = result.space.dimensions[idx]
	    
	    truth['names'].append(dim.name)
	    if(dim.prior == 'log-uniform'):
	        truth['priors'].append('log')
	        func = np.geomspace
	    else:
	        truth['priors'].append('uniform')
	        func = np.linspace
	    axes[dim.name] = func(dim.low, dim.high, num=N[idx])

	# surrogate model
	surrogate = {'mean': np.zeros(N), 'std': np.zeros(N)}

	# variable number of for-loops -> recursion
	def iterate(i, idx, xs, idxs):
	    if(i >= K):
	        # base case - evaluate
	        x = []
	        for j in range(K):
	            x.append(axes[truth['names'][j]][idx[j]])
	        idx2 = tuple(idx)
	        
	        # true value and index
	        xs.append(x)
	        idxs.append(idx2)
	        return
	    for val in range(N[i]):
	        # recursion - proceed to next dimension
	        idx.append(val)
	        iterate(i+1, idx, xs, idxs)
	        idx.pop()

	xs = []
	idxs = []
	iterate(0, [], xs, idxs)
	value = result.models[-1].predict(result.space.transform(xs), return_std=True)
	for i in range(len(idxs)):
	    surrogate['mean'][idxs[i]] = value[0][i]
	    surrogate['std'][idxs[i]] = value[1][i]

	return {'surrogate':surrogate, 'axxes':axes, 'truth':truth}


if __name__ == '__main__':
	result = load("./optims/scenario3/forest_0_100.opt")
	mdict = extract_visual(result, 10)
	savemat('./figures/hyper_opt.mat', mdict=mdict, oned_as='column')
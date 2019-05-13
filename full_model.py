import dsc
import numpy as np
import os
import supporting_files.sda as sda
import time
import sys
from img2matrix import single_img2dsift
from scipy.io import savemat, loadmat
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from supporting_files.ji_zhang import err_rate
from supporting_files.helpers import optimize

def start_matlab():
    print("\nStarting MATLAB engine...")
    print("-------------------------")
    start_time = time.time()
    
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd("./SSC_ADMM_v1.1")

    print("Elapsed: {0:.2f} sec".format(time.time()-start_time))
    
    return eng

def start_octave():
    print("\nStarting Octave...")
    print("------------------")
    start_time = time.time()

    from oct2py import octave
    octave.cd("./SSC_ADMM_v1.1")
    octave.eval("svdDriversCompare")

    print("Elapsed: {0:.2f} sec".format(time.time()-start_time))

    return octave

eng_name = os.getenv('ENGINE_CHOICE', 'MATLAB')
if(eng_name == 'MATLAB'):
    eng = start_matlab()
elif(eng_name == 'OCTAVE'):
    eng = start_octave()
else:
    raise RuntimeError("Unknown ENGINE_CHOICE: " + eng_name)

#from load import load_YaleB
#images, labels = load_YaleB()
#from scipy.io import savemat
#savemat('./saved/raw/yaleB.mat', mdict={'X':images, 'Y':labels})
#images_norm = flatten(rescale(images))
#images_norm_val, images_norm = split(images_norm, 0.2) # only do this once
#labels_val, labels = split(labels, 0.2)                # if loading pre-trained models, load the partitioned 'rescaled' data
#savemat('./saved/rescaled/yaleB.mat', mdict={'X':images_norm, 'Y':labels, 'X_val':images_norm_val, 'Y_val':labels_val})

#from scipy.io import loadmat
#data_loaded = loadmat("./saved/rescaled/yaleB.mat")
#images_norm = data_loaded['X']
#labels = data_loaded['Y'].reshape(-1)
#images_norm_val = data_loaded['X_val']
#labels_val = data_loaded['Y_val'].reshape(-1)
#run_model(images_norm, labels)

def flatten(images):
    images_flat = images.reshape(images.shape[0], -1)
    return images_flat

def rescale(images):
    maxdim = max(images.shape[1:])
    if(maxdim > 32):
        print("\nDownsampling...")
        print("----------------")
        start_time = time.time()

        # Downsample
        # for now, only supports single-channel images
        factor = maxdim / 32
        newsize = (int(images.shape[1]/factor), int(images.shape[2]/factor))
        images = np.moveaxis(np.float32(resize(np.moveaxis(images, 0, -1), output_shape=newsize, order=1, mode='reflect', anti_aliasing=True)), -1, 0)

        print("Elapsed: {0:.2f} sec".format(time.time()-start_time))


    print("\nNormalizing data...")
    print("-------------------")
    start_time = time.time()

    # Normalize scaled output
    mmin = np.min(images)
    mmax = np.max(images)
    images_norm = (np.multiply(images, 2, dtype='float32') - mmax - mmin) / (mmax - mmin)

    print("Elapsed: {0:.2f} sec".format(time.time()-start_time))

    return images_norm

def preprocess(images):
    print("\nRunning DSIFT...")
    print("----------------")
    start_time = time.time()

    images_dsift = [single_img2dsift(image) for image in images]

    print("Elapsed: {0:.2f} sec".format(time.time()-start_time))


    print("\nPerforming PCA...")
    print("-----------------")
    start_time = time.time()
    
    # Perform PCA
    pca = PCA(n_components=300, whiten=False, svd_solver='arpack', random_state=0)
    images_pca = pca.fit_transform(images_dsift)

    print("Elapsed: {0:.2f} sec".format(time.time()-start_time))


    print("\nNormalizing data...")
    print("-------------------")
    start_time = time.time()

    # Normalize PCA output
    mmin = np.min(images_pca)
    mmax = np.max(images_pca)
    images_norm = (2*images_pca - mmax - mmin) / (mmax - mmin)

    print("Elapsed: {0:.2f} sec".format(time.time()-start_time))

    return images_norm

def suppress_mlab(mlab_kwargs):
    # Suppress matlab's printed output
    if(type(eng).__name__ == 'MatlabEngine'):
        if(sys.version_info[0] < 3):
            from StringIO import StringIO
        else:
            from io import StringIO
        mlab_kwargs['stdout'] = StringIO()
    elif(type(eng).__name__ == 'Oct2Py'):
        def void(x):
            pass
        mlab_kwargs['stream_handler'] = void

def evaluate(labels, labels_pred):
    # Compare predicted labels to known ground-truths
    # Returns error rate, 1-NMI, 1-ARI
    return err_rate(labels, labels_pred), 1-nmi(labels, labels_pred, average_method="geometric"), 1-ari(labels, labels_pred)

def run_model(images_norm,
              labels,
              seed = None,
              epochs_pretrain = 101,
              epochs = 101,
              lr_pretrain = 0.08,
              lr = 0.006,
              alpha1 = 20,
              maxIter1 = 6,
              lambda1 = 0.0001,
              lambda2 = 0.001,
              lambda3 = 0.0,
              alpha2 = 20,
              maxIter2 = 16,
              trainC = False,
              giveC = False,
              verbose = True):
    # Hard-cast to avoid errors
    maxIter1 = int(maxIter1)
    maxIter2 = int(maxIter2)
    

    # Calculate C matrix
    # Matlab SSC #1
    mlab_kwargs = {}
    if(not verbose):
        suppress_mlab(mlab_kwargs)
    if(seed is None):
        seed2 = -1
    else:
        seed2 = seed
    k = len(np.unique(labels))
    C=None
    if((not trainC) or giveC):
        if(verbose):
            start_time = time.time()
            print("\nFinding affinity matrix (iter: {0:d})...".format(maxIter1))
            print("-------------------------------------")

        savemat('./temp.mat', mdict={'X': images_norm})
        eng.SSC_modified(k, 0, False, float(alpha1), False, 1, 1e-20, maxIter1, False, seed2, False, **mlab_kwargs)
        C = loadmat("./temp.mat")['C']

        if(verbose):
            print("Elapsed: {0:.2f} sec".format(time.time()-start_time))


    # Train Autoencoder
    if(verbose):
        start_time = time.time()
        print("\nTraining Autoencoder...")
        print("-----------------------")

    d = dsc.DeepSubspaceClustering(images_norm, C=C, trainC=trainC, hidden_dims=[200, 150, 200], lambda1=lambda1,
                                   lambda2=lambda2, lambda3=lambda3, learning_rate=lr, weight_init='sda-normal',
                                   weight_init_params={'epochs_max':epochs_pretrain, 'lr':lr_pretrain,
                                   'batch_num':images_norm.shape[0], 'sda_printstep':100},
                                   optimizer='Adam', decay='sqrt', sda_optimizer='Adam', sda_decay='sqrt',
                                   seed=seed, verbose=verbose)
    d.train(batch_num=images_norm.shape[0], epochs=epochs, print_step=25)
    images_HM2 = d.result

    if(verbose):
        print("Elapsed: {0:.2f} sec".format(time.time()-start_time))


    # Cluster
    # Matlab SSC #2
        print("\nClustering with SSC (iter: {0:d})...".format(maxIter2))
        print("---------------------------------")
        start_time = time.time()
    
    if(trainC):
        savemat('./temp.mat', mdict={'C': np.float64(d.outC)})
    else:
        savemat('./temp.mat', mdict={'X': images_HM2})
    grps = eng.SSC_modified(k, 0, False, float(alpha2), False, 1, 1e-20, maxIter2, True, seed2, trainC, **mlab_kwargs)
    labels_pred = np.asarray(grps, dtype=np.int32).flatten()

    if(verbose):
        print("Elapsed: {0:.2f} sec\n".format(time.time()-start_time))

    # Evaluate
    return evaluate(labels, labels_pred)

def run_ae(images_norm,
           labels,
           seed = None,
           epochs_pretrain = 101,
           epochs = 101,
           lr_pretrain = 0.08,
           lr = 0.006,
           lambda2 = 0.001,
           alpha2 = 20,
           maxIter2 = 16,
           verbose = True):
    # Hard-cast to avoid errors
    maxIter2 = int(maxIter2)


    # Train Autoencoder
    mlab_kwargs = {}
    if(verbose):
        start_time = time.time()
        print("\nTraining Autoencoder...")
        print("-----------------------")
    else:
        suppress_mlab(mlab_kwargs)

    d = dsc.DeepSubspaceClustering(images_norm, C=None, trainC=False, hidden_dims=[200, 150, 200], lambda2=lambda2, learning_rate=lr,
                                   weight_init='sda-normal', weight_init_params={'epochs_max':epochs_pretrain, 'lr':lr_pretrain,
                                   'batch_num':images_norm.shape[0], 'sda_printstep':100}, optimizer='Adam', decay='sqrt',
                                   sda_optimizer='Adam', sda_decay='sqrt', seed=seed, verbose=verbose)
    d.train(batch_num=images_norm.shape[0], epochs=epochs, print_step=25)
    images_HM2 = d.result

    if(verbose):
        print("Elapsed: {0:.2f} sec".format(time.time()-start_time))


    # Cluster
    # Matlab SSC #2
        print("\nClustering with SSC (iter: {0:d})...".format(maxIter2))
        print("---------------------------------")
        start_time = time.time()
    
    savemat('./temp.mat', mdict={'X': images_HM2})
    if(seed is None):
        seed2 = -1
    else:
        seed2 = seed
    k = len(np.unique(labels))
    grps = eng.SSC_modified(k, 0, False, float(alpha2), False, 1, 1e-20, maxIter2, True, seed2, False, **mlab_kwargs)
    labels_pred = np.asarray(grps, dtype=np.int32).flatten()

    if(verbose):
        print("Elapsed: {0:.2f} sec\n".format(time.time()-start_time))

    # Evaluate
    return evaluate(labels, labels_pred)

def run_ssc(images_norm,
            labels,
            seed = None,
            alpha = 20,
            maxIter = 6,
            verbose = True):
    # Hard-cast to avoid errors
    maxIter = int(maxIter)
    

    # Cluster
    # Matlab SSC
    mlab_kwargs = {}
    if(verbose):
        start_time = time.time()
        print("\nClustering with SSC (iter: {0:d})...".format(maxIter))
        print("---------------------------------")
    else:
        suppress_mlab(mlab_kwargs)
    
    savemat('./temp.mat', mdict={'X': images_norm})
    if(seed is None):
        seed2 = -1
    else:
        seed2 = seed
    k = len(np.unique(labels))
    grps = eng.SSC_modified(k, 0, False, float(alpha), False, 1, 1e-20, maxIter, True, seed2, False, **mlab_kwargs)
    labels_pred = np.asarray(grps, dtype=np.int32).flatten()

    if(verbose):
        print("Elapsed: {0:.2f} sec\n".format(time.time()-start_time))

    return evaluate(labels, labels_pred)

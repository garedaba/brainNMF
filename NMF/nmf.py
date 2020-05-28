import scipy.sparse as sp
import numpy as np

import warnings

from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_random_state, check_array, check_symmetric
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

# helper functions
def check_non_negative(X, whom):
    X = X.data if sp.issparse(X) else X
    if (X < 0).any():
        raise ValueError("Negative values in data passed to %s" % whom)

def normalise_features(X):
    "normalise matrix rows to unit norm"
    X_N = (X.T / np.linalg.norm(X, axis=1)).T

    return X_N

def calculate_MSE(A,B,mask):
    if mask is None:
        print('making a mask')
        mask = np.ones((np.shape(A)))
    # calculate difference between matrices
    diff = mask*(A - B)
    # square
    diff = np.square(diff)

    # get average for all (non zero) elements in mask
    diff = diff.sum() / np.count_nonzero(mask)
    # square root for RMSE
    mse = np.sqrt(diff)

    return mse

def get_KNN(X, k):
    """Identify nearest neighbours

        Parameters
        ----------
        D : array, [n_samples, n_features]
         input data
        k : int
         number of nearest neighbours
        Returns
        -------
        knn_graph : array, [n_samples, n_samples]
        Connectivity matrix with binary edges connecting nearest neighbours
        """

    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    # sparse neighbourhood graph
    W = knn.fit(X).kneighbors_graph(mode='connectivity')
    # into square matrix
    W = W.toarray()
    # enforce symmetry (not true kNN)
    W = np.fmax(W,W.T)

    #knn_graph = W*(1-D) # similarity not distance

    return W

def disarrange(a, axis=-1):
    """
        Shuffle `a` in-place along the given axis.

        Apply numpy.random.shuffle to the given axis of `a`.
        Each one-dimensional slice is shuffled independently.
        """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return


# NMF functions
def nmf_init(X, n_components, variant=None, eps=1e-6, random_state=None):
    """NNDSVD algorithm for NMF initialization.
        Computes a good initial guess for the non-negative
        rank k matrix approximation for X: X = WH

        https://github.com/GHFC/StratiPy/blob/master/stratipy/clustering.py

        Parameters
        ----------
        X : array, [n_samples, n_features]
         The data matrix to be decomposed.
        n_components : array, [n_components]
         The number of components desired in the approximation.
        variant : None | 'a' | 'ar'
         The variant of the NNDSVD algorithm.
        Accepts None, 'a', 'ar'
         None: leaves the zero entries as zero
         'a': Fills the zero entries with the average of X
         'ar': Fills the zero entries with standard normal random variates.
         Default: None
        eps: float
         Truncate all values less then this in output to zero.
        random_state : numpy.RandomState | int, optional
         The generator used to fill in the zeros, when using variant='ar'
        Default: numpy.random

        Returns
        -------
        (W, H) :
        Initial guesses for solving X ~= WH such that
        the number of columns in W is n_components.
        Remarks
        -------
        This implements the algorithm described in
        C. Boutsidis, E. Gallopoulos: SVD based
        initialization: A head start for nonnegative
        matrix factorization - Pattern Recognition, 2008
        http://tinyurl.com/nndsvd
        """
    check_non_negative(X, "NMF initialization")

    if variant not in (None, 'a', 'ar'):
        raise ValueError("Invalid variant name")

    U, S, V = randomized_svd(X, n_components)
    # dtype modification
    W, H = np.zeros(U.shape, dtype=np.float32), np.zeros(V.shape, dtype=np.float32)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = np.linalg.norm(x_p), np.linalg.norm(y_p)
        x_n_nrm, y_n_nrm = np.linalg.norm(x_n), np.linalg.norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if variant == "a":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif variant == "ar":
        random_state = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * random_state.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * random_state.randn(len(H[H == 0])) / 100)

    return W, H.T


def PNMF(X, n_components, mask=None, init='svd', maxIter=10000, verbose=True, tol=1e-6):
    """Non-negative matrix factorisation.

        Based on Matlab code from Zhirong Yang
        https://sites.google.com/site/zhirongyangcs/pnmf

        Parameters
        ----------
        X : array, [n_features, n_samples]
        The data matrix to be decomposed.
        n_components : int
        The number of components desired in the approximation.
        mask : array, [n_features, n_samples] or None
        Weight mask for NMF, if None then all elements are included
        init : 'svd' | 'random'
        NMF initialisation, if SVD then NND-SVD will be used, otherwise random matrices will be used
        max_iter : int
        Maximum number of iterations to run

        Returns
        -------
        results : dict
        W : final solution, where H = W'*X and X = W*H
        obj : reconstruction error over iterations

        """

    nFea,nSmp=np.shape(X)

    # apply weight matrix
    if mask is None:
        mask = np.ones((nFea, nSmp))
    X_ = mask * X

    if init=='svd':
        if verbose is True:
            print("*****************************************************************")
            print("initialise with non-negative SVD")
            print("*****************************************************************")
            print("")
        # initialise with NNDSVD
        W, H = nmf_init(X_, n_components, variant=None)

    elif init=='random':
        if verbose is True:
            print("*****************************************************************")
            print("performing initialisation with random start")
            print("*****************************************************************")
        # initialise with random matrices (and repeat)
        W = abs(np.random.random((nFea,n_components)))
    else:
        print("unknown initialisation method - try 'svd' or 'random'")
        return None

    if verbose is True:
        print("*****************************************************************")
        print("minimising Euclidean distance")
        print("*****************************************************************")
        print("")

    list_reconstruction_err_ = []

    nIter=0
    while nIter < maxIter:
        W_old = W.copy()

        # update W
        # Euclidean
        # W = W * (XX'W) / (WW'XX'W)  # O PNMF variant)
        W = W * ((X_.dot(np.dot(X_.T,W))) / (np.dot((mask * np.dot(W, np.dot(W.T, X_))), np.dot(X_.T,W)) + np.finfo(float).eps))

        # get rid of tiny values
        W[W<1e-10] = 0

        # W unit norm
        W = W / np.linalg.norm(W, 2)

        # has W changed much?
        diffW = np.linalg.norm(W_old-W) / np.linalg.norm(W)

        # iterate
        nIter = nIter + 1
        if nIter%100==0:
            nIter100 = nIter/100
            # reconstruction error
            err = np.linalg.norm(mask*(X_-W.dot(np.dot(W.T, X_))))
            # add to list
            list_reconstruction_err_.append(err)
            if verbose is True and nIter100 > 1:
                print(f'iteration {nIter} :: current recon. error {list_reconstruction_err_[-1]:.3f} :: W change = {diffW:.6f}')

        # check convergence
        if diffW <tol:
            print('converged after {:} iterations'.format(nIter))
            nIter=maxIter

    norms = np.fmax(1e-15, np.sqrt(np.sum(W**2,0)))

    results =  dict({'W' : W,
                    'norms' : norms,
                    'obj' : list_reconstruction_err_})

    return results

def ARDPNMF(X, n_components, mask=None, init='svd', maxIter=10000, verbose=True, tol=1e-6):
    """Non-negative matrix factorisation with Automatic Relevance Determination.

        Based on Matlab code from Zhirong Yang
        https://sites.google.com/site/zhirongyangcs/ardpnmf

        Parameters
        ----------
        X : array, [n_features, n_samples]
        The data matrix to be decomposed.
        n_components : int
        The initial number of components, relatively large.
        mask : array, [n_features, n_samples] or None
        Weight mask for NMF, if None then all elements are included
        init : 'svd' | 'random'
        NMF initialisation, if SVD then NND-SVD will be used, otherwise random matrices will be used
        max_iter : int
        Maximum number of iterations to run

        Returns
        -------
        results : dict
        W : final solution, where H = W'*X and X = W*H
        obj : reconstruction error over iterations
        sigma : column variances over iterations
        """

    nFea,nSmp=np.shape(X)

    # apply weight matrix if specified
    if mask is None:
        mask = np.ones((nFea, nSmp))
    X_ = mask * X

    if init=='svd':
        if verbose is True:
            print("*****************************************************************")
            print("initialise with non-negative SVD")
            print("*****************************************************************")
            print("")
        # initialise with NNDSVD
        W, H = nmf_init(X_, n_components, variant=None)

    elif init=='random':
        if verbose is True:
            print("*****************************************************************")
            print("performing initialisation with random start")
            print("*****************************************************************")
        # initialise with random matrices (and repeat)
        W = np.random.rand(nFea,n_components)

    else:
        print("unknown initialisation method - try 'svd' or 'random'")
        return None

    if verbose is True:
        print("*****************************************************************")
        print("minimising Euclidean distance")
        print("*****************************************************************")
        print("")

    list_reconstruction_err_ = []
    list_sigma_ = []

    nIter=0
    while nIter < maxIter:
        W_old = W.copy()
        sigma = np.diag(np.dot(W.T, W))

        # update W
        # Euclidean - check this with masking
        xxw = np.dot(X_, X_.T.dot(W))
        num = np.dot(xxw, np.diag(sigma))
        denom = np.dot(np.dot(W, W.T.dot(xxw)) + np.dot(xxw, W.T.dot(W)), np.diag(sigma))

        W = W * (num / (denom + W + np.finfo(float).eps))

        # get rid of tiny values
        W[W<1e-10] = 0

        # W columns to unit norm
        W = W / np.linalg.norm(W,2)

        # has W changed much?
        diffW = np.linalg.norm(W_old-W) / np.linalg.norm(W_old) # w_old?

        # iterate
        nIter = nIter + 1
        if nIter%100==0:
            nIter100 = nIter/100
            # reconstruction error
            err = np.linalg.norm(mask*(X_-W.dot(np.dot(W.T, X_))))
            # add to list
            list_reconstruction_err_.append(err)
            if verbose is True and nIter100 > 1:
                print(f'iteration {nIter} :: current recon. error {list_reconstruction_err_[-1]:.3f} :: W change = {diffW:.6f}')

        # record column norms
        norms = np.fmax(1e-15, np.sqrt(np.sum(W**2,0)))
        list_sigma_.append(norms)

        # check convergence
        if diffW <tol:
            print('converged after {:} iterations'.format(nIter))
            nIter=maxIter


    results =  dict({'W' : W,
                    'norms' : norms,
                    'obj' : list_reconstruction_err_,
                    'sigma': list_sigma_})

    return results



def JNMF(data, n_components, init='svd', method='Euclidean', maxIter=10000, verbose=True):

    """Joint Non-negative matrix factorisation.

        https://github.com/mstrazar/iONMF/blob/master/ionmf/factorization/model.py for similar

        Parameters
        ----------
        data : dict,
            data = {
                    "data_source_1": X_1 array [n_samples, n_features_1],
                    "data_source_2": X_2 array [n_samples, n_features_2],
                    ...
                    "data_source_N": X_N array [n_samples, n_features_N],
                    }
        Data sources must match in the number of rows.

        These arguments just pass straight throught to PNMF
        n_components : int
        The number of components for the approximation.
        init : 'svd' | 'random'
        NMF initialisation, if SVD then NND-SVD will be used, otherwise random matrices will be used
        method: 'KL' | 'Euclidean'
        error to minimise
        max_iter : int
        Maximum number of iterations to run

        Returns
        -------
        results, dict
        W : final solution, where H = W'*X and X = W*H
        obj : reconstruction error over iterations

        """
    keys = list(data.keys())
    n_sources = len(keys)
    n_samples = data[keys[0]].shape[0]
    n_features = sum([data[ky].shape[1] for ky in keys])

    # stack data together for PNMF
    X = np.zeros((n_samples, n_features))
    t = 0
    for n,k in enumerate(data):
        X[:,t:t+data[k].shape[1]] = data[k]
        t += data[k].shape[1]

    # run PNMF
    pnmfresults = PNMF(X.T, n_components, init=init, method=method, maxIter=maxIter, verbose=verbose)
    all_W = pnmfresults['W']

    # separate out W for each data matrix
    W = dict()
    t = 0
    for n,k in enumerate(data):
        W[keys[n]] = W[t:t+data[k].shape[1],:]


    results =  dict({'W' : W,
                    'obj' : pnmfresults['obj']})

    return results

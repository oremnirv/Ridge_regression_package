import numpy as np
import itertools

def x_from_standard_multivariate_normal(dim_x, num_observ):
    x = np.random.multivariate_normal(
        np.zeros(dim_x), np.eye(dim_x), size=num_observ)
    return x


def noise_from_univar_normal(mean, var, num_observ):
    noise = np.random.normal(mean, var, num_observ)
    return(noise)


def stnorm_data_for_ridge(dim_x, dim_w, num_observ, noise_mean, noise_std):

    x = np.asmatrix(x_from_standard_multivariate_normal(dim_x, num_observ))

    w = (x_from_standard_multivariate_normal(dim_x, 1)).transpose()

    noise = np.asmatrix(noise_from_univar_normal(
        noise_mean, noise_std, num_observ)).reshape(num_observ, 1)

    # create yi=xi*wi+noise,in Matrix notation this could be viewed as y=x*w+noise
    y = (x * w) + noise

    return(x, y, w)


def rand_idx_split( tr_abs, te_abs, val_abs):
    # randomaly permutate numbers, then split train & test data accordingly.

    num_obser = tr_abs + te_abs + val_abs
    rand_0_1 = np.random.rand(num_obser, 1)

    indices = np.random.permutation(rand_0_1.shape[0])

    tr_idx, te_idx, val_idx = indices[:tr_abs], indices[tr_abs:(
        te_abs + tr_abs)], indices[(te_abs + tr_abs):num_obser]

    tr_idx, te_idx, val_idx = np.asarray(sorted(tr_idx)), np.asarray(
        sorted(te_idx)), np.asarray(sorted(val_idx))
    return(tr_idx, te_idx, val_idx)


def split_tr_te(num_obser, tr_perc=0.66):
    rand_0_1 = np.random.rand(num_obser, 1)

    indices = np.random.permutation(rand_0_1.shape[0])

    tr_abs = int(num_obser * tr_perc)

    tr_idx, te_idx = indices[:tr_abs], indices[tr_abs:]

    tr_idx, te_idx = np.asarray(sorted(tr_idx)), np.asarray(
        sorted(te_idx))

    return(tr_idx, te_idx)

def cv_split(k, iter_num, x, y, tr_val_idx):
    '''# k - number of folds
    # iter - iteration number
    # x - training set
    # y- trianing targets
    # tr_val_idx - all indices of columns coressponding to training or validation
    # RETURNS: new train and validation sets'''

    val_st_idx = len(tr_idx) * (iter_num - 1) / k
    val_end_idx = len(tr_idx) * iter_num / k

    val_idx = tr_idx[val_st_idx : val_end_idx]
    tr_idx = tr_idx[~np.isin(tr_idx, val_idx)]

    x_val = x[val_idx]
    y_val = y[val_idx]
    x_tr = x[tr_idx]
    y_tr = y[tr_idx]

    return(x_tr, y_tr, x_val, y_val, tr_idx, val_idx)

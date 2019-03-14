# coding: utf-8
from loss_func import ridge
from ridge_data import ridge_data
from dual import dual
from primal import primal
import scipy.io
import numpy as np
from numpy.linalg import inv
import decimal
import pandas


# source: Kernel Methods for Pattern Analysis by J.shaw Taylor

'''embed the data into a space where the patterns can be discovered
as linear relatio'''
# a valid kernel is one which is positive semi definite, i.e all eigen
# values are ≥ 0, meaning the kernel matrix scales space either
# positively or by zero.

'''Use primal ridge when there are more training examples than dimensions
use dual ridge when there are more dimensions than training examples'''


# A function to get a series of numbers with a set jump between terms.
def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)


def main():
    data = scipy.io.loadmat(
        './boston.mat')


    x, y = data['boston'][:, :13], data['boston'][:, 13]
    rid_data = ridge_data(x, y)

    rid_model = dual(rid_data.x, rid_data.y)
    pp = drange(7, 13.5, 0.5)
    rid_model.sigma = [2 ** powr for powr in np.array(list(pp))]
    powers = range(-20, -2, 1)
    rid_model.regularizer = [2 ** powr for powr in powers]
    combo = rid_model.create_params_combo(rid_model.regularizer)


    cv_errs = np.zeros((3, rid_data.cv))
    err_w_params = np.zeros((3, len(combo)))
    trial_times_comb = np.zeros((len(combo), rid_model.trials))
    trial_err_dict = {'train': trial_times_comb,'validation': trial_times_comb,'test': trial_times_comb}

    for trial in xrange(rid_model.trials):
        print ('trial number:', trial)
        count_comb = 0

        trval_idx, te_idx, x_trval, y_trval, x_te, y_te = rid_data.trval_te_idx_split()

        for comb in combo:

            for iter_num in xrange(1, rid_data.cv + 1, 1):

                try:

                    regulizer = comb[0]

                except:
                    regulizer = comb

                try:
                    if(len(comb) == 2):
                        sig = comb[1]
                except:
                    sig = None

                val_idx, x_val, y_val, tr_idx, x_tr, y_tr = rid_data.split_cv(
                    iter_num, trval_idx)

                rid_model.calc_kernel_mat('gaus', sig)

                tr_ker = rid_model.kernel_split(tr_idx, tr_idx)
                val_ker = rid_model.kernel_split(val_idx, tr_idx)
                trval_ker = rid_model.kernel_split(trval_idx, trval_idx)
                te_ker = rid_model.kernel_split(te_idx, trval_idx)

                alpha_trval = rid_model.calc_alpha(
                    trval_ker, y_trval, regulizer)
                alpha_val = rid_model.calc_alpha(tr_ker, y_tr, regulizer)

                trval_pred = rid_model.predict(trval_ker, alpha_trval)
                val_pred = rid_model.predict(val_ker, alpha_val)
                te_pred = rid_model.predict(te_ker, alpha_trval)

                trval_mse = rid_model.mse(y_trval, trval_pred)
                val_mse = rid_model.mse(y_val, val_pred)
                te_mse = rid_model.mse(y_te, te_pred)

                cv_errs[:, (iter_num - 1)] = np.array([trval_mse, val_mse, te_mse]).reshape(-1)

            err_w_params[:, count_comb] = np.array([np.mean(cv_errs, 1)])
            count_comb += 1

        trial_err_dict['train'][:, trial] = err_w_params[0, :].reshape(-1)
        trial_err_dict['validation'][:, trial] = err_w_params[1, :].reshape(-1)
        trial_err_dict['test'][:, trial] = err_w_params[2, :].reshape(-1)

    avg_val_err = np.mean(trial_err_dict['validation'], 1).reshape(-1, 1)

    print avg_val_err
    loc = int(np.where(avg_val_err == min(avg_val_err))[0])
    mini =  min(trial_err_dict['validation'][:, 1])
    print min(trial_err_dict['validation'][:, 1])
    print min(trial_err_dict['train'][:, 1])
    print min(trial_err_dict['test'][:, 1])


if __name__ == '__main__':
    main()

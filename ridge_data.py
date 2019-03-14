import numpy as np
import scipy.io


class ridge_data(object):
    """docstring for ridge_data"""
    tr_percentage = 0.66
    cv = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def trval_te_idx_split(self):
        rand_0_1 = np.random.rand(self.x.shape[0], 1)

        indices = np.random.permutation(rand_0_1.shape[0])

        tr_abs = int(self.x.shape[0] * self.tr_percentage)

        trval_idx, te_idx = indices[:tr_abs], indices[tr_abs:]

        self.trval_idx, self.te_idx = np.asarray(
            sorted(trval_idx)), np.asarray(sorted(te_idx))

        self.x_trval, self.y_trval = self.x[self.trval_idx], self.y[self.trval_idx]
        self.x_te, self.y_te = self.x[te_idx], self.y[te_idx]

        return(self.trval_idx, self.te_idx, self.x_trval, self.y_trval, self.x_te, self.y_te)

    def split_cv(self, iter_num, trval_idx):
        val_st_idx = int(len(trval_idx) * (iter_num - 1) / self.cv)
        val_end_idx = int(len(trval_idx) * iter_num / self.cv)

        self.val_idx = trval_idx[val_st_idx: val_end_idx]
        self.tr_idx = trval_idx[~np.isin(trval_idx, self.val_idx)]

        self.x_val, self.y_val = self.x[self.val_idx], self.y[self.val_idx]
        self.x_tr, self.y_tr = self.x[self.tr_idx], self.y[self.tr_idx]

        return(self.val_idx, self.x_val, self.y_val, self.tr_idx, self.x_tr, self.y_tr)


def main():
    data = scipy.io.loadmat('/Users/omer/Documents/studies/supervised_learning/SL_assignment_1/boston.mat')
    x, y = data['boston'][:,:13], data['boston'][:,13]
    rid_data = ridge_data(x, y)
    a, b, c, d, e, f = rid_data.trval_te_idx_split()
    print a, b
    l,ll,g, h, i, j = rid_data.split_cv(5, a)
    print l

if __name__ == '__main__':
    main()
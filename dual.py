from ridge import ridge
import numpy as np
import scipy.io
from numpy.linalg import inv
import itertools


class dual(ridge):
    """docstring for ClassName"""
    kernel_type = 'lin'
    sigma = None

    def __init__(self, x, y):
        super(dual, self).__init__(x, y)

    def create_params_combo(self, regularizer):
        self.combo = list(itertools.product(regularizer, self.sigma))
        return(self.combo)

    def gaus_kernel(self, x, sig):
        '''page 77 J.shaw taylor
        also called RBF kernel, it corresponds
        to applying a gausian with mean at point z and get the prob. that
        point x came from that that gausian dist.. So a prediction at a
        new point can be viewed as a weighted combination of my
        probability to belong to any one of the gausians'''
        nrow, ncol = np.shape(x)
        K = np.matmul(x, np.transpose(x)) / (sig ** 2)
        d = np.diag(K)
        K1 = K - (d.reshape(-1, 1) / (2 * (sig ** 2)))
        K2 = K1 - (d.reshape(1, -1) / (2 * (sig ** 2)))
        K3 = np.exp(K2)
        return K3

    def calc_kernel_mat(self, kernel_type, sig=None):
        if (kernel_type == 'lin'):
            ''' corresponds to regular linear regression'''
            self.ker = np.asmatrix(x.dot(x.transpose()))

        if (kernel_type == 'quad'):
            '''corresponds as one example to
            feature map (x1^2, x2^2, sqroot2*x1*x2) '''
            self.ker = np.asmatrix(
                np.square(np.asarray(self.x.dot(self.x.transpose()))))

        if (kernel_type == 'gaus'):
            '''infinite dimensional kernel - note that
            exponential can be written as an infinite dim series'''
            self.ker = self.gaus_kernel(self.x, sig)

    def kernel_split(self, row_idx, col_idx):
        self.partial_ker = self.ker[row_idx[:, None], col_idx]

        return(self.partial_ker)

    def calc_alpha(self, kernel, _y, regu):
        self.alpha = (inv(kernel + (regu) *
                          (np.eye(kernel.shape[0])))).dot(_y)
        return(self.alpha)

    def predict(self, kernel, alpha):

        self.y_hat = kernel.dot(alpha)

        return(self.y_hat)


def main():
    data = scipy.io.loadmat('/Users/omer/Documents/studies/supervised_learning/SL_assignment_1/boston.mat')
    x, y = data['boston'][:,:13], data['boston'][:,13]
    rid_data = dual(x, y)
    rid_data.calc_kernel_mat('gaus', sig=0.1)

if __name__ == '__main__':
    main()

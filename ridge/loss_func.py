import numpy as np
import scipy.io

class ridge(object):
    trials = 3
    regularizer = []
    """docstring for ClassName"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def mse(self, y_true, y_hat):
        self.err = sum(np.square(np.asarray(y_true.reshape(-1, 1) - y_hat.reshape(-1, 1)))) / (y_true.shape[0])

        return(self.err)


    def choose_best_comb(self):
        pass

    def store_results(self):
        pass


def main():
    data = scipy.io.loadmat('./boston.mat')
    x, y = data['boston'][:,:13], data['boston'][:,13]
    rid_data = ridge(x, y)

if __name__ == '__main__':
    main()

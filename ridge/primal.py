from loss_func import ridge
import numpy as np
from numpy.linalg import inv

class primal(ridge):
    """docstring for ClassName"""
    def __init__(self, arg):
        super(ridge, self).__init__()
        self.arg = arg

    def calc_weights(self, tr_x, tr_y, regu):
        self.w_hat = ((inv(self.tr_x.transpose().dot(self.tr_x) + (self.regu) *
                  (np.eye(self.tr_x.shape[1])))).dot(self.tr_x.transpose())).dot(self.tr_y)

        return(self.w_hat)

    def predict(self, _x, w_hat):
        self.y_hat = self._x.dot(self.w_hat)

        return(self.y_hat)


def main():
    pass

if __name__ == '__main__':
    main()
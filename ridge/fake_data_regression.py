    from normal_dist_data_creation import x_from_standard_multivariate_normal, noise_from_univar_normal
    import numpy as np

    def stnorm_z_data_ridge(dim_x, num_observ, noise_mean, noise_std):

        x = np.asmatrix(x_from_standard_multivariate_normal(dim_x, num_observ))
        # Create Z by using the matrix multiplication of xi with xi for all i,
        # and then stack matrix to vector form this is an expensive
        # representation since we explicitly calculate higher dim and not using
        # kernel trick
        z = np.zeros((dim_x ** 2, num_observ))
        for row in range(num_observ):
            z[:, row] = ((np.asmatrix(x[row, :])).transpose().dot(
                np.asmatrix(x[row, :]))).reshape(dim_x ** 2)

        try:
            noise = np.asmatrix(noise_from_univar_normal(
                noise_mean, noise_std, num_observ)).reshape(num_observ, 1)
        except ValueError:
            noise = 0

        w_z = (x_from_standard_multivariate_normal(dim_x ** 2, 1)).transpose()

        try:
            y_z = w_z.transpose().dot(z) + noise.transpose()
        except AttributeError:
            y_z = w_z.transpose().dot(z)
        return(x, z, w_z, y_z)

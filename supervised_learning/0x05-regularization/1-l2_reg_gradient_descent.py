import numpy as np
""" module """

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ l2_reg_gradient_descent"""
    m = Y.shape[1]
    
    copy_w = weights.copy()
    for i in reversed(range(1, L + 1)):


        dw1 = cache['A{}'.format(i - 1)].T
        if i == L - 1:

            W = 'W{}'.format(i)
            b = 'b{}'.format(i)
            dz = cache['A{}'.format(i)] - Y
        else:
            W = 'W{}'.format(i + 1)
            b = 'b{}'.format(i + 1)
            A = 'A{}'.format(i)

            dz1 = np.matmul(copy_w[W].T, dz)
            dz = dz1 * (1 - cache[A]**2)

        dW = np.matmul(dz, dw1) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        dW_grad = dW + (lambtha / m) * copy_w[W]

        weights[W] = copy_w[W] - (alpha * dW_grad)
        weights[b] = copy_w[b] - (alpha * db)


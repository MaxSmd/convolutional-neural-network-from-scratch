import numpy as np

def categorical_cross_entropy(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    cce_loss = -np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(cce_loss)

def categorical_cross_entropy_prime(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    grad = - (y_true / y_pred) / y_true.shape[0]
    return grad
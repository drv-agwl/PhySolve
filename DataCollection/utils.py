import numpy as np


def gen_random_grads(mu, sigma):
    return np.random.normal(mu, sigma, (3,))

import numpy as np
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.style.use('fivethirtyeight')
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
import timeit
import time

image_shape = (64, 64)
# Load faces data
dataset = fetch_olivetti_faces('./')
faces = dataset.data.T

print('{} data points'.format(faces.shape[1]))
print('Shape of the faces dataset: {}'.format(faces.shape))


def mean_naive(X):
    "Compute the mean for a dataset X nby iterating over the data points"
    # X is of size (D,N) where D is the dimensionality and N the number of data points
    D, N = X.shape
    #mean = np.zeros((D, 1))
    mean = []
    ### Edit the code; iterate over the dataset and compute the mean vector.
    for d in range(D):
        total = 0
        for n in range(N):
            total += X[d, n]
            # Update the mean vector


            pass
        total = total / N
        mean.append(total)
    ###
    return mean


def cvariance(x, y, N, average):
    total = 0

    total = [((faces[x, n] - average.item(x)) * (faces[y, n] - average.item(y))) for n in range(N)]

    total = sum(total)

    return total


def cov_naive(X):
    """Compute the covariance for a dataset of size (D,N)
    where D is the dimension and N is the number of data points"""
    D, N = X.shape
    ### Edit the code below to compute the covariance matrix by iterating over the dataset.
    covariance = np.zeros((D, D))


    """Compute the covariance for a dataset of size (D,N)

    where D is the dimension and N is the number of data points"""

    # 1/N * \sum (x_i - m)(x_i - m)^T (where m is the mean)

    m = mean_naive(X)

    m = np.reshape(m, (2,1))

    U = (X - m).T

    covariance = (U.T @ U) / (X.shape[1])

    return covariance

    ###


def affine_mean(mean, A, b):
    """Compute the mean after affine transformation
    Args:
        x: ndarray, the mean vector
        A, b: affine transformation applied to x
    Returns:
        mean vector after affine transformation
    """
    ### Edit the code below to compute the mean vector after affine transformation
    affine_m = np.zeros(mean.shape)  # affine_m has shape (D, 1)
    ### Update affine_m
    affine_m = A.T @ mean @ A + b

    ###
    return affine_m


def mean(X):
    "Compute the mean for a dataset of size (D,N) where D is the dimension and N is the number of data points"
    # given a dataset of size (D, N), the mean should be an array of size (D,1)
    # you can use np.mean, but pay close attention to the shape of the mean vector you are returning.
    D, N = X.shape
    ### Edit the code to compute a (D,1) array `mean` for the mean of dataset.
    mean = np.zeros((D, 1))
    ### Update mean here

    mean = np.mean(X, axis=1).reshape(-1, 1)

    ###
    return mean


V = np.array([[3,4],[-1,-1])

temp = x.T @ inner @ y
def vectors_angle(x, y, inner):
    temp = temp / np.sqrt((x.T @ inner @ x) * (y.T @ inner @ y))
    answer = np.arccos(temp)
    return answer
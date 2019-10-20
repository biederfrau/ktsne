#!/usr/bin/env python3

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

from sys import exit

def save_embedding(X, labels, dir='img', prefix="embedding", nr=0):
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.title(f"t = {nr}")
    plt.savefig(f"{dir}/{prefix}-{nr}.png")
    plt.clf()

def plot_embedding(X, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()

def plot_errors(errors):
    plt.plot(range(errors.shape[0]), errors)
    plt.xlabel("epoch")
    plt.ylabel("kl divergence")
    plt.show()

np.random.seed(42)

def compute_entropy(pi):
    return -np.nansum(pi*np.log2(pi))

def compute_kl_divergence(P_ji, Q_ji):
    pass

def compute_distance_matrix(X):
    sum_sq = np.sum(X**2, axis=1, keepdims=True)
    dist_sq = sum_sq + sum_sq.T - 2*X@X.T
    return dist_sq.clip(min=0)

# computes eq (1). to compute q_j_given_i (affinities in low-dimensional
# map space), pass sigma = 1/sqrt(2)
def compute_pairwise_affinities(dist_sq_one_point, i, sigma=1/np.sqrt(2)):
    enumerators = np.exp(-dist_sq_one_point / (2*sigma**2))
    denomiators = np.exp(-dist_sq_one_point / (2*sigma**2))

    denomiators[i] = 0 # sum over k != i
    denomiator = np.sum(denomiators)

    p_j_given_i = enumerators / denomiator
    p_j_given_i[i] = 0 # zero out j = i

    return p_j_given_i

def run_binary_search(dist_sq_one_point, perp, i, tol=1e-5):
    lo, hi = 0.01, 666
    j = 0
    while True and j < 50:
        sigma_hat = (lo + hi)/2
        p_j_given_i = compute_pairwise_affinities(dist_sq_one_point, i, sigma_hat)

        H = compute_entropy(p_j_given_i)
        perp_diff = 2**H - perp

        if np.abs(perp_diff) < tol:
            break

        if perp_diff < 0: # 2**H is too small
            lo = sigma_hat
        else: # 2**H is too large
            hi = sigma_hat

        j += 1

    return sigma_hat

class NaiveTSNE():
    def __init__(self, n_epochs=200, perplexity=30, d=2, eta=100, k=None, k_ratio=None, stop_sampling=175, momentum=lambda t: 0.5 if t < 250 else 0.8, save_embeddings=False):
        self.n_iter = n_epochs
        self.perplexity = perplexity
        self.d = d
        self.eta = eta
        self.momentum = momentum
        self.errors = np.zeros(n_epochs)
        self.stop_sampling = stop_sampling or n_epochs
        self.save_embeddings = save_embeddings

    def _fit(self, X):
        n, m = X.shape

        sigma = np.full(n, np.nan)
        P_j_given_i = np.zeros((n, n))

        # compute pairwise affinities p_{j | i} with perplexity
        dist_sq_X = compute_distance_matrix(X)

        # tune sigma_i for every point
        for i in range(n):
            sigma[i] = run_binary_search(dist_sq_X[i], self.perplexity, i)
            P_j_given_i[i] = compute_pairwise_affinities(dist_sq_X[i], i, sigma[i])

        P_ij = P_j_given_i + P_j_given_i.T
        P_ij = P_ij / np.sum(P_ij)
        P_ij = np.maximum(P_ij, 1e-12)

        # sample some initial solution from N(0, 10**-4I)
        Y = np.random.normal(0, 10**-4, size=(n, self.d))

        dY = np.zeros_like(Y)
        iY = np.zeros_like(Y)
        gains = np.ones_like(Y)

        # run gradient descent
        for t in range(self.n_iter):
            dist_sq_Y = compute_distance_matrix(Y)
            qijZ = (1 + dist_sq_Y)**-1
            np.fill_diagonal(qijZ, 0)

            Q_ij = qijZ / np.sum(qijZ) # low dimensional affinities
            Q_ij = np.maximum(Q_ij, 1e-12)

            M = 12 if t < 50 else 1 # early exaggeration
            for i in range(n):
                F_attr = np.tile(M*P_ij[:, i]*qijZ[:, i], (self.d, 1)).T * (Y[i, :] - Y)
                F_rep = np.tile(Q_ij[:, i]*qijZ[:, i], (self.d, 1)).T * (Y[i, :] - Y)

                dY[i, :] = np.sum(F_attr - F_rep, 0)

            sign_match    = np.sign(dY) == np.sign(iY)
            sign_mismatch = np.invert(sign_match)

            gains[sign_match]    *= 0.8
            gains[sign_mismatch] += 0.2
            gains                 = np.maximum(gains, 0.01)

            iY = self.momentum(t)*iY - self.eta*(gains*dY)
            Y += iY

            Y = scale(Y, with_std=False)

            kl = np.nansum(P_ij * np.log2(P_ij / Q_ij))
            self.errors[t] = kl
            print(t, kl, linalg.norm(dY))

            if self.save_embeddings == True:
                save_embedding(Y, classes, nr=t)

        return Y

    def fit_transform(self, X):
        self.embedding_ = self._fit(X)
        return self.embedding_

    def fit(self, X):
        self.fit_transform(X)
        return self

from sklearn import datasets
data = datasets.load_digits().data
classes = datasets.load_digits().target

tsne = NaiveTSNE()
embedded = tsne.fit_transform(data)

plot_embedding(embedded, classes)
plot_errors(tsne.errors)

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 21:46:44 2023

"""
import numpy as np

def normalize(X):
    """Normalizes the given dataset X to have zero mean.
    Args:
        X: ndarray
    Returns:
        (Xbar, mean): tuple of ndarray,
            Xbar - normalized dataset
            mean - sample mean of the dataset   """

    N, D = X.shape
    mu = np.zeros((D,))
    mu = np.mean(X, axis = 0)
    Xbar = X - mu
    return Xbar, mu


def eig(S):
    """Computes the eigenvalues and eigenvectors for the covariance matrix S.
    Args:
        S: ndarray, covariance matrix
    Returns:
        (eigvals, eigvecs): ndarray, sorted in descending order of eigenvals    """

    eigvals, eigvecs = np.linalg.eig(S)
    sort_indices = np.argsort(eigvals)[::-1]
    return eigvals[sort_indices], eigvecs[sort_indices]

def projection_matrix(B):
    """Computes the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace
    Returns:
        P: projection matrix    """

    P = B @ np.linalg.inv(B.T @ B) @ B.T
    return P

def PCA(X, num_components):
    """
    Args:
        X: ndarray
        num_components: int, number of principal components to reduce to
    Returns:
        reconstructed data,
        the sample mean of the X,
        principal values,
        principal components    """

    X_normalized, mean = normalize(X)
    N, D = X.shape
    # computing data covariance matrix, eigen vectors and eigen values
    S = np.cov(X, rowvar = False, bias = True)
    S = (X_normalized.T @ X_normalized) / N
    eig_vals, eig_vecs = eig(S)
    
    # taking top `num_components` of eig_vals and eig_vecs. only take their real parts
    principal_vals, principal_components = eig_vals[:num_components], eig_vecs[:, num_components]
    principal_components = np.real(principal_components)
    principal_components = np.reshape(principal_components, (np.shape(principal_components)[0],1))

    P = projection_matrix(principal_components)

    # reconstructig the data by projecting the normalized data on the basis spanned by the principal components
    reconst = np.zeros_like(X_normalized)
    reconst = (P @ X_normalized.T).T + mean
    return reconst, mean, principal_vals, principal_components

X = np.array([[3, 6, 7],
              [8, 9, 0],
              [1, 5, 2]])

reconst, mean, principal_vals, principal_components = PCA(X, 1)
print('Reconstruction matrix: \n', reconst)
print('Sample mean: ', mean)
print('Principal values: ', principal_vals)
print('Principal components: ', principal_components)


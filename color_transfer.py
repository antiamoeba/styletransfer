import numpy as np

# TODO: Make more efficient so we don't have to compute
# the histogram each time
"""
Input: Style image S, Guess Image X

Output: Guess X, with transferred color through histogram matching
source: https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
"""
def color_transfer(S, X):
    oldshape = X.shape
    X = X.ravel()
    S = S.ravel()
    X_values, bin_idx, X_counts = np.unique(X, return_inverse=True,return counts=True)
    S_values, S_counts = np.unique(S, return_counts=True)
    X_quantiles = np.cumsum(X_counts).astype(np.float64)
    X_quantiles = X_quantiles / X_quantiles[-1]
    S_quantiles = np.cumsum(S_counts).astype(np.float64)
    S_quantiles = S_quantiles / S_quantiles[-1]
    interp_S_values = np.interp(X_quantiles, S_quantiles, S_quantiles)
    return interp_S_values[bin_idx].reshape(old_shape)

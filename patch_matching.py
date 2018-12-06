from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction import image
import numpy as np

"""
Input: Style image S, patch size (one dimension, square)

Output: [patch images], NearestNeighbors object for all patches in S

use auto for now, but it seems like BallTree is better
for higher dimensions
"""
def construct_matcher(S, patch_size):
    patches = image.extract_patches_2d(S, (patch_size, patch_size))
    return patches, NearestNeighbors(n_neighbors=1, algorithm="auto").fit(patches)

"""
Input: img, sample_gap is subsampling gap

Output: [(y, x)] of upper-left corners of image neighborhoods

currently uses np.linspace (patches will overlap the normal
amount or even more)
"""
def get_neighborhoods(img, sample_gap):
    ys = np.linspace(0, img.shape[0], num=(img.shape[0]+sample_gap-1)//sample_gap, endpoint=False)
    xs = np.linspace(0, img.shape[1], num=(img.shape[1]+sample_gap-1)//sample_gap, endpoint=False)
    ys, xs = np.meshgrid(ys, xs)
    return ys, xs
    

# TODO: make this more efficient so you only need to make the NN
# classifier once per time required
# TODO: Vectorize
# TODO: Add option to use other neighborhoods
"""
Input: Style image S, guess image X, patch size, subsampling gap
patch size should be >= subsampling gap

Output: Corresponding patches for each sample from X to S
{ (y, x) : patch image }, (y, x) is the upper left corner

"""
def find_nearest_neighbors(S, X, patch_size, sample_gap):
    S_patches, matcher = construct_matcher(S, patch_size)
    neighb_ys, neigh_xs = get_neighborhoods(X, sample_gap)
    X_patches = []
    for pt in np.dstack([neighb_ys, neigh_xs]):
        y, x = pt[0], pt[1]
        X_patches.append(X[y:y+patch_size, x:x+patch_size])
    X_patches = np.array(X_patches)
    distances, indices = matcher.kneighbors(X_patches)
    matches = S_patches[indices]
    return {neighborhoods[i] : matches[i] for i in range(len(neighborboods))}




import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction import image
import numpy as np
import pdb

# TODO: Do we need to change the default heuristic of how to
# determine how patches are close?
"""
Input: Style image S, patch size (one dimension, square)

Output: [patch images], NearestNeighbors object for all patches in S

use auto for now, but it seems like BallTree is better
for higher dimensions
"""
def construct_matcher(S, patch_size):
    patches = image.extract_patches_2d(S, (patch_size, patch_size))
    patches = np.reshape(patches, (patches.shape[0], -1))
    return patches, NearestNeighbors(n_neighbors=1, algorithm="auto").fit(patches)

"""
Input: img, sample_gap is subsampling gap

Output: [(y, x)] of upper-left corners of image neighborhoods

currently uses np.linspace (patches will overlap the normal
amount or even more)
"""
def get_neighborhoods(img, sample_gap, patch_size):
    ys = np.linspace(0, img.shape[0] - patch_size, num=(img.shape[0]+sample_gap-1)//sample_gap, endpoint=True)
    xs = np.linspace(0, img.shape[1] - patch_size, num=(img.shape[1]+sample_gap-1)//sample_gap, endpoint=True)
    ys, xs = np.meshgrid(ys, xs)
    ys, xs = ys.astype(int).flatten(), xs.astype(int).flatten()
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
    neighb_ys, neighb_xs = get_neighborhoods(X, sample_gap, patch_size)
    X_patches = []
    for i in range(len(neighb_ys)):
        y, x = neighb_ys[i], neighb_xs[i]
        X_patches.append(X[y:y+patch_size, x:x+patch_size])
    X_patches = np.array(X_patches).reshape(len(X_patches), -1)
    distances, indices = matcher.kneighbors(X_patches)
    matches = S_patches[indices].reshape(X_patches.shape[0], patch_size, patch_size, 3)
    neighborhoods = np.array([neighb_ys, neighb_xs]).T
    return {(neighborhoods[i][0], neighborhoods[i][1]) : matches[i] for i in range(len(neighborhoods))}

if __name__ == "__main__":
    import skimage.io as skio
    img = skio.imread("img/starry_tiny.jpg")
    ret = find_nearest_neighbors(img, img, 60, 50)
    results = list(ret.values())  
    for i in range(3):
        skio.imshow(results[i])
        skio.show()

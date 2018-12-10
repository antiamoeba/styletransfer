import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction import image
import numpy as np
import pdb

class PatchMatcher:

    # TODO: Make it so we don't actually have to save all the patches, like use indices instead?
    """
    S is the style image and patch_size is the length and width of the patch
    """
    def __init__(self, S, patch_size):
        self.S = S
        self.patch_size = patch_size
        self.S_patches = []
        self.yw, self.xw = np.meshgrid(np.arange(self.patch_size), np.arange(self.patch_size))
        self.yw = self.yw.flatten()
        self.xw = self.xw.flatten()
        self.matcher = None 
        self.construct_matcher()

    # TODO: Do we need to change the default heuristic of how to
    # determine how patches are close?
    """
    Creates [patch images] and NearestNeighbors object for all patches in S
    """
    def construct_matcher(self):
        patches = image.extract_patches_2d(self.S, (self.patch_size, self.patch_size))
        self.S_patches = np.reshape(patches, (patches.shape[0], -1))
        self.matcher = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(self.S_patches)

    """
    Input: img, sample_gap is subsampling gap

    Output: [(y, x)] of upper-left corners of image neighborhoods

    currently uses np.linspace (patches will overlap the normal
    amount or even more)
    """
    def get_neighborhoods(self, img, sample_gap):
        ys = np.linspace(0, img.shape[0] - self.patch_size, num=(img.shape[0]+sample_gap-1)//sample_gap, endpoint=True)
        xs = np.linspace(0, img.shape[1] - self.patch_size, num=(img.shape[1]+sample_gap-1)//sample_gap, endpoint=True)
        ys, xs = np.meshgrid(ys, xs)
        ys, xs = ys.astype(int).flatten(), xs.astype(int).flatten()
        return ys, xs
    
    # TODO: Add option to use other neighborhoods
    """
    Input: Guess image X, subsampling gap
    subsampling gap should be <= patch size

    Output: Corresponding patches for each sample from X to S
    [[y1, y2, y3], [x1, x2, x3]] and [patch images],
    (y1, x1) is the upper left corner
    """
    def find_nearest_neighbors(self, X, sample_gap):
        neighb_ys, neighb_xs = self.get_neighborhoods(X, sample_gap)
        X_patches = X[neighb_ys[:, None] + self.yw, neighb_xs[:, None] + self.xw, :]
        X_patches = np.array(X_patches).reshape(len(X_patches), -1)
        distances, indices = self.matcher.kneighbors(X_patches)
        matches = self.S_patches[indices].reshape(X_patches.shape[0], self.patch_size, self.patch_size, 3)
        neighborhoods = np.array([neighb_ys, neighb_xs])
        return neighborhoods, matches

if __name__ == "__main__":
    import skimage.io as skio
    img = skio.imread("images/starry_tiny.jpg")
    img2 = skio.imread("images/cat_small.jpg")
    matcher = PatchMatcher(img, 30)
    coords, images = matcher.find_nearest_neighbors(img2, 27)
    for i in range(3):
        skio.imshow(images[i])
        skio.show()

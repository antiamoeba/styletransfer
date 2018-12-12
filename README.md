# styletransfer
### No machine learning - the best way
#### Dorian Chan & Michelle Hwang

For transferring styles from a style image to a content image. Based on the
paper [Style-Transfer via
Texture-Synthesis](https://arxiv.org/pdf/1609.03057.pdf) by Michael Elad and
Peyman Milanfar. 

Report is [here](paper.pdf).

## Usage

Install dependencies with `pip install -r requirements.txt`.

Import the style_transfer module and use the `style_transfer` function as
documented, or run within the `style_transfer.py` file. See `style_transfer.py`
for example usage.

## Other Files

* `images`: folder containing the content (and mask) and style images we used.
* `outputs`: folder containing some images resulting as outputs of `style_transfer`.
* `patch_matching.py`: For matching patches between images based on their L2 distance.
* `patch_matching_pca.py`: For matching patches between images based on the L2
  distances between their PCA decompositions.
* `robust.py`: For aggregating patches.
* `fusion.py`: For fusing images together.
* `color_transfer.py`: For performing color transfers from a source to target
  image based on histogram matching.
* `denoise.py`: For denoising images.

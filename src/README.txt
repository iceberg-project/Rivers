

This repo is an implementation of U-Net for river extraction and depends on the following libraries:

- Keras \&gt;= 1.0
- Numpy
- Tifffile

This code should be compatible with Python versions 2.7-3.5.

Training for this model was done on a NVIDIA Tesla P100 16GB GPU.

_Training and test phases of U-Net to extract rivers_

**I. Pre-processing &amp; tiling (in Matlab):**

1. Copy the orthorectified 16-bit 8-band WordView (WV) surface reflectance imagery of the training area to Rivers/src/tiling
2. Copy the corresponding 8-bit river mask (hand-digitized) to Rivers/src/tiling
3. Run tiling\_pretraining.m in the Rivers/src/tiling

\* MAKE SURE saveastiff.m function is in the same directory as tiling\_pretraining.m tiling\_pretraining.m tiles the WV image and its corresponding river mask into smaller images of 800×800 pixels with steps of 100 pixels, and saves the generated multi-page tils in the &#39;tiled multi-page image&#39; and &#39;tiled multi-page river mask&#39; folders. The entire data set is comprised of 209 image tiles and their corresponding river mask tiles.

1. Run randselect\_pretraining.m in the Rivers/src/training

randselect\_pretraining.m makes a &#39;data&#39; directory and randomly:

- renames and saves 80% (167 tiles) and 20% (42 tiles) of the generated multi-page image tiles in &#39;data/image\_tiles\_fortraining&#39; and &#39;data/image\_tiles\_fortest&#39;, respectively.
- renames and saves 80% (167 tiles) and 20% (42 tiles) of the generated multi-page river mask tiles in &#39;data/mask\_tiles\_fortraining&#39; and &#39;data/mask\_tiles\_fortest&#39;, respectively.

**II. Training (in Python):**

1. Run train\_unet.py in the Rivers/src/training

\* MAKE SURE data, gen\_patches.py, and unet\_model.py are in the same directory as train\_unet.py

Training is performed for 150 epochs (35 sec per epoch), with a batch size of 32 patches. Using a NVIDIA Tesla P100 GPU the training lasts for about 1.5 hours. Training weights are saved as &#39;unet\_weights.hdf5&#39; in the weights folder.

**III. Testing (in Python):**

1. Run predict\_fortest.py in the Rivers/src/training

predict\_fortest.py reads 42 image tiles of 800\*800 pixels from data/image\_tiles\_fortest and saves map of each tile showing river pixels with more than 50% probability in the &#39;data/mask\_tiles\_fortest\_predicted&#39;.

_Prediction phase of U-Net to extract rivers_

**I. Prediction (in Python):**

1. Run predict.py in the Rivers/src/training

\* MAKE SURE the orthorectified 16-bit 8-band WV image and its multi-page tiff are in the same directory as predict.py

\*\* MAKE SURE to enter the name of the WorldView image in line 11 of the code:

image= normalize(tiff.imread(&#39;name of the WV image.tif&#39;).transpose([1,2,0]))

predict.py uses the trained U-Net weights to process the multi-page tiff with 800×800 sliding-window with 50% overlap, and generates 800×800 maps where each pixel value is the probability of the pixel belonging to the river class. The maps are saved in the data/WV\_predicted.

**II. Image mosaic (in Matlab):**

1. Run mosaic.m in the Rivers/src/training

\* MAKE SURE natsort.m and natsortfiles.m functions are in the same directory as mosaic.m

mosaic.m merges the 800\*800 maps of prediction in data/WV\_predicted into an image-level prediction, and exports the Image-level prediction back to a GeoTiff file.



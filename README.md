# Rivers (Arctic hydrology)

We provide a classification algorithm for ice surface features from high-resolution imagery.  This algorithm was developed by convolutional neural network training to detect regions of large and small rivers and to distinguish them from crevasses and non-channelized slush. We also provide a detection algorithm to extract polyline water features from the classified high-probability river areas.

## Prerequisites - all available on bridges via the commands below
- Linux
- Python 3
- CPU and NVIDIA GPU + CUDA CuDNN

## Software Dependencies - these will be installed automatically with the installation below.
- numpy
- scipy
- tifffile
- keras==2.3.1
- opencv-python
- rasterio
- affine
- pygdal==3.0.4
- tensorflow-gpu==1.15.0

## Installation
Preliminaries:  
These instructions are specific to XSEDE Bridges2 but other resources can be used if cuda, python3, and a NVIDIA P100 GPU are present, in which case 'module load' instructions can be skipped, which are specific to Bridges.  
  
For Unix or Mac Users:    
Login to Bridges 2 via ssh using a Unix or Mac command line terminal.  Login is available to bridges directly or through the XSEDE portal. Please see the [Bridges 2 User's Guide](https://www.psc.edu/resources/bridges-2/user-guide-2/).  

For Windows Users:  
Many tools are available for ssh access to Bridges 2.  Please see [Ubuntu](https://ubuntu.com/tutorials/tutorial-ubuntu-on-windows#1-overview), [MobaXterm](https://mobaxterm.mobatek.net) or [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/)

### PSC Bridges 2
Once you have logged into Bridges 2, you can follow one of two methods for installing iceberg-rivers.

#### Method 1 (Recommended):  

The lines below following a '$' are commands to enter (or cut and paste) into your terminal (note that all commands are case-sensitive, meaning capital and lowercase letters are differentiated.)  Everything following '#' are comments to explain the reason for the command and should not be included in what you enter.  Lines that do not start with '$' or '[rivers_env] $' are output you should expect to see.


```bash
$ pwd
/home/username
$ cd $PROJECT                      # switch to your working space.
$ mkdir Rivers                      # create a directory to work in.
$ cd Rivers                         # move into your working directory.
$ module load AI/anaconda3-tf1.2020.11
$ export PATH=/ocean/projects/mcb110096p/paraskev/gdal-3.0.4/bin:$PATH
$ export LD_LIBRARY_PATH=/ocean/projects/mcb110096p/paraskev/gdal-3.0.4/lib:$LD_LIBRARY_PATH
$ export GDAL_DATA=/ocean/projects/mcb110096p/paraskev/gdal-3.0.4/share/gdal
$ conda create --prefix iceberg_rivers --clone $AI_ENV
$ source activate iceberg_rivers/
[iceberg_rivers] $ pwd
/ocean/projects/group/username/Rivers
[iceberg_rivers]$ export PYTHONPATH=/ocean/projects/group/username/iceberg_rivers/lib/python3.7/site-packages/
[iceberg_rivers]$ pip install iceberg_rivers.search
```

#### Method 2 (Installing from source; recommended for developers only): 

```bash
$ cd $PROJECT                      # switch to your working space.
$ mkdir Rivers                      # create a directory to work in.
$ cd Rivers                         # move into your working directory.
$ module load AI/anaconda3-tf1.2020.11
$ export PATH=/ocean/projects/mcb110096p/paraskev/gdal-3.0.4/bin:$PATH
$ export LD_LIBRARY_PATH=/ocean/projects/mcb110096p/paraskev/gdal-3.0.4/lib:$LD_LIBRARY_PATH
$ export GDAL_DATA=/ocean/projects/mcb110096p/paraskev/gdal-3.0.4/share/gdal
$ conda create --prefix iceberg_rivers --clone $AI_ENV
$ source activate iceberg_rivers/
[rivers_env] $ pwd
/ocean/projects/group/username/Rivers
$ git clone https://github.com/iceberg-project/Rivers.git
[iceberg_rivers] $ export PYTHONPATH=/ocean/projects/group/username/iceberg_rivers/lib/python3.7/site-packages/
[iceberg_rivers] $ pip install .
```

#### To test
```bash
[iceberg_rivers] $ deactivate       # exit your virtual environment.
$ interact --gpu  # request a compute node.  This package has been tested on P100 GPUs on bridges, but that does not exclude any other resource that offers the same GPUs. (this may take a minute or two or more to receive an allocation).
$ cd $PROJECT/Rivers                # make sure you are in the same directory where everything was set up before.
$ module load AI/anaconda3-tf1.2020.11
$ export PATH=/ocean/projects/mcb110096p/paraskev/gdal-3.0.4/bin:$PATH
$ export LD_LIBRARY_PATH=/ocean/projects/mcb110096p/paraskev/gdal-3.0.4/lib:$LD_LIBRARY_PATH
$ export GDAL_DATA=/ocean/projects/mcb110096p/paraskev/gdal-3.0.4/share/gdal
$ source activate iceberg_rivers/    # activate your environment, no need to create a new environment because the Rivers tools are installed and isolated here.
[iceberg_rivers] $ export PYTHONPATH=/ocean/projects/group/username/iceberg_rivers/lib/python3.7/site-packages/
[iceberg_rivers] $ iceberg_rivers.tiling --help  # this will display a help screen of available usage and parameters.
```
## Prediction
- Download a pre-trained model at: 

You can download to your local machine and use scp, ftp, rsync, or Globus to [transfer to bridges](https://portal.xsede.org/psc-bridges).

Rivers predicting is executed in three steps: 
First, follow the environment setup commands under 'To test' above. Then create tiles from an input GeoTiff image and write to the output_folder. The scale_bands parameter (in pixels) depends on the trained model being used.  The default scale_bands is 299 for the pre-trained model downloaded above.  If you use your own model the scale_bands may be different.
```bash
[iceberg_rivers] $ iceberg_rivers.tiling --tile_size=224 --step=112 --input=<image_abspath> --output=./test/
```
Then, detect rivers on each tile and output counts and confidence for each tile.
```bash
[iceberg_rivers] $ iceberg_rivers.predict --input <tile_folder> -o <output_folder> -w <model>
```
Finally, mosaic all the tiles back into one image
```bash
[iceberg_rivers] $ iceberg_rivers.mosaic --input_WV image --input <masks_folder> --tile_size 224 --step 112 --output_folder ./mosaic
```

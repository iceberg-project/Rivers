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
- keras==2.2.4
- opencv-python
- rasterio
- affine
- pygdal==2.2.1
- tensorflow-gpu==1.7.1

## Installation
Preliminaries:  
These instructions are specific to XSEDE Bridges but other resources can be used if cuda, python3, and a NVIDIA P100 GPU are present, in which case 'module load' instructions can be skipped, which are specific to Bridges.  
  
For Unix or Mac Users:    
Login to bridges via ssh using a Unix or Mac command line terminal.  Login is available to bridges directly or through the XSEDE portal. Please see the [Bridges User's Guide](https://portal.xsede.org/psc-bridges).  

For Windows Users:  
Many tools are available for ssh access to bridges.  Please see [Ubuntu](https://ubuntu.com/tutorials/tutorial-ubuntu-on-windows#1-overview), [MobaXterm](https://mobaxterm.mobatek.net) or [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/)

### PSC Bridges
Once you have logged into bridges, you can follow one of two methods for installing iceberg-rivers.

#### Method 1 (Recommended):  

The lines below following a '$' are commands to enter (or cut and paste) into your terminal (note that all commands are case-sensitive, meaning capital and lowercase letters are differentiated.)  Everything following '#' are comments to explain the reason for the command and should not be included in what you enter.  Lines that do not start with '$' or '[rivers_env] $' are output you should expect to see.


```bash
$ pwd
/home/username
$ cd $SCRATCH                      # switch to your working space.
$ mkdir Rivers                      # create a directory to work in.
$ cd Rivers                         # move into your working directory.
$ module load keras/2.2.0_tf1.7_py3_gpu gdal/2.2.1  # load keras libraries and GDAL.
$ virtualenv rivers_env --system-site-packages # create a virtual environment to isolate your work from the default system.
$ source rivers_env/bin/activate    # activate your environment. Notice the command line prompt changes to show your environment on the next line.
[rivers_env] $ pwd
/pylon5/group/username/Rivers
[rivers_env] $ export PYTHONPATH=<path>/rivers_env/lib/python3.6/site-packages:$PYTHONPATH # set a system variable to point python to your specific code. (Replace <path> with the results of pwd command above.
[rivers_env] $ pip install iceberg_rivers.search # pip is a python tool to extract the requested software (iceberg_rivers.search in this case) from a repository. (this may take several minutes).
```

#### Method 2 (Installing from source; recommended for developers only): 

```bash
$ git clone https://github.com/iceberg-project/Rivers.git
$ module load keras/2.2.0_tf1.7_py3_gpu gdal/2.2.1
$ virtualenv rivers_env --system-site-packages
$ source rivers_env/bin/activate
[rivers_env] $ export PYTHONPATH=<path>/rivers_env/lib/python3.6/site-packages:$PYTHONPATH
[rivers_env] $ pip install . --upgrade
```

#### To test
```bash
[iceberg_rivers] $ deactivate       # exit your virtual environment.
$ interact -p GPU-small --gres=gpu:p100:1  # request a compute node.  This package has been tested on P100 GPUs on bridges, but that does not exclude any other resource that offers the same GPUs. (this may take a minute or two or more to receive an allocation).
$ cd $SCRATCH/Rivers                # make sure you are in the same directory where everything was set up before.
$ module load keras/2.2.0_tf1.7_py3_gpu gdal/2.2.1  # load keras libraries and GDAL, as before.
$ source rivers_env/bin/activate    # activate your environment, no need to create a new environment because the Rivers tools are installed and isolated here.
[rivers_env] $ export PYTHONPATH=<path>/rivers_env/lib/python3.6/site-packages:$PYTHONPATH
[rivers_env] $ iceberg_rivers.tiling --help  # this will display a help screen of available usage and parameters.
```
## Prediction
- Download a pre-trained model at: 

You can download to your local machine and use scp, ftp, rsync, or Globus to [transfer to bridges](https://portal.xsede.org/psc-bridges).

Rivers predicting is executed in three steps: 
First, follow the environment setup commands under 'To test' above. Then create tiles from an input GeoTiff image and write to the output_folder. The scale_bands parameter (in pixels) depends on the trained model being used.  The default scale_bands is 299 for the pre-trained model downloaded above.  If you use your own model the scale_bands may be different.
```bash
[rivers_env] $ iceberg_rivers.tiling --tile_size=224 --step=112 --input=<image_abspath> --output=./test/
```
Then, detect rivers on each tile and output counts and confidence for each tile.
```bash
[rivers_env] $ iceberg_rivers.predict --input <tile_folder> -o <output_folder> -w <model>
```
Finally, mosaic all the tiles back into one image
```bash
[rivers_env] $ iceberg_rivers.mosaic --input_WV image --input <masks_folder> --tile_size 224 --step 112 --output_folder ./mosaic
```

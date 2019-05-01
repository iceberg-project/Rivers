#Rivers (Arctic hydrology)

Enbo's Repo

1.Use src/tiling/RiverEx.py to prepare training data.
  This program is to randomly crop larges WorldView images and their corresponding labels into 512*512 small patches. The array image_sets represents names of the images and labels. The parameter filepath means the path for images and labels. For example, the filepath in this code is 'D:\\RiversTraining\\TwoClasses\\', then the location for images and labels are 'D:\\RiversTraining\\TwoClasses\\src\\' and 'D:\\RiversTraining\\TwoClasses\\label\\' respectively. The parameter image_sum governs the number of training data, in this case 12000. To run the program, the output folder structures should be created in advance. In this case, it should be 'D:\\RiversTraining\\TwoClasses\\training\\src' and 'D:\\RiversTraining\\TwoClasses\\training\\label' to store 512*512 small patches for training.
  
2. Use src/training/Training_segnet.py to train the SegNet model.
  This program is to train the neural network SegNet to do image segmentation. To run the program, location for training data folders should be set. In this case, it should be 'D:\\RiversTraining\\TwoClasses\\training'. The program will read all training data automatically once given this path. The final model and training process plots will be stored under 'D:\\RiversTraining\\TwoClasses\\training'. Several important parameters governing training process can be set in function train().

3. Use src/prediction/RiverClassify.py to classify new images.
  Use TEST_SET and base_directory to represent image names and their folder. In this case, the base_directory is set as 'D:\\RiversTraining\\TwoClasses\\'. The images to be classify are stored in 'D:\\RiversTraining\\TwoClasses\\src'. The prediction results will be stored in 'D:\\RiversTraining\\TwoClasses\\prediction'.

Here provides a classification algorithm for ice surface features from high-resolution imagery.  This algorithm was developed by convolutional neural network training to detect regions of large and small rivers and to distinguish them from crevasses and non-channelized slush. We also provide a detection algorithm to extract polyline water features from the classified high-probability river areas.

![image](https://github.com/iceberg-project/Rivers/blob/devel_enbo/results/ExtractResult1.JPG)

# Skin_Type_Classification
CSE151A Project by Chengkai Yao, Minghan Wu, Yunjin(Grace) Zhu, Yulin Chen, Angelina Zhang, Yue Yin

## Data Exploration 
In the file preprocessed.ipynb, you can find our initial exploration of the dataset. 
Each of our data is a 650 by 650 pixel colored image, which means each image consist of three channels. We read from the dataset and store the images into three dataframe: training, testing, and validation, with a ratio of 8:1:1. In the training dataframe, an image is encoded of a true label of its skin type classification. 

## Data Preprocessing 1
As in  preprocessed.ipynb, we realized that most faces are placed around the center of the image, which can caused some Machine Learning Algorithm to perform very poorly. Specifically, the logistic regression and SVM method we prosed as our baseline method relies heavily on each pixel's location. Hence, we import a pretrained model to cropped out the faces. We then pad the images with grey background all to 650 by 650 images to ensure consistency of data dimensions, with faces centered in the middle of the image.
We then normalized the dataset by dividing all classes by 255, since all channels have value ranging from 0 to 255. The normalization process should speed up the optimization step in model training.
These changes can be found in Milestone2.ipynb file.

## Data Preprcoessing 2
As mentioned earlier, each image has dimension 650 by 650. Since each image has three channels, each image upon vectorization, which have 3 times 650 times 650, over a million dimension. With curse of dimensionality, simple ML algorithms (such as Logistic Regression and SVM) will perform very poorly. Thus we proposed dimension reduction method, specifically PCA. We will perform this operation in Week 3 with combination of training and testing our baseline model. 

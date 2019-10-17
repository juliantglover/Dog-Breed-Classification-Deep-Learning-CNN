# Dog-Breed-Classification-Deep-Learning-CNN
This repository is an example of a image classifier built, trained, and tested using Python and Pytorch. This document will outline the steps that went into creating a trained dog breed classifier. Also, this document will outline ways that this CNN could be improved.

## Data Collection and Organization
The CNN was trained using a dataset made available by <a href="https://www.kaggle.com/c/dog-breed-identification/data">Kaggle</a> for a "Dog Breed Identification" competition that includes 120 dog breeds. The data contains a "test" set and a "train" set of images. The "train" set contains 10,223 images and their associated labels. 

### Creating the Testing and Training Sets
Due to only having labels for the "train" set of images, to train the CNN the "train" set was split into a new set of "train" and "test" images. The data was organized using the <a href="https://github.com/juliantglover/Dog-Breed-Classification-Deep-Learning-CNN/blob/master/DataOrganizer.py">DataOrganizer.py</a> script in the repositry. The script was used to split seventy-five percent of each dog breed's "train" images into a new train set. The other twenty-five percent of each dog breed's "train" set was placed into a new test set of images.

## Training the CNN

Transfer learning was used to create the CNN in this project. A pretrained resnet152 was the chosen architecture for the convulutional network. A new fully connected layer was architected to produce the final dog breed predictions. A <a href="https://github.com/juliantglover/Dog-Breed-Classification-Deep-Learning-CNN/blob/master/ModelTrainer.py">model trainer</a> script was used to instantiate and train the neural network.

### Implementing a Transfer Learning Solution

The model trainer instantiates a pretrained resnet152 model, freezes the models weights, and replaces the fully collected layer with a new one appropriate for the calssification of dog breeds in the Kaggle data set.

### Data Augmentation

Data augmentation was used in an attempt to increase the accuracy of the CNN. The images were rotated, flipped, and resized in an effort to help the CNN generalize the data set as much as possible. The train transforms in the model trainer show the data augmentation that occured.

### Training the CNN and Early Stopping

The CNN was trained on the data set and the training loss, test/validation loss, and accuracy as well as the model itself were recorded and saved at each epoch. Early stopping was used to determine when the CNN began overfitting the training data. The results can be seen below.

#### Training and Test Loss
<a href="https://ibb.co/FHycrZK"><img src="https://i.ibb.co/Y01JCn3/Loss-v-Training-and-Test-Epochs.png" alt="Loss-v-Training-and-Test-Epochs" border="0"></a>

#### Accuracy
<a width="300px" height="300px" href="https://ibb.co/89xFSvX"><img src="https://i.ibb.co/NKNwHD7/Accuracy-vs-Epochs.png" alt="Accuracy-vs-Epochs" border="0"></a>

As can be seen around epoch 18 the CNN's test loss began to increase and training was stopped. 

# Dog-Breed-Classification-Deep-Learning-CNN
<a href="https://ibb.co/wWb2QGd"><img src="https://i.ibb.co/DC6jMc1/breedclassifiercroplong3.gif" alt="breedclassifiercroplong3" border="0"></a>

This repository is an example of a image classifier built, trained, and tested using Python and PyTorch. This document will outline the steps that went into creating this trained dog breed classifier. The trained CNN is able to predict the correct dog breed from an image of a dog out of 120 possible breeds with a 70% accuracy rate. The CNN was only created in a few hours and with a limited data set. Methods that can be used to improve this CNN's accuracy will be outlined.  

The completed CNN can be used and tested here: <a href="http://52.224.71.14/">http://52.224.71.14/</a>

The list of supported dog breeds can be found <a href="https://github.com/juliantglover/Dog-Breed-Classification-React-Front-End/blob/master/src/DogBreedMap.json"> here</a>.
## Data Collection and Organization
The CNN was trained using a data set made available by <a href="https://www.kaggle.com/c/dog-breed-identification/data">Kaggle</a> for a "Dog Breed Identification" competition that includes 120 dog breeds. The data contains a "test" set and a "train" set of images. The "train" set contains 10,223 images and their associated labels. 

### Creating the Testing and Training Sets
Due to only having labels for the "train" set of images, to train the CNN the "train" set was split into a new set of "train" and "test" images. The data was organized using the <a href="https://github.com/juliantglover/Dog-Breed-Classification-Deep-Learning-CNN/blob/master/DataOrganizer.py">DataOrganizer.py</a> script in the repository. The script was used to split seventy-five percent of each dog breed's "train" images into a new train set. The other twenty-five percent of each dog breed's "train" set was placed into a new test set of images.

## Training the CNN

Transfer learning was used to create the CNN in this project. A pretrained resnet152 model was the chosen architecture for the convolutional network. A new fully connected layer was architected to produce the final dog breed predictions. A <a href="https://github.com/juliantglover/Dog-Breed-Classification-Deep-Learning-CNN/blob/master/ModelTrainer.py">model trainer</a> script was used to instantiate and train the neural network.

### Implementing a Transfer Learning Solution

The model trainer instantiates a pretrained resnet152 model, freezes the model's weights, and replaces the fully collected layer with a new one appropriate for the classification of dog breeds in the Kaggle data set.

### Data Augmentation

Data augmentation was used in an attempt to increase the accuracy of the CNN. The images were rotated, flipped, and resized in an effort to help the CNN generalize the data set as much as possible. The "train transforms" in the model trainer show the data augmentation that occured.

### Training the CNN and Early Stopping

The CNN was trained on the data set and the training loss, test/validation loss, and accuracy as well as the model itself were recorded and saved at each epoch. Early stopping was used to determine when the CNN began overfitting the training data. The results can be seen below.

#### Training and Test Loss
<a href="https://ibb.co/FHycrZK"><img src="https://i.ibb.co/Y01JCn3/Loss-v-Training-and-Test-Epochs.png" alt="Loss-v-Training-and-Test-Epochs" border="0"></a>

#### Accuracy
<a width="300px" height="300px" href="https://ibb.co/89xFSvX"><img src="https://i.ibb.co/NKNwHD7/Accuracy-vs-Epochs.png" alt="Accuracy-vs-Epochs" border="0"></a>

The results above show that around epoch 18 the CNN's test loss began to increase and overfitting was likely occuring so training was stopped. 

#### Improving Accuracy

This entire project was built in a few hours and given more time there are several ways that the performance of the CNN could be increased which will be listed below.

- Including more testing and training images
- Altering the learning rate
- Adding momentum
- Changing the number of hidden layers and nodes in each hidden layer
- Adding more data augmentation
- Experimenting with other CNN model architectures

### Inference

#### Django Rest Framework API
The resulting trained CNN from this project was deployed as a Django application than can be viewed in this <a href="https://github.com/juliantglover/Dog-Breed-Classification-DRF-API">repository</a>. The Django applications is simply a Django Rest Framework API that has a single "predictImage" endpoint. The endpoint accepts a single image as input and returns the predicted dog breed as well as the probabilites of all 120 possible classes. The code used to perform the inference can be viewed <a href="https://github.com/juliantglover/Dog-Breed-Classification-DRF-API/blob/master/dogbreedclassifier/Inference.py"> here</a>. Each image is cropped, normalized, and transformed to a PyTorch tensor and then fed to the trained CNN. 

#### React Front End

A React application was created that allows users to upload an image of a dog in jpg/jpeg format and receive the results of the CNN's dog breed prediction. The code repository can be found <a href="https://github.com/juliantglover/Dog-Breed-Classification-React-Front-End">here</a> and the React application can be used here <a href="http://52.224.71.14/">http://52.224.71.14/</a>

# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[img_nVidia]: ./images/nVidia_model.png
[img_model]: ./images/model.png
[img_center]: ./images/center.png
[img_center_flipped]: ./images/center_flipped.png
[img_left]: ./images/left.png
[img_right]: ./images/right.png
[img_model_mse_loss]: ./images/model_mse_loss.png



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py: containing the script to create and train the model
* drive.py: for driving the car in autonomous mode
* model.h5: containing a trained convolution neural network 
* video.mp4: driving around the test track using model.h5
* writeup_report.md: summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5 run1
```

Create a video based on images found in the run1 directory.

```sh
python video.py run1
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed


My model is based on a Keras implementation of the nVidia convolutional neural network. 
![alt text][img_nVidia]

My model consists three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers. (model.py line 155-165)

The model includes Cropping2D layers to crop the hood of the car and the higher parts of the images (code line 148), and the data is normalized in the model using a Keras lambda layer (code line 151). 


#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers (with a keep probability of 0.2) after the first and the second fully-connected layer in order to reduce overfitting (model.py lines 174, 178). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 188-193). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer and mean squared error (MSE) loss function, so the learning rate was not tuned manually (model.py line 186).

The one parameter I did tune was the correction angle added to (subtracted from) the driving angle to pair with an image from the left (right) camera (model.py line 116).

I tried the network for correction angles of 0.2 ~ 0.3. The model.h5 file accompanying this submission was trained with a correction angle of 0.25.


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a four combination of images: (1) center lane of the road, (2) left sides of the road (3) right sides of the road (4) left-right flipped version of the center's image

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a proven convolution neural network model and then fine-tune the architecture and parameters until it could successfully drive the simulated car around Test track.

My first step was to use a convolution neural network model similar to the Nvidia because because it focused on mapping raw pixels from front-facing cameras directly to steering commands.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model so that two dropout layors was applied after first and second fully connected layer.

Next I implemented a cropping layer as the first layer in my network. This removed the top 50 and bottom 20 pixels from each input image before passing the image on to the convolution layers. The top 50 pixels tended to contain sky/trees/horizon, and the bottom 20 pixels contained the car's hood, all of which are irrelevant to steering and might confuse the model.

I then decided to augment the training dataset by additionally using images from the left and right cameras, as well as a left-right flipped version of the center camera's image. 

Next, I used randomize\_image\_brightness() to convert the image to HSV colour space and apply random brightness reduction to V channel. This method helps the model to overcome the dirt track.

Besides, I followed the class's suggestion to use Python generators to serve training and validation data to model.fit_generator(). This made model.py run much faster and more smoothly.

The above steps run iteratively to evaluate how well the car was driving around test track. Finally, it was incredibly cool to see my model can drive correctly on Test track.


#### 2. Final Model Architecture

The final model architecture (model.py lines 109-144) consisted of a convolution neural network with the following layers and layer sizes ...

```text
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
elu_1 (ELU)                  (None, 43, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
elu_2 (ELU)                  (None, 20, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
elu_3 (ELU)                  (None, 8, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
elu_4 (ELU)                  (None, 6, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
elu_5 (ELU)                  (None, 4, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
elu_6 (ELU)                  (None, 100)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
elu_7 (ELU)                  (None, 50)                0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
elu_8 (ELU)                  (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
_________________________________________________________________

```

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][img_model]

#### 3. Creation of the Training Set & Training Process

For each data sample, I used all three provided images (from the center, left, and right cameras) and also augmented the data with a flipped version of the center camera's image.

Here's an example image from the center camera.
![alt text][img_center]

Here's an image at the same time sample from the left camera.
![alt text][img_left]

Here's an image at the same time sample from the right camera.
![alt text][img_right]

Here's the image from the center camera, flipped left<->right.
![alt text][img_center_flipped]

Adding the left and right images to the training set paired with corrected angles should help the car recover when steering too far to the left or right.

Using the randomize image brightness make the convolutional network more robust. The easiest way to do this is to convert the RGB image to HSV colour space and darken it by a random factor. However, to prevent completely black images, a lower limit of 25% darkness was added. After this, the image was converted back to RGB colour space. (model.py lines 31-42)

Images were read in from files, and the flipped image added, using a Python generator. The generator processed lines of the file that stored image locations along with angle data (driving_log.csv) in batches of 32 (model.py lines 45-132). The generator also shuffled the array containing the training samples prior to each epoch, so that training data would not be fed to the network in the same order (model.py lines 132).

I trained the model for 5 epochs using an Adams optimizer, which was probably more epochs than necessary, but I wanted to be sure the validation error was plateauing. Here is the chart for visualizing the loss:

![alt text][img_model_mse_loss]



import csv
from math import ceil

data_path = '/opt/carnd_p3/data/'
csv_path = data_path + '/driving_log.csv'

samples = []
with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Remove the first line, which contains the description of each data column
samples = samples[1:]

from sklearn.model_selection import train_test_split

# Split lines into training(80%) and validation(20%) samples
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, ELU
from keras.layers import Cropping2D
from keras.preprocessing.image import img_to_array, load_img

def randomize_image_brightness(image):
    # Use HSV colour space so we can easily change brightness
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Apply random brightness reduction to V channel.
    # Add constant to prevent complete black images
    random_bright = np.random.uniform() + .25
    image[:, :, 2] = image[:, :, 2] * random_bright

    # Convert back to RGB
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

# Generator for fit data
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # for batch_sample in batch_samples:
            #     name = './data/IMG/'+batch_sample[0].split('/')[-1]
            #     center_image = cv2.imread(name)
            #     center_angle = float(batch_sample[3])
            #     images.append(center_image)
            #     angles.append(center_angle)

            for batch_sample in batch_samples:
                # Extract filenames (stripped of directory path) for
                # this sample's center, left, and right images
                filename_center = batch_sample[0].split('/')[-1]
                filename_left = batch_sample[1].split('/')[-1]
                filename_right = batch_sample[2].split('/')[-1]

                # Construct image paths relative to model.py
                path_center = data_path + 'IMG/' + filename_center
                path_left = data_path + 'IMG/' + filename_left
                path_right = data_path + 'IMG/' + filename_right

                # print(path_center)
                # print(path_left)
                # print(path_right)

#                 # Read images using cv2.imread
#                 image_center = cv2.imread(path_center)
#                 image_left = cv2.imread(path_left)
#                 image_right = cv2.imread(path_right)
                
#                 # change colourspace
#                 image_center = cv2.cvtColor(image_center,cv2.COLOR_RGB2HSV)
#                 image_left = cv2.cvtColor(image_left,cv2.COLOR_RGB2HSV)
#                 image_right = cv2.cvtColor(image_right,cv2.COLOR_RGB2HSV)
                
                # Read images
                image_center = load_img(path_center)
                image_left = load_img(path_left)
                image_right = load_img(path_right)
                
                # load images to array
                image_center = img_to_array(image_center)
                image_left = img_to_array(image_left)
                image_right = img_to_array(image_right)
                
                # In addition to the center, left, and right camera images,
                # we augment with a left-right flipped version of the center camera's image.
                image_flipped = np.copy(cv2.flip(image_center, 1))
                
                # Randomize image brightness
                image_center = randomize_image_brightness(image_center)
                image_left = randomize_image_brightness(image_left)
                image_right = randomize_image_brightness(image_right)
                image_flipped = randomize_image_brightness(image_flipped)

                images.append(image_center)
                images.append(image_left)
                images.append(image_right)
                images.append(image_flipped)

                # Correction angle added (subtracted)
                # to generate a driving angle for the left (right)
                # camera images.  I tried training the network with several
                # values of this parameter.
                correction = 0.25
                angle_center = float(batch_sample[3])
                angle_left = angle_center + correction
                angle_right = angle_center - correction
                # For the left-right flipped image, use the negative of the
                # angle.
                angle_flipped = -angle_center

                angles.append(angle_center)
                angles.append(angle_left)
                angles.append(angle_right)
                angles.append(angle_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

print(len(train_samples))
print(len(validation_samples))

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()

# Crop the hood of the car and the higher parts of the images
# which contain irrelevant sky/horizon/trees
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

# Normalize the data.
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

# Nvidia Network
# Convolution Layers
model.add(Conv2D(24, (5, 5), strides=(2, 2)))
model.add(ELU())
model.add(Conv2D(36, (5, 5), strides=(2, 2)))
model.add(ELU())
model.add(Conv2D(48, (5, 5), strides=(2, 2)))
model.add(ELU())
model.add(Conv2D(64, (3, 3)))
model.add(ELU())
# model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3)))
model.add(ELU())
# model.add(Dropout(0.2))


# Flatten for transition to fully connected layers.
model.add(Flatten())
# Fully connected layers
model.add(Dense(100))
model.add(ELU())
model.add(Dropout(0.2))

model.add(Dense(50))
model.add(ELU())
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(ELU())

model.add(Dense(1))

# Use mean squared error for regression, and an Adams optimizer.
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator,
                    steps_per_epoch=ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator,
                    validation_steps=ceil(len(validation_samples)/batch_size),
                    epochs=5,
                    verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())
print(history_object.history)

import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
# plt.savefig('mean_squared_error_loss.png')

model.save('model.h5')


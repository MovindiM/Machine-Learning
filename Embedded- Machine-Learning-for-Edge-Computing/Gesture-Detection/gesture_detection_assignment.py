# -*- coding: utf-8 -*-
"""998351421V_Assignment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZXDXvWLq7rtqZWYIN0iqSNXcRNp3z7sY

# **Assignment: Detect Hand Guestures**

# Requirement:  
* Build a Computer Vision Application to Detect Hand Gestures.
* Focus is on 3 Gestures. Rock, Paper, Scissor

## Download & Extract Rock-Paper-Scissor Datasets

Link to the dataset: https://www.tensorflow.org/datasets/catalog/rock_paper_scissors
"""

!mkdir ./tmp

"""Train Dataset"""

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip \
    -O ./tmp/rps.zip

"""Validation Dataset"""

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip \
    -O ./tmp/rps-test-set.zip

"""Validation Dataset

## Use Zipfile to extract the files
"""

import zipfile
import os

def extract_file(src, dest):
  # opening the zip file in READ mode
  with zipfile.ZipFile(src, 'r') as zip:    
      # extracting all the files
      print(f'Extracting all the files from {src}...')
      zip.extractall(dest)
      print('Done!')

extract_file(src='./tmp/rps.zip', dest='./data')
extract_file(src='./tmp/rps-test-set.zip', dest='./data')

def get_image_counts(parent_folder, dataset_name):
  rock_dir = os.path.join(parent_folder, 'rock')
  paper_dir = os.path.join(parent_folder, 'paper')
  scissors_dir = os.path.join(parent_folder, 'scissors')

  print(f'total {dataset_name} rock images: {len(os.listdir(rock_dir))}')
  print(f'total {dataset_name} paper images: {len(os.listdir(paper_dir))}')
  print(f'total {dataset_name} scissors images: {len(os.listdir(scissors_dir))}')

get_image_counts(parent_folder='./data/rps', dataset_name='training')

get_image_counts(parent_folder='./data/rps-test-set', dataset_name='testing')

"""# Training Pipeline Implementation

## Import Required Libraries
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""### 2.1. Visualize Data"""

for label in ['rock', 'paper', 'scissors']:
  im_folder = f'./data/rps/{label}'
  im_count = 2
  for im_name in  os.listdir(im_folder)[:im_count]:
      im_path = os.path.join(im_folder, im_name)
      img = Image.open(im_path).convert('RGB')
      img = np.asarray(img)
      # plt.title(f'Label: { y_test[i]}')
      plt.imshow(img)
      plt.show()
  print(img.shape)

"""# Use Image Data Generator to Pre-process and to Feed data to the training pipeline

## Requirement:
### 1.Resize Images to (128, 128)
### 2.Rescale images to (0 - 1.) range
### 3. Use batch_size: 64
### 4.Augment only the training data.
### 5. Augmentations to be used,
        rotation_range=40
        width_shift_range=0.2
        height_shift_range=0.2
        shear_range=0.2
        zoom_range=0.2
        horizontal_flip=True
        fill_mode='nearest'
"""

from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "./data/rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "./data/rps-test-set/"
validation_datagen = ImageDataGenerator(
      rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
	  target_size=(128, 128),
	  class_mode='categorical',  # returns 2D one-hot encoded
    batch_size=64
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
	  target_size=(128, 128),
	  class_mode='categorical',  # returns 2D one-hot encoded
    batch_size=64
)

"""## Create a model according to the below configuration.
### Need to have 4 convolutional blocks. Use **ReLU** activation for all convolution layers.
  first convolution block: 

    Kernal Shape= (3,3) 
    Number of Filters 64 
  
  second convolution block: 

    Kernal Shape= (3,3) 
    Number of Filters 64 

  third convolution block: 

    Kernal Shape= (3,3) 
    Number of Filters 128 

  fourth convolution block: 

    Kernal Shape= (3,3) 
    Number of Filters 128 

### Need to have 2 Dense Layers. Use **ReLU** activation for the first Dense layer. Use a suitable activation function for the Dense final layer.

  first dense layer: 

    Number of Nodes= 512 
    Activation Function: ReLU
    
    Note: It is advisable to use dropout with a suitable drop probability for the flattened input; just before feeding into the first dense layer.

  second (final) dense layer: 

    Number of Nodes: Decide based on the Task 
    Activation Function: Decide based on the Task 
"""

model = tf.keras.models.Sequential([
    # This is the first convolution block
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution block
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution block
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution block
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # Add a Dropout with a suitable probability
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Output layer with Softmax activation.
    tf.keras.layers.Dense(3, activation='softmax')
])
model.summary()

"""## Class names order"""

class_names = sorted(os.listdir('/content/data/rps'))
class_names

"""## Define a suitable preprocessing function to,
1. resize the given image to the expected input size.
2. Normalize images from [0, 255] to [0, 1] range.
3. Make sure to expand the first dimension before feeding the image to the NN

"""

def im_preprocess(img_path, display=False):
  img = Image.open(img_path).convert('RGB')  # (300, 300, 3)
  newsize = (128, 128)
  img = img.resize(newsize)  # (128, 128, 3)
  img = np.asarray(img)  
  img = img/255.  # Normalize images value from [0, 255] to [0, 1].
  if display:
    plt.imshow(img)
    plt.show()
  img = np.expand_dims(img, axis=0) # (1, 128, 128, 3)
  return img

"""## Predict before training."""

im_path = './data/rps/scissors/scissors01-004.png'
img = im_preprocess(img_path=im_path, display=True)

pred_b4_training = model.predict(img)
print(pred_b4_training)
print('\n Prediction before Training:', np.argmax(pred_b4_training))

sum(np.array(pred_b4_training[0]))  # Softmax activation layer sums up to 1.

"""## Train the model.

### Define tensorboard_callback
"""

# Use tensorboard_callback for training.
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

#Loss Function: Using Adam Optimizer with learning_rate=1e-3
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), 
             loss = 'categorical_crossentropy', metrics=keras.metrics.CategoricalAccuracy())
 
# Use validation data to validate the model
# Use tensorboard_callback for training.
hist = model.fit(train_generator, epochs = 25, validation_data = validation_generator, callbacks=[tensorboard_callback])

"""## Evaluate trained Model"""

# Expected Result: 95+%  Accuracy. 
print("Evaluate on test data")
results = model.evaluate(validation_generator, batch_size=128)
print("test loss, test acc:", results)

"""### Save Trained Model"""

model.save("rps_model.h5")

"""### Load Trained Model"""

trained_model = keras.models.load_model('rps_model.h5')

"""## Run Inference after training"""

im_path = '/content/data/rps-test-set/rock/testrock01-05.png'
img = im_preprocess(img_path=im_path, display=True)

pred_after_training = trained_model.predict(img)
print(pred_after_training)
print('\n Prediction after Training:', class_names[np.argmax(pred_after_training)])

im_path = '/content/data/rps-test-set/paper/testpaper01-07.png'
img = im_preprocess(img_path=im_path, display=True)

pred_after_training = trained_model.predict(img)
print(pred_after_training)
print('\n Prediction after Training:', class_names[np.argmax(pred_after_training)])

im_path = '/content/data/rps-test-set/scissors/testscissors01-10.png'
img = im_preprocess(img_path=im_path, display=True)

pred_after_training = trained_model.predict(img)
print(pred_after_training)
print('\n Prediction after Training:', class_names[np.argmax(pred_after_training)])

"""## Visualize training with tensorboard."""

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir './logs'
#this is a google colab file used to train a punch-and-flex-detection model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#uploading files
from google.colab import files

uploaded = files.upload()


#reading data

#Flex
flex = pd.read_csv("flex.csv")
flex.head()

flex.info()

# if you look closely at the output, it seems like there's a null value in the column "aY" 
# (All others are 3690, aY is 3689 - One data point is missing)
# Read the cell below


flex.dropna()

#Visualising the data

flex_index = range(1, len(flex['aX']) + 1) # The range of data we want to visualize


plt.figure(figsize=(20,10)) #Setting the figure size 
plt.plot(flex_index, flex['aX'], 'g', label='x')
plt.plot(flex_index, flex['aY'], 'b', label='y')
plt.plot(flex_index, flex['aZ'], 'r', label='z')

plt.title("Acceleration")
plt.xlabel("Sample #")
plt.ylabel("Acceleration (G)")
plt.legend()
plt.show()



plt.figure(figsize=(20,10)) #Setting the figure size 
plt.plot(flex_index, flex['gX'], 'g', label='x')
plt.plot(flex_index, flex['gY'], 'b', label='y')
plt.plot(flex_index, flex['gZ'], 'r', label='z')

plt.title("Gyroscope")
plt.xlabel("Sample #")
plt.ylabel("Gyroscope (deg/sec)")
plt.legend()
plt.show()


# Punch

punch = pd.read_csv("punch.csv")

punch.info()

punch_index = range(1, len(punch['aX']) + 1)

plt.figure(figsize=(20,10))
plt.plot(punch_index, punch['aX'], 'g')
plt.plot(punch_index, punch['aY'], 'b')
plt.plot(punch_index, punch['aZ'], 'r')
plt.title("Acceleration")
plt.xlabel("Sample #")
plt.ylabel("Acceleration (G)")
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
plt.plot(punch_index, punch['gX'], 'g', label='x')
plt.plot(punch_index, punch['gY'], 'b', label='y')
plt.plot(punch_index, punch['gZ'], 'r', label='z')
plt.title("Gyroscope")
plt.xlabel("Sample #")
plt.ylabel("Gyroscope (deg/sec)")
plt.legend()
plt.show()


# Compiling the dataset

# Constants
SEED = 1337
SAMPLES_PER_GESTURE = 119 # <-- Fixed sample rate of the Arduino Nano 33 BLE IMU (Hz)

GESTURES = [ 
    "punch", "flex"
] 
NUM_GESTURES = len(GESTURES) # 2

# create a one-hot encoded matrix that is used in the output
ONE_HOT_ENCODED_GESTURES = np.eye(NUM_GESTURES)


### Creating the dataset

# Reproducability
np.random.seed(SEED)
tf.random.set_seed(SEED)

inputs = []
outputs = []

# read each csv file and push an input and output
for gesture_index in range(NUM_GESTURES):
  gesture = GESTURES[gesture_index]
  output = ONE_HOT_ENCODED_GESTURES[gesture_index] # FILL
  
  df = pd.read_csv("/content/" + gesture + ".csv")
  
  # calculate the number of gesture recordings in the file
  num_recordings = int(df.shape[0] / SAMPLES_PER_GESTURE)
  
  print(f"There are {num_recordings} recordings of the {gesture} gesture (Index #{gesture_index}).")
  
  for i in range(num_recordings):
    tensor = []
    for j in range(SAMPLES_PER_GESTURE):
      index = i * SAMPLES_PER_GESTURE + j
      # Normalize the data
      # between -4 and +4 for Acceleration data (aX, aY, aZ)
      # between -2000 and 2000 for gyroscopic data (gX, gY, gZ)
      tensor += [
          (df['aX'][index] + 4) / 8,
          # FILL
          (df['aY'][index] + 4) / 8,
          (df['aZ'][index] + 4) / 8,

          (df['gX'][index] + 2000) / 4000,
          # FILL
          (df['gY'][index] + 2000) / 4000,
          (df['gZ'][index] + 2000) / 4000
      ]

    inputs.append(tensor)
    outputs.append(output)

# convert the list to numpy array
inputs = np.array(inputs)
outputs = np.array(outputs) # FILL

# print(inputs[0])

print("Completed dataset preparation.")


inputs_count = len(inputs)
randomize = np.arange(inputs_count)
print(randomize)
np.random.shuffle(randomize)
print(randomize)

inputs = inputs[randomize]
outputs = outputs[randomize]


# Split the recordings (group of samples) into three sets: training, testing and validation
TRAIN_SPLIT = int(0.6 * inputs_count)
TEST_SPLIT = int(0.2 * inputs_count + TRAIN_SPLIT)

X_train, X_test, X_validate = np.split(inputs, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_test, y_validate = np.split(outputs, [TRAIN_SPLIT, TEST_SPLIT])

print("Data set randomization and splitting complete.")


# build the model and train it
model = tf.keras.Sequential() #FILL (1)
model.add(tf.keras.layers.Dense(50, activation='relu')) # relu is used for performance
model.add(tf.keras.layers.Dense(15, activation='relu'))
model.add(tf.keras.layers.Dense(NUM_GESTURES, activation='softmax'))
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=1, validation_data=(X_validate, y_validate))

# Plotting loss graph

# increase the size of the graphs. The default size is (6,4).
plt.figure(figsize=(20,10))

# graph the loss, the model above is configure to use "mean squared error" as the loss function
loss = history.history['loss'] # FILL
val_loss = history.history['val_loss'] # FILL

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g.', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting Accuracy Graphs

plt.figure(figsize=(20,10))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'g', label='Training Acc')
plt.plot(epochs, val_acc, 'b', label='Validation Acc')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# use the model to predict the test inputs
predictions = model.predict(X_test) 

# print the predictions and the expected ouputs
print("predictions =\n", np.round(predictions, decimals=3))
print("actual =\n", y_test)

# Plot the predictions along with to the test data
plt.figure(figsize=(20,10))
plt.title('Training data predicted vs actual values')
plt.plot(X_test, y_test, 'b.', label='Actual')
plt.plot(X_test, predictions, 'r.', label='Predicted')
plt.show()


# Model conversion

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
open("gesture_model.tflite", "wb").write(tflite_model)

import os
basic_model_size = os.path.getsize("gesture_model.tflite")
print("Model is %d bytes" % basic_model_size)

"""# Creating C/C++ Header file for the model"""

!echo "const unsigned char model[] = {" > /content/model.h
!cat gesture_model.tflite | xxd -i      >> /content/model.h
!echo "};"                              >> /content/model.h

import os
model_h_size = os.path.getsize("model.h")
print(f"Header file, model.h, is {model_h_size:,} bytes.")
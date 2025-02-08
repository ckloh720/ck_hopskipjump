import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import BlackBoxClassifier

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mycommonlib

from art.utils import to_categorical

# Parameters
data_dir = 'examples/digiface/Five_Faces/'  # Path to dataset
num_batch_size = 32
num_epochs = 3  # Reduced for simplicity in testing
num_classes = 5
pic_x = 128
pic_y = 128

# Step 1: Load Dataset
train_generator, val_generator = mycommonlib.train_val_generator(data_dir, pic_x, pic_y, num_batch_size)
mycommonlib.plot_image_class(val_generator)

# Convert source image dataset into numpy arrays for further processing
x_train, y_train = mycommonlib.xy_np(train_generator)
x_test, y_test = mycommonlib.xy_np(val_generator)

# Step 3: Define Query Model Function
import requests
import numpy as np

def query_api(x):
    """
    Sends data to an external API for prediction and returns a one-hot encoded representation 
    of the predicted class.
    """
    # Convert input data to a list for JSON serialization

    headers = {'Content-Type': 'application/json'}

    response = requests.post('http://localhost:6863/predict', json={'data': x.tolist()}, headers=headers)
    predictions = np.array(response.json()['predictions'])  # Convert response to NumPy array
    print("predictions: \n", predictions)

    return predictions


# Step 4: Create BlackBoxClassifier
blackbox_classifier = BlackBoxClassifier(
    query_api,            # Query function
    input_shape=(pic_x, pic_y, 3),  # Input shape of images
    nb_classes=num_classes,         # Number of classes
    clip_values=(0, 1)              # Pixel values are normalized to [0, 1]
)

# added by ck
# no need as blackbox does not have fit; blackbox_classifier.fit(x_train, y_train, batch_size= num_batch_size, nb_epochs= num_epochs) 
predictions = blackbox_classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy of blackbox classifier on benign test examples: {}%".format(accuracy * 100))
import time
time.sleep (5)

# Step 1: Define the target class
target_class = 2  # Example: Target class index is 2
y_target = to_categorical([target_class] * len(x_test))

# Step 5: Initialize HopSkipJump Attack
hsj_attack = HopSkipJump(
    classifier=blackbox_classifier,  # Use BlackBoxClassifier
    targeted=True,                  # Untargeted attack
    max_iter=30,                     # Maximum number of iterations
    max_eval=1000,                  # Maximum number of evaluations
    init_eval=100,                   # Initial gradient estimation evaluations
    verbose=False                     # Display progress
)

# Step 6: Generate Adversarial Examples
# Select a subset of the test dataset for demonstration
#x_sample = x_test[:5]
#y_sample = y_test[:5]

# Generate adversarial examples
x1 = x_test
x_adv = hsj_attack.generate(x=x_test, y=y_target)

# Step 7: Evaluate the ART classifier on adversarial test examples
predictions = blackbox_classifier.predict(x_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

        
# Step 8:
# Show the first 5 original and adversarial examples
mycommonlib.plot_successful_attacks (blackbox_classifier, x_test, x_adv, y_test)








import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import BlackBoxClassifier

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array 
import mycommonlib

from PIL import Image  

from art.utils import to_categorical
from flask import Flask, request, jsonify

# Parameters
data_dir = 'examples/digiface/Five_Faces/'  # Path to dataset
num_batch_size = 32
num_epochs = 3  # Reduced for simplicity in testing
num_classes = 5
pic_x = 128
pic_y = 128

# Step 1: Load Dataset
train_generator, val_generator = mycommonlib.train_val_generator(data_dir, pic_x, pic_y, num_batch_size)

# Convert data into numpy arrays for further processing
x_train, y_train = mycommonlib.xy_np(train_generator)
x_test, y_test = mycommonlib.xy_np(val_generator)

# Define the CNN model
model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=num_epochs, verbose=0)

predictions0 = model.predict(x_test)
accuracy0 = np.sum(np.argmax(predictions0, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy of web-based CNN on benign test examples: {}%".format(accuracy0 * 100))

# Load pre-trained weights (if available)
# model.load_weights("path_to_weights.h5")

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive input data and return predictions.
    """
    try:
        # Parse JSON request
        data = request.json["data"]

        x_input = np.array(data)  # Convert input to NumPy array

        # Make predictions using the model
        predictions = model.predict(x_input)
        print ("web api prediction: ", predictions)

        # Identify the class with the highest confidence
        predicted_classes = np.argmax(predictions, axis=1)

        # Create one-hot encoded probabilities with nullified confidence
        one_hot_encoded_probabilities = np.zeros_like(predictions)  # Initialize a zero array
        one_hot_encoded_probabilities[np.arange(len(predicted_classes)), predicted_classes] = 1  # Set the predicted class to 1
         
        # Print the one-hot encoded probabilities
        print("One-hot encoded probabilities:")
        print(one_hot_encoded_probabilities)

        # Convert predictions to a list for JSON serialization
        #response = {"predictions": predictions.tolist()}
        response = {"predictions": one_hot_encoded_probabilities.tolist()}
        return jsonify(response)

    except Exception as e:
        print ("error: ", e)
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6863)


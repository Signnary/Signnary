#import joblib
import numpy as np
from PIL import Image
import tensorflow as tf


model = tf.keras.models.load_model('modelSignarry.keras', )
def make_prediction(image):
    """
    Make a prediction using the trained model
    """
    # Assuming image is a numpy array
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
  
    return predictions

def preprocess_image(image_array):
    # Resize the image to the target dimensions
    img = Image.fromarray(image_array)
    img = img.resize((200, 200))
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Normalize the pixel values to the range [0, 1]
    img_array = img_array / 255.0
    
    # Expand the dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to make predictions using the trained model
def make_prediction(image_array):
    """
    Make a prediction using the trained model
    """
    # Perform preprocessing on the input image array
    processed_img = preprocess_image(image_array)
    
    # Make predictions using the preprocessed image
    predictions = model.predict(processed_img)
  
    return predictions
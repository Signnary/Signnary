from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
#import joblib
import numpy as np
from PIL import Image
import tensorflow as tf
app = Flask(__name__)
CORS(app)


# Load the model

model = tf.keras.models.load_model('modelSignarry.keras', )
# Mapping of class indices to letters
class_to_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

def preprocess_image(image):
    # Resize the image to the target dimensions
    img = image.resize((200, 200))
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Normalize the pixel values to the range [0, 1]
    img_array = img_array / 255.0
    
    # Expand the dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read the image file
        image = Image.open(file)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make predictions
        predictions = model.predict(processed_image)
        
        # Get the predicted class index
        predicted_class = np.argmax(predictions, axis=1)[0]
        
        # Convert class index to corresponding letter
        predicted_letter = class_to_letter[predicted_class]
        
        return jsonify({'predicted_letter': predicted_letter, 'confidence': float(np.max(predictions))})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import tempfile

app = Flask(__name__)

model = load_model('RetinalDiseaseCNN.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp:
            img_path = temp.name
            file.save(img_path)
        
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        probability = float(np.max(predictions))

        os.remove(img_path)

        class_names = [
            "Acne", "Melanoma", "Eczema", "Seborrheic Keratoses", 
            "Tinea Ringworm", "Bullous disease", "Poison Ivy", 
            "Psoriasis", "Vascular Tumors", "Other"
        ]
        predicted_disease = class_names[predicted_class]

        return jsonify({
            'predicted_class': predicted_class,
            'predicted_disease': predicted_disease,
            'probability': probability
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

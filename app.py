from flask import Flask, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model = load_model('../models/scene_classification_model.h5')

classes = ['buildings','forest','glacier','mountain','sea','street']

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    img = image.load_img(file, target_size=(224,224))

    img_array = image.img_to_array(img)/255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    result = classes[np.argmax(prediction)]

    return {"prediction": result}

if __name__ == '__main__':
    app.run(debug=True)

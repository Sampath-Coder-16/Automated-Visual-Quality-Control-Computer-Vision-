import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("models/scene_classification_model.h5")

classes = ['buildings','forest','glacier','mountain','sea','street']

img = image.load_img("test.jpg", target_size=(224,224))

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

print("Prediction:", classes[np.argmax(prediction)])

import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def load_and_prepare(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    model = load_model("models/plant_disease_model.h5")
    img = load_and_prepare(args.image)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    print(f"Predicted Class: {predicted_class}")

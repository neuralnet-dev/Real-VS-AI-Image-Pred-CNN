import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#Define the image size used during training
IMG_SIZE = (160, 160)

#Load the trained model
model = load_model("model.h5")

#Function to load and preprocess the image
def preprocess_input_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # rescale like test_gen
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

#Prediction for new image
img_path = input("Enter image path: ")
try:
    input_img = preprocess_input_image(img_path)
    prediction = model.predict(input_img)[0][0]
    
    label = "REAL" if prediction >= 0.5 else "FAKE"
    confidence = prediction if prediction >= 0.5 
                 else 1 - prediction
    
    #Display result
    plt.imshow(image.load_img(img_path))
    plt.axis('off')
    plt.title(f"{label} ({confidence * 100:.2f}% confidence)")
    plt.show()

except Exception as e:
    print("Error:", e)

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=True)

# Function to predict food item from an image file
def predict_food(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    food_predictions = []
    for _, food_class, score in decoded_predictions:
        food_predictions.append((food_class.capitalize(), score))

    return food_predictions

# Other parts of your script
# ... (additional code or functionalities)

# Example usage:
if __name__ == "__main__":
    image_file = r'C:\\Users\\Vivek\\Desktop\\Training\\images.jpeg'
    predictions = predict_food(image_file)

    if predictions:
        print("Predicted Food Items:")
        for food_class, score in predictions:
            print(f"{food_class} ({score:.2f})")
    else:
        print("No predictions or error occurred while predicting.")

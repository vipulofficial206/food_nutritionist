from fastapi import FastAPI ,File ,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras

app = FastAPI()

Model = keras.layers.TFSMLayer("2", call_endpoint="serving_default")

class_names = ['apple_pie',
 'bread_pudding',
 'carrot_cake',
 'cheese_plate',
 'cheesecake',
 'chicken_curry',
 'chicken_quesadilla',
 'chicken_wings',
 'chocolate_cake',
 'chocolate_mousse',
 'club_sandwich',
 'cup_cakes',
 'deviled_eggs',
 'donuts',
 'dumplings',
 'eggs_benedict',
 'fish_and_chips',
 'french_fries',
 'french_toast',
 'fried_calamari',
 'fried_rice',
 'frozen_yogurt',
 'garlic_bread',
 'grilled_cheese_sandwich',
 'hamburger',
 'ice_cream',
 'lasagna',
 'lobster_bisque',
 'omelette',
 'onion_rings',
 'pancakes',
 'pizza',
 'red_velvet_cake',
 'risotto',
 'samosa',
 'spring_rolls',
 'strawberry_shortcake',
 'waffles']

@app.get('/ping')
async def ping():
    return "hello i am alive bob"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    # Resize to your model's input size
    img_resized = np.array(Image.fromarray(image).resize((224, 224)))

    # Ensure it's RGB (3 channels)
    if img_resized.ndim == 2:  # grayscale → convert to RGB
        img_resized = np.stack((img_resized,)*3, axis=-1)
    elif img_resized.shape[-1] == 4:  # RGBA → drop alpha
        img_resized = img_resized[..., :3]

    # Add batch dimension → (1,224,224,3)
    img_batch = np.expand_dims(img_resized, 0).astype("float32")

    # Run prediction
    prediction_dict = Model(img_batch)

    # Extract tensor from dict
    prediction = list(prediction_dict.values())[0].numpy()

    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=9087)
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow_hub as hub

# Load the model with custom objects
model = tf.keras.models.load_model("birdie.h5", custom_objects={'KerasLayer': hub.KerasLayer})


# Define the prediction function
def predict_bird(image):
    image = Image.fromarray(image).resize((224, 224))  # Resize as needed for your model
    image = np.array(image) / 255.0  # Normalize if required by your model
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return {"Bird Probability": prediction[0][0]}

# Set up Gradio interface
interface = gr.Interface(
    fn=predict_bird,
    inputs="image",
    outputs="label"
)

# Launch the app
if __name__ == "__main__":
    interface.launch()

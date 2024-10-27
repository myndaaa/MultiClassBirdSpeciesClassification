import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


model = tf.keras.models.load_model("birdie.h5")

class_labels = [...]  

def predict_bird_species(img):
   
    img = img.resize((224, 224))  
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    return {predicted_class: float(confidence)}

# Gradio interface
image_input = gr.inputs.Image(shape=(224, 224))
label_output = gr.outputs.Label(num_top_classes=5)

app = gr.Interface(
    fn=predict_bird_species, 
    inputs=image_input, 
    outputs=label_output, 
    title="Bird Species Classifier",
    description="Upload an image of a bird to predict its species."
)

app.launch()

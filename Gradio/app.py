import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow_hub as hub


model = tf.keras.models.load_model("birdie.h5", custom_objects={'KerasLayer': hub.KerasLayer})



def predict_bird(image):
    image = Image.fromarray(image).resize((224, 224))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return {"Bird Probability": prediction[0][0]}


interface = gr.Interface(
    fn=predict_bird,
    inputs="image",
    outputs="label"
)


if __name__ == "__main__":
    interface.launch()

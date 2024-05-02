from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request, redirect, url_for

import torch
# from diffusers import StableDiffusionPipeline
from transformers import SamModel, SamProcessor
from diffusers.utils import load_image
import base64
from io import BytesIO

# # Load model
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)
# pipe.to("cuda")

# Start flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)


# Load the SAM model and processor
try:
    model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50")
    model.to("cuda")
    processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")
    print("SAM model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading SAM model and processor: {e}")



@app.route('/')
def index():
  try:
    print("Initial page loaded!")
    img = load_image("https://picsum.photos/seed/picsum/200/300")
    print("Image generated! Converting image ...", img)

    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()
    img_bytes = base64.b64encode(img_bytes)
    img_bytes = img_bytes.decode("utf-8")
    print("Image converted! Sending image ...",img_bytes)
    return render_template('index.html', img_bytes=img_bytes)
  except Exception as e:
    print(f"Error loading initial---: {e}")
    return "Error loading initial page.======>"

if __name__ == '__main__':
    app.run()